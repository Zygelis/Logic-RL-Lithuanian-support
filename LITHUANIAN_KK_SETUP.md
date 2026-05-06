# Lithuanian KK Dataset Integration Guide

## Overview

This guide helps you adapt the KK (Knights and Knaves) RL training repo to work with a **Lithuanian dataset**. The key changes:

1. ✅ **JSON output format** - Model outputs structured JSON instead of text
2. ✅ **Language-agnostic reward function** - Works with any language
3. ✅ **Data preprocessing** - Adapted for Lithuanian language

---

## What Changed

### 1. **Preprocessing** (`examples/data_preprocess/kk_lithuanian.py`)
- **Now requests JSON format in prompts**
- Ground truth stores knights/knaves as lists (not text)
- Simpler, more structured data format

Example prompt now asks for:
```json
{"knights": ["Lukas"], "knaves": ["Daiva", "Gintare"]}
```

### 2. **Reward Function** (`verl/utils/reward_score/kk_lithuanian.py`)
- **Completely rewritten - much simpler**
- Extracts JSON from `<answer>` tags
- Compares directly against ground truth lists
- **Language-independent** (no Lithuanian keyword hardcoding)

Scoring:
- **Format validation**: +1.0 for correct `<think>...</think><answer>...</answer>` tags
- **Answer correctness**: 
  - +2.0 for perfect match
  - -1.5 for partial match
  - -2.0 for mismatch/parse error
- **Total range**: [-3.0, 3.0]

### 3. **Evaluation Integration** (`verl/trainer/main_eval.py`)
- Registered `kk_lithuanian` reward function for `data_source='kk_logic_lithuanian'`

---

## Dataset Format

Your Lithuanian dataset must be JSONL with this structure:

```json
{
  "id": 1,
  "difficulty": "lengvas",
  "islanders": ["Lukas", "Daiva", "Gintare"],
  "statements": ["Lukas sako: ...", "Gintare sako: ..."],
  "answer": {
    "knights": ["Lukas"],
    "knaves": ["Daiva", "Gintare"]
  },
  "prompt": "Ypatingoje saloje gali gyventi...",
  "system_prompt": "Tu esi pagalbinis asistentas..."
}
```

**Required fields:**
- `prompt` - The problem statement (Lithuanian text)
- `answer.knights` - List of character names who are knights
- `answer.knaves` - List of character names who are knaves
- `statements` - List of statements (used in ground truth)

---

## Quick Start: Prepare Data & Train

### Step 1: Prepare your JSONL dataset
Place your Lithuanian KK examples in `your_dataset.jsonl` (JSON lines format, one example per line).

Required fields per example:
```json
{
  "prompt": "Ypatingoje saloje...",
  "answer": {
    "knights": ["Lukas"],
    "knaves": ["Daiva", "Gintare"]
  },
  "statements": ["Lukas sako: ..."]
}
```

### Step 2: Preprocess

```bash
python3 examples/data_preprocess/kk_lithuanian.py \
    --local_dir ./data/kk/instruct/lithuanian \
    --data_path ./your_dataset.jsonl \
    --train_size 10 \
    --test_size 3 \
    --template_type qwen-instruct
```

Output:
- `./data/kk/instruct/lithuanian/train.parquet`
- `./data/kk/instruct/lithuanian/test.parquet`

### Step 3: Train (Reinforce++)

```bash
chmod +x scripts/train_kk_lithuanian_reinforce_plus_plus.sh
./scripts/train_kk_lithuanian_reinforce_plus_plus.sh
```

Or modify the script for your GPU count/hyperparameters.

---

## Manual Steps (If Not Using Script)

### 1. Preprocess your data

```bash
python3 examples/data_preprocess/kk_lithuanian.py \
    --local_dir ./data/kk/instruct/lithuanian \
    --data_path ./your_dataset.jsonl \
    --train_size 900 \
    --test_size 100 \
    --template_type qwen-instruct
```

Output:
- `./data/kk/instruct/lithuanian/train.parquet`
- `./data/kk/instruct/lithuanian/test.parquet`

### 2. Run GRPO training

```bash
python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files=./data/kk/instruct/lithuanian/train.parquet \
    data.val_files=./data/kk/instruct/lithuanian/test.parquet \
    data.train_batch_size=64 \
    data.val_batch_size=32 \
    data.max_prompt_length=512 \
    data.max_response_length=256 \
    actor_rollout_ref.model.path="Qwen/Qwen2-7B-Instruct" \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.rollout.n=1 \
    trainer.total_epochs=1 \
    trainer.n_gpus_per_node=4 \
    trainer.nnodes=1 \
    trainer.project_name='verl_kk_lithuanian'
```

### 3. Evaluate results

```bash
python3 -m verl.trainer.main_eval \
    data.path=./outputs/your_model/generation_samples.parquet \
    data.prompt_key=prompt \
    data.response_key=responses \
    data.data_source_key=data_source \
    data.reward_model_key=reward_model
```

---

## Verification: Check Reward Function

Test the reward function standalone:

```python
from verl.utils.reward_score import kk_lithuanian

# Ground truth
gt = {
    "knights": ["Lukas"],
    "knaves": ["Daiva", "Gintare"]
}

# Model response
response = """
<think>
Let me think through this step by step...
</think>
<answer>{"knights": ["Lukas"], "knaves": ["Daiva", "Gintare"]}</answer>
"""

score = kk_lithuanian.compute_score(response, gt)
print(f"Score: {score}")  # Expected: 3.0 (format +1.0 + answer +2.0)
```

---

## Troubleshooting

### ❌ "Nerasti jokio atsakymo žymės" (No answer tags found)

**Problem:** Model output doesn't have `<answer>` tags

**Solution:** 
- Check system prompt is requesting JSON in answer tags
- Verify model follows instruction format
- Lower temperature for generation

### ❌ "JSON neturi reikalingų raktų" (JSON missing keys)

**Problem:** JSON doesn't have `knights` and `knaves` keys

**Solution:**
- Update prompt to explicitly request JSON structure
- Example: `<answer>{"knights": [...], "knaves": [...]}</answer>`

### ❌ Score is always -2.0

**Problem:** Answer parsing fails

**Solution:**
- Ensure format is valid (structure validation pass)
- Check JSON is valid with `json.loads()`
- Verify knights/knaves lists contain full names

---

## Next Steps

1. **Scale up**: Increase train_size/test_size
2. **Tune hyperparameters**: Adjust KL loss, learning rate, batch size
3. **Use larger model**: Try Qwen2-7B or Qwen2.5-14B
4. **Monitor training**: Check wandb project for reward curves
5. **Evaluate**: Compare before/after model performance

---

## Files Modified/Created

| File | Change |
|------|--------|
| `examples/data_preprocess/kk_lithuanian.py` | Updated to use JSON format |
| `verl/utils/reward_score/kk_lithuanian.py` | Rewrote for JSON + language-agnostic |
| `verl/trainer/main_eval.py` | Registered Lithuanian reward function |
| `train_kk_lithuanian_test.sh` | New: Quick test script |
| `lithuanian_test_mini.jsonl` | New: Test dataset |

---

## Questions?

Check these files for reference:
- Original English KK: `verl/utils/reward_score/kk.py`
- English preprocessing: `examples/data_preprocess/kk.py`
- Training config: `verl/trainer/config/ppo_trainer.yaml`
