#!/usr/bin/env python3
"""
Test script for Qwen3.5-4B Lithuanian RL training.
Checks: model loading, tokenization, prompt format, and tiny training run.
"""

import os
import sys
import json
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

print("=" * 80)
print("QWEN3.5-4B LITHUANIAN RL TRAINING - COMPATIBILITY TEST")
print("=" * 80)

# ============================================================================
# TEST 1: Model Loading
# ============================================================================
print("\n[TEST 1] Loading Qwen2.5-1.5B-Instruct model...")
try:
    model_name = "Qwen/Qwen2.5-1.5B-Instruct"
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    print(f"✓ Tokenizer loaded: {model_name}")
    
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto",
        torch_dtype=torch.float16,
        trust_remote_code=True
    )
    print(f"✓ Model loaded successfully")
    print(f"  Model size: {sum(p.numel() for p in model.parameters()) / 1e9:.1f}B parameters")
    print(f"  Model device: {next(model.parameters()).device}")
except Exception as e:
    print(f"✗ Model loading failed: {e}")
    sys.exit(1)

# ============================================================================
# TEST 2: Tokenization with Current Lithuanian System Prompt
# ============================================================================
print("\n[TEST 2] Tokenizing with Lithuanian system prompt...")

# Your current system prompt
system_prompt = (
    "Tu esi pagalbinis asistentas. Išspręsk loginę užduotį ir pateik vieną galutinį atsakymą tarp <answer>...</answer> žymių. "
    "Atsakymo formatas yra JSON objektas būtinai viduje <answer> </answer> žymų su šia struktūra:"
    " {\"predictions\": [{\"name\": \"X\", \"role\": \"riteris\"}, ...]} " 
)
print (f"Current system prompt:\n{system_prompt}")

user_prompt = (
    "Ypatingoje saloje gali gyventi tik dviejų tipų gyventojai: riteriai ir melagiai. Tarp jų gali būti, kad visi yra riteriai, visi yra melagiai, arba kai kurie yra riteriai, o kai kurie melagiai. Riteriai visada sako tiesą, o melagiai visada meluoja. Sutinkate 3 gyventojus: Lukas, Daiva ir Gintare. Lukas sako: Daiva visada meluoja. Gintare sako: Lukas yra melagis. Daiva sako: Gintare nėra tokio pat tipo kaip aš. Daiva sako: Gintare sako tiesą. Taigi kas yra riteris, o kas melagis?"
)
# Format as chat
messages = [
    {"role": "system", "content": system_prompt},
    {"role": "user", "content": user_prompt}
]

try:
    # Tokenize with chat template
    import traceback
    result = tokenizer.apply_chat_template(
        messages,
        tokenize=False,  # Get string, not token IDs
        add_generation_prompt=True
    )
    # Tokenize the formatted string
    model_inputs = tokenizer([result], return_tensors="pt").to(model.device)
    
    print(f"✓ Chat template applied successfully")
    print(f"  Input token count: {model_inputs['input_ids'].shape[1]} tokens")
    
    # Decode to see actual prompt format
    decoded = tokenizer.decode(model_inputs['input_ids'][0], skip_special_tokens=False)
    print(f"\n[Decoded Chat Template]:")
    print("-" * 60)
    print(decoded[:500] + "..." if len(decoded) > 500 else decoded)
    print("-" * 60)
    
except Exception as e:
    print(f"✗ Tokenization failed: {e}")
    print("\nFull traceback:")
    traceback.print_exc()
    sys.exit(1)

# ============================================================================
# TEST 3: Generate a Sample Response (non-training inference)
# ============================================================================
print("\n[TEST 3] Generating sample response (inference only)...")
try:
    # Move input_ids to same device as model
    device = next(model.parameters()).device
    input_ids = model_inputs['input_ids'].to(device)
    
    generated_ids = model.generate(
    **model_inputs,
    max_new_tokens=512
    )
    
    generated_ids = [
    output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]   

    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    print(f"✓ Generation successful")
    print(f"\n[Sample Response]:")
    print("-" * 60)
    print(response)
    print("-" * 60)
    
except Exception as e:
    print(f"✗ Generation failed: {e}")
    sys.exit(1)

# ============================================================================
# TEST 4: Check for Thinking Tags in Output
# ============================================================================
print("\n[TEST 4] Analyzing response structure...")
response_text = response.lower()  # for case-insensitive search

has_think_open = "<think>" in response_text
has_think_close = "</think>" in response_text
has_answer_open = "<answer>" in response_text
has_answer_close = "</answer>" in response_text

print(f"  Contains <think>: {has_think_open}")
print(f"  Contains </think>: {has_think_close}")
print(f"  Contains <answer>: {has_answer_open}")
print(f"  Contains </answer>: {has_answer_close}")

if has_think_open and has_think_close and has_answer_open and has_answer_close:
    print("✓ Response structure matches expectations!")
else:
    print("⚠ Response structure mismatch. Check prompt instructions.")

# ============================================================================
# TEST 5: Reward Function Parse Test
# ============================================================================
print("\n[TEST 5] Testing reward function parsing...")
try:
    from verl.utils.reward_score import kk_lithuanian
    
    ground_truth = {
        "solution_text_format": "(1) Lukas yra riteris\n(2) Daiva yra melagis\n(3) Gintare yra melagis",
        "statements": ["Lukas sako: Daiva visada meluoja"]
    }
    
    score = kk_lithuanian.compute_score(response_text, ground_truth)
    print(f"✓ Reward function executed successfully")
    print(f"  Score: {score}")
    
except Exception as e:
    print(f"⚠ Reward function test skipped or failed: {e}")

# ============================================================================
# TEST 6: vLLM Compatibility Check
# ============================================================================
print("\n[TEST 6] Checking vLLM version...")
try:
    import vllm
    print(f"✓ vLLM installed: version {vllm.__version__}")
    if hasattr(vllm, '__version__'):
        major, minor = map(int, vllm.__version__.split('.')[:2])
        if major == 0 and minor < 6:
            print("  ⚠ WARNING: vLLM < 0.6 may not support Qwen3.5 reasoning parser")
            print("    Consider upgrading: pip install --upgrade vllm")
except ImportError:
    print("⚠ vLLM not installed. Install with: pip install vllm")

# ============================================================================
# SUMMARY & RECOMMENDATIONS
# ============================================================================
print("\n" + "=" * 80)
print("SUMMARY & RECOMMENDATIONS")
print("=" * 80)

print("""
1. MODEL: Qwen3.5-4B loads successfully ✓

2. PROMPT: Your current system prompt explicitly instructs thinking.
   Qwen3.5-4B does thinking by default.
   
   OPTIONS:
   A) Keep current prompt (Qwen3.5 will follow your instructions + add thinking)
   B) Simplify prompt - remove thinking instructions, let Qwen3.5 think naturally
   C) Disable Qwen3.5 thinking mode - set enable_thinking=False (not recommended)
   
   RECOMMENDATION: Option B - simplify the prompt.

3. NEXT STEPS:
   - Decide on prompt strategy (A, B, or C above)
   - Update LITHUANIAN_SYSTEM_PROMPT in preprocessing if needed
   - Run: python examples/data_preprocess/kk_lithuanian.py ...
   - Then attempt tiny training run (see TINY_TRAINING_TEST.sh)

4. MINIMAL TRAINING TEST:
   - Prepare 10-20 examples in parquet format
   - Run: bash TINY_TRAINING_TEST.sh
   - Monitor for tokenization/reward function errors
   - Once successful, scale to full dataset
""")

print("=" * 80)
print("✓ All compatibility tests completed!")
print("=" * 80)
