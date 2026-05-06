#!/usr/bin/env python3
"""
Test script for Qwen2.5-1.5B-Instruct Lithuanian RL training.
Checks: model loading, tokenization, Lithuanian prompt format, and inference.
"""

import os
import sys
import json
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

print("=" * 80)
print("QWEN2.5-1.5B-INSTRUCT LITHUANIAN RL TRAINING - COMPATIBILITY TEST")
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
# TEST 2: Tokenization with Lithuanian System Prompt
# ============================================================================
print("\n[TEST 2] Tokenizing with Lithuanian system prompt...")

# Lithuanian system prompt from kk_lithuanian.py (matching original KK format with thinking tags)
system_prompt = (
    "Tu esi pagalbinis asistentas. Asistentas pirmiausia galvoja apie samprotavimo procesą ir tada pateikia atsakymą. "
    "Samprotavimo procesas ir atsakymas yra uždaryti <think> </think> ir <answer> </answer> žymėse, atitinkamai, t.y., "
    "<think> samprotavimo procesas čia </think><answer> atsakymas čia </answer>. "
    "Dabar vartotojas prašo jūsų išspręsti loginės samprotavimo problemą. "
    "Pagalvojus, kai pagaliau padarote išvadą, aiškiai nurodykite kiekvieno personažo tapatybę <answer> </answer> žymėse. "
    "Nurodykite kiekvieno žmogaus tapatybę po vieną, pavyzdžiui, <answer> (1) Lukas yra riteris\\n(2) Daiva yra melagis\\n(3)... </answer>."
)
print(f"System prompt (with thinking tags):\n{system_prompt}\n")

# Lithuanian knights and knaves logic puzzle
user_prompt = (
    "Ypatingoje saloje gali gyventi tik dviejų tipų gyventojai: riteriai ir melagiai. Tarp jų gali būti, kad visi yra riteriai, visi yra melagiai, arba kai kurie yra riteriai, o kai kurie melagiai. "
    "Riteriai visada sako tiesą, o melagiai visada meluoja. Sutinkate 3 gyventojus: Lukas, Daiva ir Gintare. "
    "Lukas sako: Daiva visada meluoja. "
    "Gintare sako: Lukas yra melagis. "
    "Daiva sako: Gintare nėra tokio pat tipo kaip aš. "
    "Daiva sako: Gintare sako tiesą. "
    "Taigi kas yra riteris, o kas melagis?"
)
print(f"User prompt:\n{user_prompt}\n")

# Format as chat
messages = [
    {"role": "system", "content": system_prompt},
    {"role": "user", "content": user_prompt}
]

try:
    import traceback
    
    # Apply chat template and tokenize
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
    print(f"\n[Decoded Chat Template (first 500 chars)]:")
    print("-" * 60)
    print(decoded[:500] + "..." if len(decoded) > 500 else decoded)
    print("-" * 60)
    
except Exception as e:
    print(f"✗ Tokenization failed: {e}")
    print("\nFull traceback:")
    traceback.print_exc()
    sys.exit(1)

# ============================================================================
# TEST 3: Generate a Sample Response (inference only)
# ============================================================================
print("\n[TEST 3] Generating sample response (inference only)...")
try:
    with torch.no_grad():
        generated_ids = model.generate(
            **model_inputs,
            max_new_tokens=4000,
            temperature=0.7,
            top_p=0.95,
        )
    
    # Remove input tokens to get only generated output
    generated_ids = [
        output_ids[len(input_ids):] 
        for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]
    
    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    print(f"✓ Generation successful")
    print(f"\n[Sample Response]:")
    print("-" * 60)
    print(response)
    print("-" * 60)
    
except Exception as e:
    print(f"✗ Generation failed: {e}")
    print("\nFull traceback:")
    traceback.print_exc()
    sys.exit(1)

# ============================================================================
# TEST 4: Check for Expected Tags in Output
# ============================================================================
print("\n[TEST 4] Analyzing response structure...")
response_text_lower = response.lower()  # for case-insensitive search

has_think_open = "<think>" in response_text_lower
has_think_close = "</think>" in response_text_lower
has_answer_open = "<answer>" in response_text_lower
has_answer_close = "</answer>" in response_text_lower

print(f"  Contains <think>: {has_think_open}")
print(f"  Contains </think>: {has_think_close}")
print(f"  Contains <answer>: {has_answer_open}")
print(f"  Contains </answer>: {has_answer_close}")

if has_think_open and has_think_close and has_answer_open and has_answer_close:
    print("✓ Response contains expected <think> and <answer> tags!")
    # Extract thinking content
    try:
        think_start = response_text_lower.find("<think>") + len("<think>")
        think_end = response_text_lower.find("</think>")
        if think_start > len("<think>") - 1 and think_end > think_start:
            think_content = response[think_start:think_end].strip()
            print(f"\n[Thinking Process]:\n")
            print("-" * 60)
            print(think_content[:200] + "..." if len(think_content) > 200 else think_content)
            print("-" * 60)
    except Exception as e:
        print(f"  ⚠ Could not extract thinking content: {e}")
    
    # Extract answer content
    try:
        answer_start = response_text_lower.find("<answer>") + len("<answer>")
        answer_end = response_text_lower.find("</answer>")
        if answer_start > len("<answer>") - 1 and answer_end > answer_start:
            answer_content = response[answer_start:answer_end].strip()
            print(f"\n[Final Answer]:\n")
            print("-" * 60)
            print(answer_content)
            print("-" * 60)
    except Exception as e:
        print(f"  ⚠ Could not extract answer content: {e}")
else:
    print(f"⚠ Response missing expected tags:")
    print(f"   <think> tags: {has_think_open and has_think_close}")
    print(f"   <answer> tags: {has_answer_open and has_answer_close}")
    print("   Check prompt instructions and system message.")

# ============================================================================
# TEST 5: Reward Function Parse Test
# ============================================================================
print("\n[TEST 5] Testing Lithuanian reward function...")
try:
    from verl.utils.reward_score import kk_lithuanian
    
    # Ground truth for the puzzle
    ground_truth = {
        "solution_text_format": "(1) Lukas yra riteris\n(2) Daiva yra melagis\n(3) Gintare yra melagis",
        "statements": [
            "Lukas sako: Daiva visada meluoja",
            "Gintare sako: Lukas yra melagis",
            "Daiva sako: Gintare nėra tokio pat tipo kaip aš",
            "Daiva sako: Gintare sako tiesą"
        ]
    }
    
    score = kk_lithuanian.compute_score(response, ground_truth)
    print(f"✓ Reward function executed successfully")
    print(f"  Score: {score:.4f}")
    
except ImportError as e:
    print(f"⚠ Reward function not available (expected if not in training env): {e}")
except Exception as e:
    print(f"⚠ Reward function test failed: {e}")

# ============================================================================
# TEST 6: Chat Template Format Verification
# ============================================================================
print("\n[TEST 6] Verifying Qwen2.5 chat template format...")
try:
    # Check if template uses im_start/im_end (Qwen standard)
    template_str = str(result)
    has_im_tags = "<|im_start|>" in template_str and "<|im_end|>" in template_str
    has_assistant_tag = "<|im_start|>assistant" in template_str
    
    if has_im_tags and has_assistant_tag:
        print("✓ Qwen2.5 chat template format verified")
        print(f"  Uses Qwen IM_START/IM_END format: {has_im_tags}")
        print(f"  Has assistant role: {has_assistant_tag}")
    else:
        print("⚠ Unexpected chat template format")
        
except Exception as e:
    print(f"⚠ Could not verify template format: {e}")


print("=" * 80)
print("✓ All Qwen2.5 Lithuanian compatibility tests completed!")
print("=" * 80)
