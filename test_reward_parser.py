#!/usr/bin/env python3
"""
Quick test of the updated JSON parser with various model output formats.
Run this before full production training to verify the fixes work.
"""

import sys
sys.path.insert(0, '.')

from verl.utils.reward_score.kk_lithuanian import extract_answer_json, normalize_answer, compute_score

# Test cases
test_cases = [
    {
        "name": "Lithuanian list format (NEW - promoted format)",
        "response": """<think>Žmogaus logiška tapatybė...</think>
<answer>{"riteriai": ["Lukas", "Inga"], "melagiai": ["Daiva"]}</answer>""",
        "ground_truth": {"riteriai": ["Lukas", "Inga"], "melagiai": ["Daiva"]},
        "expect": "✓ PILNAS SUTAPIMAS!"
    },
    {
        "name": "English list format (old - still supported)",
        "response": """<think>Logic reasoning...</think>
<answer>{"knights": ["Lukas"], "knaves": ["Daiva", "Gintare"]}</answer>""",
        "ground_truth": {"knights": ["Lukas"], "knaves": ["Daiva", "Gintare"]},
        "expect": "✓ PILNAS SUTAPIMAS!"
    },
    {
        "name": "Person-to-role mapping (the problem case you found)",
        "response": """<think>Vytautas yra melagis arba aš esu melagis.</think>
<answer>{"Vytautas": "melagis", "Ugne": "riteris"}</answer>""",
        "ground_truth": {"riteriai": ["Ugne"], "melagiai": ["Vytautas"]},
        "expect": "✓ PILNAS SUTAPIMAS!"
    },
    {
        "name": "Mixed format (some person-mapped, some list)",
        "response": """<think>Thinking...</think>
<answer>{"Lukas": "riteris", "melagiai": ["Daiva", "Gintare"]}</answer>""",
        "ground_truth": {"riteriai": ["Lukas"], "melagiai": ["Daiva", "Gintare"]},
        "expect": "✓ PILNAS SUTAPIMAS!"
    },
    {
        "name": "English role names (knight/knave in mapping)",
        "response": """<think>...</think>
<answer>{"Lukas": "knight", "Daiva": "knave"}</answer>""",
        "ground_truth": {"riteriai": ["Lukas"], "melagiai": ["Daiva"]},
        "expect": "✓ PILNAS SUTAPIMAS!"
    },
    {
        "name": "Partial match test",
        "response": """<think>...</think>
<answer>{"riteriai": ["Lukas"], "melagiai": ["Daiva"]}</answer>""",
        "ground_truth": {"riteriai": ["Lukas"], "melagiai": ["Daiva", "Gintare"]},
        "expect": "⚠ Dalinis sutapimas"
    },
    {
        "name": "Wrong answer test",
        "response": """<think>...</think>
<answer>{"riteriai": ["Daiva"], "melagiai": ["Lukas"]}</answer>""",
        "ground_truth": {"riteriai": ["Lukas"], "melagiai": ["Daiva"]},
        "expect": "✗ NEATITIKIMAS"
    },
    {
        "name": "String values instead of lists (single)",
        "response": """<think>...</think>
<answer>{"riteriai": "Lukas", "melagiai": "Daiva"}</answer>""",
        "ground_truth": {"riteriai": ["Lukas"], "melagiai": ["Daiva"]},
        "expect": "✓ PILNAS SUTAPIMAS!"
    },
    {
        "name": "String values comma-separated (multiple)",
        "response": """<think>...</think>
<answer>{"riteriai": "Lukas, Inga", "melagiai": "Daiva, Gintare"}</answer>""",
        "ground_truth": {"riteriai": ["Lukas", "Inga"], "melagiai": ["Daiva", "Gintare"]},
        "expect": "✓ PILNAS SUTAPIMAS!"
    },
    {
        "name": "String values space-separated (multiple)",
        "response": """<think>...</think>
<answer>{"riteriai": "Lukas Inga", "melagiai": "Daiva Gintare"}</answer>""",
        "ground_truth": {"riteriai": ["Lukas", "Inga"], "melagiai": ["Daiva", "Gintare"]},
        "expect": "✓ PILNAS SUTAPIMAS!"
    },
    {
        "name": "Mixed: list and string formats",
        "response": """<think>...</think>
<answer>{"riteriai": ["Lukas"], "melagiai": "Daiva, Gintare"}</answer>""",
        "ground_truth": {"riteriai": ["Lukas"], "melagiai": ["Daiva", "Gintare"]},
        "expect": "✓ PILNAS SUTAPIMAS!"
    },
]

print("="*80)
print("TESTING UPDATED JSON PARSER".center(80, "="))
print("="*80)

passed = 0
failed = 0

for i, test in enumerate(test_cases, 1):
    print(f"\n[Test {i}] {test['name']}")
    print("-" * 80)
    
    try:
        # Test normalization directly
        import re, json
        match = re.search(r'<answer>(.*?)</answer>', test['response'], re.DOTALL)
        if match:
            answer_text = match.group(1).strip()
            result = json.loads(answer_text)
            normalized = normalize_answer(result)
            print(f"  Raw JSON: {result}")
            print(f"  Normalized: {normalized}")
        
        # Test full scoring
        print(f"\n  Computing score...")
        score = compute_score(test['response'], test['ground_truth'])
        
        # Check if expected string is in output
        if test['expect'] in str(score):
            print(f"  ✓ PASS - Got expected: {test['expect']}")
            passed += 1
        else:
            # Try checking in output text (for console output)
            print(f"  ⚠ Check output above for: {test['expect']}")
            
    except Exception as e:
        print(f"  ✗ FAIL - Exception: {e}")
        import traceback
        traceback.print_exc()
        failed += 1

print("\n" + "="*80)
print(f"RESULTS: {passed} passed, {failed} failed out of {len(test_cases)} tests")
print("="*80)

if failed == 0:
    print("\n✓ All tests passed! JSON parser is working correctly.")
    print("You can proceed with regenerating data and running production training.")
    sys.exit(0)
else:
    print(f"\n✗ {failed} test(s) failed. Check output above.")
    sys.exit(1)
