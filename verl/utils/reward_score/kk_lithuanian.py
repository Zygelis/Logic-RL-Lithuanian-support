import re
import json
from typing import Dict, Optional

def extract_answer_json(solution_str: str) -> Optional[Dict]:
    """Extract and parse JSON from <answer> tags.
    
    Args:
        solution_str: Raw model response
        
    Returns:
        Parsed JSON dict or None if extraction fails
    """
    # Find <answer> tags
    match = re.search(r'<answer>(.*?)</answer>', solution_str, re.DOTALL | re.IGNORECASE)
    if not match:
        print("[Error] Nerasti atsakymo žymės <answer>")
        return None
    
    answer_text = match.group(1).strip()
    
    # Try to parse as JSON
    try:
        result = json.loads(answer_text)
        if isinstance(result, dict) and 'knights' in result and 'knaves' in result:
            print(f"✓ JSON išanalizuotas sėkmingai: {result}")
            return result
        else:
            print(f"[Error] JSON neturi reikalingų raktų 'knights' ir 'knaves'")
            return None
    except json.JSONDecodeError as e:
        print(f"[Error] JSON išanalizavimo klaida: {e}")
        print(f"  Bandytas tekstas: {answer_text[:100]}...")
        return None


def validate_structure(solution_str: str) -> bool:
    """Validate that response has required <think> and <answer> tags in correct order.
    
    Args:
        solution_str: Model response
        
    Returns:
        True if structure is valid
    """
    has_think_open = '<think>' in solution_str
    has_think_close = '</think>' in solution_str
    has_answer_open = '<answer>' in solution_str
    has_answer_close = '</answer>' in solution_str
    
    if not (has_think_open and has_think_close and has_answer_open and has_answer_close):
        print("[Formato klaida] Trūksta reikalingų žymių (<think>, </think>, <answer>, </answer>)")
        return False
    
    # Check order
    think_open_pos = solution_str.find('<think>')
    think_close_pos = solution_str.find('</think>')
    answer_open_pos = solution_str.find('<answer>')
    answer_close_pos = solution_str.find('</answer>')
    
    if not (think_open_pos < think_close_pos < answer_open_pos < answer_close_pos):
        print("[Formato klaida] Netinkama žymių tvarka")
        return False
    
    print("✓ Struktūra validna: <think>...</think><answer>...</answer>")
    return True


def compute_score(solution_str: str, 
                 ground_truth: Dict,
                 format_reward: float = 1.0,
                 answer_reward: float = 2.0) -> float:
    """
    Compute score for model response.
    
    Scoring:
    - Format validation: +1.0 or -1.0
    - Answer correctness: +2.0 (full match), -1.5 (partial), -2.0 (wrong/missing)
    - Total range: [-3.0, 3.0]
    
    Args:
        solution_str: Raw model response
        ground_truth: Dict with 'knights' and 'knaves' lists
        format_reward: Points for correct format
        answer_reward: Points for correct answer
        
    Returns:
        Total score
    """
    print("\n" + "="*80)
    print(" Vertinimas ".center(80, '='))
    
    gt_knights = set(ground_truth.get('knights', []))
    gt_knaves = set(ground_truth.get('knaves', []))
    
    print(f"\n[Pagrindinė tiesa]")
    print(f"  Riteriai: {list(gt_knights)}")
    print(f"  Melagiai: {list(gt_knaves)}")
    
    # Validate structure
    format_valid = validate_structure(solution_str)
    format_score = format_reward if format_valid else -abs(format_reward)
    print(f"  Formato taškai: {format_score}")
    
    # Extract and parse JSON
    answer_score = 0
    if format_valid:
        parsed = extract_answer_json(solution_str)
        if parsed:
            pred_knights = set(parsed.get('knights', []))
            pred_knaves = set(parsed.get('knaves', []))
            
            print(f"\n[Modelio atsakymas]")
            print(f"  Riteriai: {list(pred_knights)}")
            print(f"  Melagiai: {list(pred_knaves)}")
            
            # Compare
            if pred_knights == gt_knights and pred_knaves == gt_knaves:
                answer_score = answer_reward
                print(f"  ✓ PILNAS SUTAPIMAS!")
            else:
                # Check partial match
                knights_correct = len(pred_knights & gt_knights) == len(gt_knights)
                knaves_correct = len(pred_knaves & gt_knaves) == len(gt_knaves)
                
                if knights_correct or knaves_correct:
                    answer_score = -1.5
                    print(f"  ⚠ Dalinis sutapimas")
                else:
                    answer_score = -2.0
                    print(f"  ✗ NEATITIKIMAS")
        else:
            answer_score = -2.0
            print(f"\n  ✗ Nepavyko išanalizuoti JSON")
    else:
        answer_score = -2.0
        print(f"\n  ✗ Praleista dėl formato klaidų")
    
    total_score = format_score + answer_score
    
    print("\n" + "-"*80)
    print(f" Rezultatas ".center(80, '-'))
    print(f"  Formatas: {format_score}")
    print(f"  Turinys: {answer_score}")
    print(f"  Iš viso: {total_score}")
    print("="*80 + "\n")
    
    return total_score

