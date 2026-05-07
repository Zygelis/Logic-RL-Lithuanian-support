import re
import json
from typing import Dict, Optional, Set

def normalize_answer(result: Dict) -> Optional[Dict]:
    """Normalize various JSON formats to standard format.
    
    Supports:
    1. List format: {"riteriai": ["Lukas"], "melagiai": ["Daiva"]}
    2. List format (English): {"knights": ["Lukas"], "knaves": ["Daiva"]}
    3. String values (single): {"riteriai": "Lukas", "melagiai": "Daiva"}
    4. String values (comma/space-separated): {"melagiai": "Lukas, Simas"} or {"melagiai": "Lukas Simas"}
    5. Person-to-role mapping: {"Lukas": "riteris", "Daiva": "melagis"}
    6. Mixed: {"Lukas": "riteris", "melagiai": ["Daiva"]}
    
    Returns:
        Normalized dict with keys 'riteriai' and 'melagiai' (lists)
    """
    if not isinstance(result, dict):
        return None
    
    # Initialize result
    riteriai = set()
    melagiai = set()
    
    def parse_names(value):
        """Parse names from various formats (string, list, comma-separated, space-separated)."""
        names = set()
        
        if isinstance(value, list):
            # List of names
            names.update([str(v).strip() for v in value if v])
        elif isinstance(value, str):
            # String value - try to parse as comma-separated or space-separated
            value = value.strip()
            if not value:
                return names
            
            # Try comma-separated first (higher priority)
            if ',' in value:
                names.update([v.strip() for v in value.split(',') if v.strip()])
            else:
                # Try space-separated (for multiple words)
                # But be careful not to split "John Smith" into separate people
                # Heuristic: if all parts are capitalized or single words, might be space-separated
                parts = value.split()
                if len(parts) > 1 and all(len(part) <= 10 for part in parts):
                    # Likely multiple short names
                    names.update(parts)
                else:
                    # Single name (possibly multi-word like "John Smith")
                    names.add(value)
        
        return names
    
    for key, value in result.items():
        key_lower = key.lower()
        value_lower = str(value).lower() if value else ""
        
        # Handle list/string formats: "riteriai", "melagiai", "knights", "knaves"
        if key_lower in ['riteriai', 'knights']:
            riteriai.update(parse_names(value))
        elif key_lower in ['melagiai', 'knaves']:
            melagiai.update(parse_names(value))
        
        # Handle person-to-role mapping: key=person_name, value="riteris"/"melagis"/"knight"/"knave"
        elif isinstance(value, str) and value_lower in ['riteris', 'knight', 'riteriu', 'riteriaus']:
            riteriai.add(key)
        elif isinstance(value, str) and value_lower in ['melagis', 'knave', 'melagio', 'melagius']:
            melagiai.add(key)
    
    # If we found any assignments, return normalized format
    if riteriai or melagiai:
        return {
            'riteriai': list(riteriai),
            'melagiai': list(melagiai)
        }
    
    return None

def extract_answer_json(solution_str: str) -> Optional[Dict]:
    """Extract and parse JSON from <answer> tags with flexible format support.
    
    Args:
        solution_str: Raw model response
        
    Returns:
        Normalized dict with 'riteriai' and 'melagiai' keys, or None if extraction fails
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
        # Normalize to standard format
        normalized = normalize_answer(result)
        if normalized:
            print(f"✓ JSON išanalizuotas sėkmingai: {normalized}")
            return normalized
        else:
            print(f"[Error] JSON neturinys neatitinka jokio žinomo formato: {result}")
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
        ground_truth: Dict with 'riteriai' and 'melagiai' (or 'knights'/'knaves') lists
        format_reward: Points for correct format
        answer_reward: Points for correct answer
        
    Returns:
        Total score
    """
    print("\n" + "="*80)
    print(" Vertinimas ".center(80, '='))
    
    # Normalize ground truth (handle both Lithuanian and English keys)
    gt_riteriai = set(ground_truth.get('riteriai', ground_truth.get('knights', [])))
    gt_melagiai = set(ground_truth.get('melagiai', ground_truth.get('knaves', [])))
    
    print(f"\n[Pagrindinė tiesa]")
    print(f"  Riteriai: {list(gt_riteriai)}")
    print(f"  Melagiai: {list(gt_melagiai)}")
    
    # Validate structure
    format_valid = validate_structure(solution_str)
    format_score = format_reward if format_valid else -abs(format_reward)
    print(f"  Formato taškai: {format_score}")
    
    # Extract and parse JSON
    answer_score = 0
    if format_valid:
        parsed = extract_answer_json(solution_str)
        if parsed:
            pred_riteriai = set(parsed.get('riteriai', []))
            pred_melagiai = set(parsed.get('melagiai', []))
            
            print(f"\n[Modelio atsakymas]")
            print(f"  Riteriai: {list(pred_riteriai)}")
            print(f"  Melagiai: {list(pred_melagiai)}")
            
            # Compare
            if pred_riteriai == gt_riteriai and pred_melagiai == gt_melagiai:
                answer_score = answer_reward
                print(f"  ✓ PILNAS SUTAPIMAS!")
            else:
                # Check partial match
                riteriai_correct = len(pred_riteriai & gt_riteriai) == len(gt_riteriai) if gt_riteriai else True
                melagiai_correct = len(pred_melagiai & gt_melagiai) == len(gt_melagiai) if gt_melagiai else True
                
                if riteriai_correct or melagiai_correct:
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

