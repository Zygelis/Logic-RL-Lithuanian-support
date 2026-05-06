""" Preprocess Lithuanian Knights and Knaves dataset """

import os
from datasets import Dataset
from verl.utils.hdfs_io import copy, makedirs
import argparse
import json


def make_prefix(dp, template_type):
    """Create prompt with system message and user question in Lithuanian."""
    prompt_text = dp['prompt']
    system_prompt = (
        "Tu esi pagalbinis asistentas. Asistentas pirmiausia galvoja apie samprotavimo procesą ir tada pateikia atsakymą. "
        "Samprotavimo procesas ir atsakymas yra uždaryti <think> </think> ir <answer> </answer> žymėse, atitinkamai, t.y., "
        "<think> samprotavimo procesas čia </think><answer> atsakymas čia </answer>. "
        "Dabar vartotojas prašo jūsų išspręsti loginės samprotavimo problemą. "
        "Pagalvojus, kai pagaliau padarote išvadą, aiškiai nurodykite kiekvieno personažo tapatybę <answer> </answer> žymėse "
        "JSON formato pavidalu, pvz.: <answer>{\"knights\": [\"Lukas\"], \"knaves\": [\"Daiva\", \"Gintare\"]}</answer>."
    )
    
    if template_type == 'base':
        prefix = f"{system_prompt}\n\nVartotojas: {prompt_text}\nAsistentas: <think>"
    elif template_type == 'qwen-instruct':
        prefix = f"""<|im_start|>system
{system_prompt}
<|im_end|>
<|im_start|>user
{prompt_text}
<|im_end|>
<|im_start|>assistant
<think>"""
    else:
        raise ValueError(f"Unknown template_type: {template_type}")
    
    return prefix


def create_solution_dict(dp, statements):
    """Create ground truth solution dictionary from answer field.
    
    Args:
        dp: answer dict with 'knights' and 'knaves' keys
        statements: list of statements (from top level of example)
    """
    knights = dp.get('knights', [])
    knaves = dp.get('knaves', [])
    
    return {
        "knights": knights,
        "knaves": knaves,
        "statements": statements,
    }


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--local_dir', default='./data/kk/instruct/lithuanian')
    parser.add_argument('--hdfs_dir', default=None)
    parser.add_argument('--data_path', default='./lithuanian_dataset.jsonl')
    parser.add_argument('--train_size', type=int, default=900)
    parser.add_argument('--test_size', type=int, default=100)
    parser.add_argument('--template_type', type=str, default='qwen-instruct')
    
    args = parser.parse_args()
    
    data_source = 'kk_logic_lithuanian'
    TRAIN_SIZE = args.train_size
    TEST_SIZE = args.test_size

    # Load custom JSONL dataset
    def gen_from_jsonl(path):
        with open(path) as f:
            for line in f:
                if line.strip():
                    yield json.loads(line)
    
    raw_dataset = Dataset.from_generator(gen_from_jsonl, gen_kwargs={'path': args.data_path})
    print(f"Loaded {len(raw_dataset)} examples from {args.data_path}")

    if len(raw_dataset) < TRAIN_SIZE + TEST_SIZE:
        print(f"Warning: Dataset has {len(raw_dataset)} examples, but requested {TRAIN_SIZE + TEST_SIZE}")
        TRAIN_SIZE = len(raw_dataset) - TEST_SIZE
        print(f"Adjusted to TRAIN_SIZE={TRAIN_SIZE}, TEST_SIZE={TEST_SIZE}")
    
    train_dataset = raw_dataset.select(range(TRAIN_SIZE))
    test_dataset = raw_dataset.select(range(TRAIN_SIZE, TRAIN_SIZE + TEST_SIZE)) if TEST_SIZE > 0 else None

    def make_map_fn(split):
        def process_fn(example, idx):
            question = make_prefix(example, template_type=args.template_type)
            solution = create_solution_dict(example.get('answer', {}), example.get('statements', []))
            
            data = {
                "data_source": data_source,
                "prompt": [{
                    "role": "user",
                    "content": question,
                }],
                "ability": "logic",
                "reward_model": {
                    "style": "rule",
                    "ground_truth": solution
                },
                "extra_info": {
                    'split': split,
                    'index': idx,
                    'difficulty': example.get('difficulty', 'unknown'),
                    'id': example.get('id', idx),
                }
            }
            return data
        return process_fn

    train_dataset = train_dataset.map(function=make_map_fn('train'), with_indices=True)
    if test_dataset is not None:
        test_dataset = test_dataset.map(function=make_map_fn('test'), with_indices=True)

    local_dir = args.local_dir
    hdfs_dir = args.hdfs_dir

    # Create local directory if not exists
    os.makedirs(os.path.expanduser(local_dir), exist_ok=True)

    print(f"\nSaving train dataset to {os.path.join(local_dir, 'train.parquet')}")
    train_dataset.to_parquet(os.path.join(local_dir, 'train.parquet'))
    
    if test_dataset is not None:
        print(f"Saving test dataset to {os.path.join(local_dir, 'test.parquet')}")
        test_dataset.to_parquet(os.path.join(local_dir, 'test.parquet'))

    if hdfs_dir is not None:
        makedirs(hdfs_dir)
        copy(src=local_dir, dst=hdfs_dir)
        print(f"Copied to HDFS: {hdfs_dir}")
    
    print("✓ Data preprocessing complete!")
