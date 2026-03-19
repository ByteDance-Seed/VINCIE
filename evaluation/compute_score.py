import openai
import json
import pandas as pd
import os
import zipfile
import yaml
from tqdm import tqdm
import base64
from io import BytesIO
from PIL import Image
import json
import random
import argparse
import multiprocessing
from functools import partial
import time
import copy
import traceback
import math
import csv
from datasets import load_dataset

from utils import parse_mixed_string, session_img_path_to_base64

NUM_TRY = 20

def save_json(data, json_path):
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)


def read_json(json_path):
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data

# prompt template
prompt_template = \
""" Assume you are an expert in evaluating multi-turn image editing. In this task, a user interacts with an image editing system across multiple turns. At the first turn, the user provides a source image and an editing prompt. The system returns the edited image. In each subsequent turn, the user supplies a new prompt, and the system generates a new image based on the output from the previous turn.
Your goal is to evaluate how successfully the editing instruction of the LAST turn (turn-{num_prompts}) has been executed. 

You will be given {num_prompts} user editing prompts and {num_images} images: the first image is the original source image, and the next are the edited results from each turn for each prompt.
You should focus more on the last prompt and the last edited image, but you may also consider the previous prompts and images as context. 

The {num_prompts} user editing prompts are: {editing_prompt}

Please follow these evaluation rules:
1) Last-turn Evaluation: For the last turn, you should first assess the result based on two criteria by giving a reason: 1) prompt_following, does the last edited image fulfill the last user’s editing prompt? 2) consistency: Are the untouched parts of the last result image consistent with the input reference (the source image at the first turn, or the result image at the previous turn)?
2) Scoring: Based on the reason, you assign scores for "prompt_following" and "consistency". 
From scale 0 to 10: 
A "prompt_following" score from 0 to 10 will be given based on the editing success of prompt following. (0 indicates that the scene in the last edited image does not follow the last editing instruction at all. 10 indicates that the scene in the last edited image follow the last editing instruction perfectly.)
A "consistency" score from 0 to 10 will rate the degree of overediting in the last edited image. (0 indicates that the scene in the last edited image is completely different from the original. 10 indicates that the last edited image can be recognized as a minimal edited yet effective version of original.)

3) Return your results in a JSON structure, following this format:
{{"reason": "...", "prompt_following": int, "consistency": int}}
"""

# define call func
def evaluate_with_gpt(client_info, editing_prompt, base64_images):
    api_key, model = client_info
    client = openai.OpenAI(api_key=api_key)
    promptTest = prompt_template.format(editing_prompt=str(editing_prompt), num_prompts=len(editing_prompt), num_images=len(base64_images))
    content = [{"type": "input_text", "text": promptTest}] + [{"type": "input_image", "image_url": f"data:image/jpeg;base64,{img}"} for img in base64_images]
    
    # Add retry mechanism
    max_retries = 3
    retry_delay = 1  # seconds
    
    for attempt in range(max_retries):
        try:
            # Use the new API method
            response = client.responses.create(
                model=model, 
                input=[{
                    "type": "message",
                    "role": "user",
                    "content": content
                }]
            )
            evaluateResult = response.output_text
            return evaluateResult
        except Exception as e:
            traceback.print_exc()
            if attempt < max_retries - 1:
                time.sleep(retry_delay)
                retry_delay *= 2  # Exponential backoff
            else:
                print(f"Failed after {max_retries} attempts: {str(e)}")
                return None


def process_batch(batch_data, data_path, res_image_cache_dir, args, client_infos):
    results = []
    
    for i_data, data in batch_data:
        res_dir = os.path.splitext(args.res_path)[0]
        res_path_cache = os.path.join(res_dir, f'{i_data}.json')
        
        if os.path.exists(res_path_cache):
            eval_res = read_json(res_path_cache)
        else:
            is_valid = False
            editing_prompt = {f"turn{i_c+1}": c for i_c, c in enumerate(data['context'])}
            seed = i_data
            images_path = [os.path.join(data_path, os.path.basename(data['img_paths'][0]))] + \
                    [os.path.join(res_image_cache_dir, f'{i_data}_turn{i_turn}_rep0_seed{seed}.png') for i_turn in range(len(data['context']))]

            if not os.path.exists(images_path[1]):
                images_path = [p.replace('rep0_', '') for p in images_path]
        

            for img_p in images_path:
                assert os.path.exists(img_p), f'{img_p} does not exist'
            assert len(images_path) == 6

            images_base64 = session_img_path_to_base64(images_path, resize=True)                     
            client_info = random.choice(client_infos)

            eval_res = {}
            succ_thr = args.success_threshold
            for i_turn in range(len(data['context'])):
                for i_try in range(NUM_TRY):
                    try:
                        editing_prompt_pre_i = {f'turn{i_+1}': editing_prompt[f'turn{i_+1}'] for i_ in range(i_turn + 1)}
                        eval_res_i_turn = evaluate_with_gpt(client_info, editing_prompt_pre_i, images_base64[:i_turn+2])
                        eval_res_i_turn = parse_mixed_string(eval_res_i_turn)
                        eval_res_i_turn['all'] = int(eval_res_i_turn['prompt_following'] > succ_thr and eval_res_i_turn['consistency'] > succ_thr)
                        eval_res[f"turn{i_turn+1}"] = eval_res_i_turn
                        is_valid = True
                    except:
                        is_valid = False
                        traceback.print_exc()

                    if is_valid:
                        break

                if not is_valid:
                    break
                if eval_res_i_turn['all'] == 0:
                    break
            
            if not is_valid:
                if len(images_path) == 1:
                    eval_res = {f"empty": "gpt4o could not deal with this case. "}
                else:
                    eval_res = {"error": f"Failed to get valid response after {NUM_TRY} attempts at turn {i_turn}"}
                

            os.makedirs(os.path.dirname(res_path_cache), exist_ok=True)
            save_json(eval_res, res_path_cache)
        
        results.append((i_data, eval_res))
    
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--hf_data_name', type=str, default='leigangqu/MSE-Bench')
    parser.add_argument('--model_name', type=str, required=True)
    parser.add_argument('--image_cache_dir', type=str, default='./tmp_data/images')
    parser.add_argument('--res_path', type=str, required=True)
    parser.add_argument('--num_workers', type=int, default=32, help='Number of parallel workers')
    parser.add_argument('--batch_size', type=int, default=2, help='Batch size for each worker')
    parser.add_argument('--success_threshold', type=int, default=6, help='Success threshold for each turn. Success if the two scores are both above the threshold.')
    parser.add_argument('--api_model', default='gpt-5-nano', help='The model name to use for evaluation, e.g., gpt-4o, gpt-5-nano.')
    args = parser.parse_args()

    # dataset
    data_name = args.hf_data_name.split('/')[-1]
    data_path = f'./tmp_data/dataset/{data_name}'
    os.makedirs(data_path, exist_ok=True)

    dataset = load_dataset(args.hf_data_name)
    train_dataset = dataset["train"]
    all_prompts = []
    for i in range(len(train_dataset)):
        example = train_dataset[i]
        
        # Save the PIL image to data_path so the rest of the script can load it
        img_filename = f"{example['index']}.jpg"
        img_save_path = os.path.join(data_path, img_filename)
        if not os.path.exists(img_save_path):
            example['image'].save(img_save_path)

        all_prompts.append({
            'context': example['context'],
            'img_paths': [img_filename],
            'index': example['index']
        })

    # Download experimental results from Hugging Face
    # print(f"Loading results from huggingface: leigangqu/MSE-Bench-results")
    results_dataset = load_dataset("leigangqu/MSE-Bench-results", split=args.model_name)

    res_image_cache_dir = os.path.join(args.image_cache_dir, args.model_name)
    os.makedirs(res_image_cache_dir, exist_ok=True)

    print(f"Saving result images to {res_image_cache_dir}...")
    for i_data, example in tqdm(enumerate(results_dataset), total=len(results_dataset)):
        seed = i_data
        
        for key, value in example.items():
            # Check for columns like "turn_0_image", "turn_1_image"
            # Handle potential spaces in keys if any, though unlikely in standard datasets
            clean_key = key.strip()
            if clean_key.startswith("turn_") and clean_key.endswith("_image"):
                try:
                    # key format: turn_{i_turn}_image
                    parts = clean_key.split('_')
                    # expected split: ['turn', '0', 'image']
                    if len(parts) == 3 and parts[1].isdigit():
                        i_turn = int(parts[1])
                        filename = f'{i_data}_turn{i_turn}_rep0_seed{seed}.png'
                        save_path = os.path.join(res_image_cache_dir, filename)
                        
                        if not os.path.exists(save_path):
                            if value is not None:
                                value.save(save_path)
                except Exception as e:
                    print(f"Error saving {key} for index {i_data}: {e}")

    api_keys = [os.environ['OPENAI_API_KEY']]
    models = [args.api_model]

    # start evaluation 
    res_path = args.res_path
    os.makedirs(os.path.dirname(res_path), exist_ok=True)
    res_dir = os.path.splitext(res_path)[0]
    os.makedirs(res_dir, exist_ok=True)
    
    client_infos = [(api_key, model) for api_key, model in zip(api_keys, models)]
    
    if not os.path.exists(res_path):
        # Prepare data for multiprocessing
        data_items = [(i_data, data) for i_data, data in enumerate(all_prompts)]
        
        # Split data into batches
        batches = []
        for i in range(0, len(data_items), args.batch_size):
            batches.append(data_items[i:i + args.batch_size])
        
        all_res = [None] * len(all_prompts)
        
        # Process batches in parallel
        with multiprocessing.Pool(processes=args.num_workers) as pool:
            process_func = partial(process_batch, data_path=data_path, res_image_cache_dir=res_image_cache_dir, args=args, client_infos=client_infos)
            
            # Process batches and collect results
            for batch_results in tqdm(pool.imap(process_func, batches), total=len(batches)):
                for i_data, eval_res in batch_results:
                    all_res[i_data] = eval_res
        
        # save results
        save_json(all_res, res_path)
        
    else:
        all_res = read_json(res_path)

    # summary
    n_turn = len(all_prompts[0]['context'])
    for p in all_prompts: assert len(p['context']) == n_turn, f'len(p["context"]) = {len(p["context"])} != {n_turn}'
    dict_report = {f'turn{i+1}': [] for i in range(n_turn)}
    pf_report = {f'turn{i+1}': [] for i in range(n_turn)} # prompt_following
    c_report = {f'turn{i+1}': [] for i in range(n_turn)} # consistency
    ov_report = {f'turn{i+1}': [] for i in range(n_turn)}
    valid_turns = {f'turn{i+1}': 0 for i in range(n_turn)}
    for res in all_res:
        assert 'error' not in res, f'error: {res}'
        for i in range(n_turn):
            if f'turn{i+1}' in res:
                valid_turns[f'turn{i+1}'] += 1
                dict_report[f'turn{i+1}'].append(res[f'turn{i+1}']['all'])
                pf, c = res[f'turn{i+1}']['prompt_following'], res[f'turn{i+1}']['consistency']
                pf_report[f'turn{i+1}'].append(pf)
                c_report[f'turn{i+1}'].append(c)
                ov_report[f'turn{i+1}'].append(math.sqrt(pf * c))

    res_str = ''
    pf_str = ''
    c_str = ''
    ov_str = ''
    for turn, res in dict_report.items():
        res_str += f'{turn}: {(sum(res) / len(all_prompts)):.4f}, '
        pf_str += f'{turn}: {(sum(pf_report[turn]) / len(all_prompts)):.4f}, '
        c_str += f'{turn}: {(sum(c_report[turn]) / len(all_prompts)):.4f}, '
        ov_str += f'{turn}: {(sum(ov_report[turn]) / len(all_prompts)):.4f}, '

    print('valid_turns:             ', valid_turns)
    print('success rate:            ', res_str)
    print('prompt_following score:  ', pf_str)
    print('consistency score:       ', c_str)
    print('overall score:           ', ov_str)

    print(args.res_path)
    print(os.path.basename(args.res_path).strip('.json'))


    # -------------------- to csv --------------------
    rows = []
    for turn in dict_report.keys():
        rows.append([
            turn,
            f'{(sum(dict_report[turn]) / len(all_prompts)):.4f}',
            f'{(sum(pf_report[turn]) / len(all_prompts)):.4f}',
            f'{(sum(c_report[turn]) / len(all_prompts)):.4f}',
            f'{(sum(ov_report[turn]) / len(all_prompts)):.4f}'
        ])
    # Write to CSV
    csv_path = os.path.join('./tmp_data/csv/', os.path.basename(args.res_path) + '.csv')
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Turn', 'Success Rate', 'Prompt Following', 'Consistency', 'Overall'])
        writer.writerows(rows)

if __name__ == "__main__":
    # Set start method for multiprocessing
    multiprocessing.set_start_method('spawn', force=True)
    main()