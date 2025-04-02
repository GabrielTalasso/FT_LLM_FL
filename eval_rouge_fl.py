import sys
import os
from tqdm import tqdm
import numpy as np
import torch
import json
import pandas as pd
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel
from datasets import load_dataset, concatenate_datasets
from accelerate import Accelerator
from torch.utils.data import DataLoader
import evaluate

sys.path.append(".")
from utils.template import TEMPLATE_DICT

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

def load_model(path, MODEL_NAME, DEVICE = 'cuda'):
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME).to(DEVICE)
    model = PeftModel.from_pretrained(model, path).to(DEVICE)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, padding_side="left")
    tokenizer.pad_token = tokenizer.eos_token
    return model, tokenizer

def load_data(DATASET_NAME, tasks, eval=False):
    if DATASET_NAME == "databricks/databricks-dolly-15k":
        dataset = load_dataset(DATASET_NAME, split="train")
        dataset = dataset.train_test_split(test_size=0.2, seed=0)
        dataset = dataset['test'] if eval else dataset['train']
        dataset = dataset.filter(lambda x: x['category'] in tasks)
        dataset = dataset.map(dolly_format)
        return dataset

    if DATASET_NAME == "CohereForAI/aya_dataset":
 
        dataset = load_dataset(DATASET_NAME, split="train")
        languages = ['English', 'Dutch', 'Turkish', 'Portuguese', 'Spanish']
        dataset = dataset.filter(lambda x: x['language'] in languages)
        dataset = dataset.train_test_split(test_size=0.2, seed=0)
        dataset = dataset['test'] if eval else dataset['train']
        #tasks = [task.capitalize() for task in tasks]
        dataset = dataset.filter(lambda x: x['language'] == tasks)
        dataset = dataset.map(aya_format)
        return dataset

    if DATASET_NAME == 'multitask':
        if tasks == 'boolq' or 'boolq' in tasks:
            dataset = prepare_boolq(eval=eval).shuffle(seed=0)
            return dataset
        if tasks == 'webnlg' or 'webnlg' in tasks:
            dataset = prepare_webnlg(eval=eval).shuffle(seed=0)
            return dataset
        if tasks == 'samsum' or 'samsum' in tasks:
            dataset = prepare_samsum(eval=eval).shuffle(seed=0)
            return dataset
        if tasks == 'gigaword' or 'gigaword' in tasks:
            dataset = prepare_gigaword(eval=eval).shuffle(seed=0)
            return dataset
        if tasks == 'all_tasks' or 'all_tasks' in tasks:
            boolq = prepare_boolq(eval=eval).map(lambda x: {'instruction': x['instruction'], 'response': x['response'], 'task': 'boolq'})
            webnlg = prepare_webnlg(eval=eval).map(lambda x: {'instruction': x['instruction'], 'response': x['response'], 'task': 'webnlg'})
            samsum = prepare_samsum(eval=eval).map(lambda x: {'instruction': x['instruction'], 'response': x['response'], 'task': 'samsum'})
            gigaword = prepare_gigaword(eval=eval).map(lambda x: {'instruction': x['instruction'], 'response': x['response'], 'task': 'gigaword'})
            dataset = concatenate_datasets([boolq, webnlg, samsum, gigaword]).shuffle(seed=0)
            return dataset

def prepare_webnlg(eval=False):
    dataset = load_dataset('GEM/web_nlg', 'en', split='train')
    dataset = dataset.train_test_split(test_size=0.2, seed=0)
    dataset = dataset['test'] if eval else dataset['train']
    dataset = dataset.map(webnlg_format)
    return dataset

def prepare_boolq(eval=False):
    dataset = load_dataset('google/boolq', split='train')
    dataset = dataset.train_test_split(test_size=0.2, seed=0)
    dataset = dataset['test'] if eval else dataset['train']
    dataset = dataset.map(boolq_format)
    return dataset

def prepare_samsum(eval=False):
    dataset = load_dataset('Samsung/samsum', split='train', trust_remote_code=True)
    dataset = dataset.train_test_split(test_size=0.2, seed=0)
    dataset = dataset['test'] if eval else dataset['train']
    dataset = dataset.map(samsum_format)
    return dataset

def prepare_gigaword(eval=False):
    dataset = load_dataset('Harvard/gigaword', split='train', trust_remote_code=True)
    dataset = dataset.train_test_split(test_size=0.2, seed=0)
    dataset = dataset['test'] if eval else dataset['train']
    dataset = dataset.shuffle(seed=0)
    dataset = dataset.select(range(30000))
    dataset = dataset.map(gigaword_format)
    return dataset

def boolq_format(example):
    #example["instruction"] = example['passage'] + " Based on the passage, answer this question:" + example['question']
    example["instruction"] = example['passage'] + '-' + example['question']
    example["response"] = str(example['answer'])
    return example

def webnlg_format(example):
    example['input'] = str(example['input'])
    #example["instruction"] = "Organize this data into a readable text: " + example['input']
    example["instruction"] = example['input']
    example["response"] = example['target']
    return example

def samsum_format(example):
    #example["instruction"] = "Summarize this conversation: " + example['dialogue']
    example["instruction"] = example['dialogue']
    example["response"] = example['summary']
    return example

def gigaword_format(example):
    #example["instruction"] = "Summarize this text: " + example['document']
    example["instruction"] = example['document']
    example["response"] = example['summary']
    return example

def dolly_format(example):
    if example['context'] == "":
        example["inputs"] = example["instruction"]
    else:
        example["inputs"] = example["instruction"] + " " + example['context']
    return example

def aya_format(example):
    example["instruction"] = example['inputs']
    example["response"] = example['targets']
    return example

alpaca_template = """Below is an instruction that describes a task. Write a response that appropriately completes the request.

### Instruction:
{} 

### Response: {}{}"""

TEMPLATE_DICT = {
    'alpaca': (alpaca_template, '\n### Response:'),
}

def tokenize_function(examples):
    inputs = tokenizer(examples["inputs"], return_tensors="pt", padding='max_length', truncation=True, max_length=512)
    targets = inputs.copy()
    return {
        "input_ids": inputs["input_ids"].squeeze(),
        "attention_mask": inputs["attention_mask"].squeeze(),
        "labels": targets["input_ids"].squeeze()
    }

def format_instruction(instruction, response, eos):
    template = """Below is an instruction that describes a task. Write a response that appropriately completes the request.

### Instruction:
{} 

### Response: {}{}"""
    return template.format(instruction, response, eos)

def apply_template_to_dataset(dataset):
    dataset = dataset.map(lambda x: {'inputs': format_instruction(x, '', '')})
    return dataset

def get_model_responses(model, tokenizer, dataset, batch_size=8):
    model_responses = []
    for i in tqdm(range(0, len(dataset), batch_size)):
        batch = dataset[i:i+batch_size]
        tokenized = tokenizer(batch['inputs'],  padding='max_length',return_tensors='pt', truncation=True, max_length=512)
        input_ids = tokenized['input_ids'].to('cuda')
        attention_mask = tokenized['attention_mask'].to('cuda')
        #print(input_ids[0], input_ids[1])
        with torch.no_grad():
            outputs = model.generate(input_ids=input_ids, attention_mask = attention_mask, max_new_tokens=512, do_sample=False, use_cache=True, eos_token_id=tokenizer.eos_token_id, pad_token_id=tokenizer.pad_token_id)
            batch_responses = [tokenizer.decode(output, skip_special_tokens=True) for output in outputs]
            model_responses.extend(batch_responses)
    return model_responses

def save_model_responses(dataset, model_responses, path):
    dataset.add_column('model_responses', model_responses)
    dataset.save_to_disk(path)

def calcule_rogue1(model_responses, dataset):
    metric = evaluate.load("rouge")
    references = [dataset[i]['targets'] for i in range(len(dataset))]
    #predictions  = [response.split("### Response: ")[-1] for response in model_responses]
    predictions = [dataset[i]['model_responses'].split("### Response: ")[-1] for i in range(len(dataset))]
    scores = metric.compute(predictions=predictions, references=references)
    return scores

def calculate_perplexity(instruction, output, model, tokenizer, device):
    model = model.to(device)
    full_text = instruction + output
    full_encodings = tokenizer(full_text, return_tensors="pt")
    input_ids = full_encodings["input_ids"].to(device)
    output_encodings = tokenizer(output, return_tensors="pt")
    output_len = output_encodings["input_ids"].shape[1]
    labels = input_ids.clone()
    labels[:, :-output_len] = -100
    with torch.no_grad():
        outputs = model(input_ids, labels=labels)
        loss = outputs.loss
    return torch.exp(loss).item()

def get_formatting_prompts_func_test(template_name, eos_token):
    if template_name in TEMPLATE_DICT:
        overall_temp, response_temp = TEMPLATE_DICT[template_name]
        def formatting_prompts_func(example):    
            text = overall_temp.format(example['instruction'], '', '')
            return text
    elif template_name == 'ag_news':
        formatting_prompts_func = None
        response_temp = None
    return formatting_prompts_func, response_temp

def evaluate_model(model_path, base_model, dataset_name, task, device, eval_len, eval_rouge, eval_perplexity, output_dir):
    print(f"Evaluating model: {model_path} on task: {task}")
    print(f"Base model: {base_model}")
    print(f"Dataset: {dataset_name}")
    
    # Load model
    model, tokenizer = load_model(model_path, base_model, device)
    
    dataset = load_data(dataset_name, task, eval=True)
    dataset = dataset.select(range(eval_len))
    
    # Apply formatting
    formatting_prompts_func, _ = get_formatting_prompts_func_test('alpaca', '\n### Response:')
    dataset = dataset.map(lambda x: {'inputs': formatting_prompts_func(x), 'targets': x['response']})
    
    # Create output directory (this output dir is used in addition to saving a JSON in the model path)
    model_name = os.path.basename(model_path) if os.path.isdir(model_path) else os.path.basename(os.path.dirname(model_path))
    base_model_name = base_model.split('/')[-1]
    out_dir = os.path.join(output_dir, f"{model_name}_{base_model_name}_{task}")
    os.makedirs(out_dir, exist_ok=True)
    
    results = {}
    
    if eval_rouge:
        print("Calculating ROUGE scores...")
        model_responses = get_model_responses(model, tokenizer, dataset, batch_size=128)
        dataset_with_responses = dataset.select(range(len(dataset)))
        dataset_with_responses = dataset_with_responses.add_column('model_responses', model_responses)
        dataset_with_responses.save_to_disk(os.path.join(out_dir, "responses"))
        scores = calcule_rogue1(model_responses, dataset_with_responses)
        print(f"ROUGE scores: {scores}")
        with open(os.path.join(out_dir, "rouge.json"), 'w') as f:
            json.dump(scores, f)
        results['rouge'] = scores
    
    if eval_perplexity:
        print("Calculating perplexity...")
        perplexity_scores = []
        for i in tqdm(range(len(dataset))):
            ppx = calculate_perplexity(dataset[i]['inputs'], dataset[i]['targets'], model, tokenizer, device)
            perplexity_scores.append(ppx)
        avg_perplexity = np.mean(perplexity_scores)
        median_perplexity = np.median(perplexity_scores)
        print(f"Median perplexity: {median_perplexity}")
        print(f"Average perplexity: {avg_perplexity}")
        perplexity_results = {
            "average_perplexity": float(avg_perplexity), 
            "median_perplexity": float(median_perplexity), 
            "perplexity_scores": [float(p) for p in perplexity_scores]
        }
        with open(os.path.join(out_dir, "perplexity.json"), 'w') as f:
            json.dump(perplexity_results, f)
        results['perplexity'] = perplexity_results

    with open(os.path.join(out_dir, "results.json"), 'w') as f:
        json.dump(results, f)
        
    print(f"Evaluation complete. Results saved to {out_dir}")
    return results

if __name__ == "__main__":
    #For Multitask
    path_model_clustred = 'output_multitask/Llama-3.2-1B/clustered_multitask_clustered_c20s5_i10_b16a1_l1024_r8a16_20250401151553'
    model_list = ['output_multitask/Llama-3.2-1B/fedavg_multitask_clustered_c20s5_i10_b16a1_l1024_r8a16_20250401152254/cluster_0_checkpoint-200', #fedavg
                    path_model_clustred + '/cluster_0_checkpoint-200', #clustered
                    path_model_clustred + '/cluster_1_checkpoint-200', #clustered
                    path_model_clustred + '/cluster_2_checkpoint-200', #clustered
                    path_model_clustred + '/cluster_3_checkpoint-200', #clustered
    ]

    #for Aya
    path_model_clustered  = '/home/gabriel.talasso/FT_LLM_FL/output_aya/Llama-3.2-1B/clustered_aya_dataset_clustered_c20s5_i10_b16a1_l1024_r8a16_20250401163027'
    model_list  = ['/home/gabriel.talasso/FT_LLM_FL/output_aya/Llama-3.2-1B/fedavg_aya_dataset_clustered_c20s5_i10_b16a1_l1024_r8a16_20250401162934/cluster_0_checkpoint-200', #fedavg,
                    path_model_clustered + '/cluster_0_checkpoint-200', #clustered
                    path_model_clustered + '/cluster_1_checkpoint-200', #clustered
                    path_model_clustered + '/cluster_2_checkpoint-200', #clustered
                    path_model_clustered + '/cluster_3_checkpoint-200', #clustered
                    path_model_clustered + '/cluster_4_checkpoint-200', #clustered
    ]
    #base_model = 'HuggingFaceTB/SmolLM-360M'

    base_model = 'unsloth/Llama-3.2-1B'

    dataset_name = 'multitask'
    dataset_name = 'CohereForAI/aya_dataset'

    task_list =  ['all_tasks', 'boolq', 'gigaword', 'webnlg', 'samsum']
    task_list = ['English', 'Dutch', 'Turkish', 'Portuguese', 'Spanish']
    
    
    device = 'cuda'
    eval_len = 100
    eval_rouge = True
    eval_perplexity = True

    # For each model, evaluate on each task and then save a performance JSON in the model path.
    for model_path in model_list:
        performance_for_model = {}
        for task in task_list:
            print(f"\nEvaluating Model: {model_path} for Task: {task}")
            results = evaluate_model(
                model_path=model_path,
                base_model=base_model,
                dataset_name=dataset_name,
                task=task,
                device=device,
                eval_len=eval_len,
                eval_rouge=eval_rouge,
                eval_perplexity=eval_perplexity,
                output_dir=model_path
            )
            performance_for_model[task] = results

        performance_json_path = os.path.join(model_path, "performance.json")
        with open(performance_json_path, "w") as f:
            json.dump(performance_for_model, f)
        print(f"Saved performance results for model {model_path} to {performance_json_path}")
