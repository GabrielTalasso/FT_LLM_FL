import sys
import os
from tqdm import tqdm
import numpy as np
import torch
sys.path.append(".")
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel
from datasets import load_dataset, concatenate_datasets
from accelerate import Accelerator
from torch.utils.data import DataLoader
from utils.template import TEMPLATE_DICT
import json
import pandas as pd
import torch
from tqdm import tqdm
import numpy as np
import evaluate

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

def load_model(path, MODEL_NAME, DEVICE):
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME).to(DEVICE)
    model = PeftModel.from_pretrained(model, path).to(DEVICE)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, device=DEVICE, use_fast=False, padding_side="left")
    tokenizer.pad_token = tokenizer.eos_token

    return model, tokenizer

def load_data(DATASET_NAME, tasks, eval = False):

    if DATASET_NAME == "databricks/databricks-dolly-15k":
        dataset = load_dataset(DATASET_NAME, split="train")
        dataset = dataset.train_test_split(test_size=0.2, seed=0)
        if eval:
            dataset = dataset['test']
        else:
            dataset = dataset['train']
        dataset = dataset.filter(lambda x: x['category'] in tasks) #['open_qa', 'general_qa', 'closed_qa', 'classification', 'brainstorming', 'information_extraction', 'summarization'])
        dataset = dataset.map(dolly_format)
        return dataset
    
    if DATASET_NAME == "CohereForAI/aya_dataset":

        dataset = load_dataset(DATASET_NAME, split="train")

        languages = ['English', 'Swedish', 'German', 'Portuguese', 'Spanish'] #filtrar nas linguas
        dataset = dataset.filter(lambda x: x['language'] in languages)

        dataset = dataset.train_test_split(test_size=0.2, seed=0)
        if eval:
            dataset = dataset['test']
        else:
            dataset = dataset['train']

        tasks = [task.capitalize() for task in tasks]
        dataset = dataset.filter(lambda x: x['language'] in tasks) 
        dataset = dataset.map(aya_format)
        return dataset

    if DATASET_NAME == 'multitask':
        if tasks == 'boolq' or 'boolq' in tasks:
            dataset = prepare_boolq(eval=eval)
            dataset = dataset.shuffle(seed=0)
            return dataset
        if tasks == 'webnlg' or 'webnlg' in tasks:
            dataset = prepare_webnlg(eval=eval)
            dataset = dataset.shuffle(seed=0)
            return dataset
        if tasks == 'samsum' or 'samsum' in tasks:
            dataset = prepare_samsum(eval=eval)
            dataset = dataset.shuffle(seed=0)
            return dataset
        if tasks == 'gigaword' or 'gigaword' in tasks:
            dataset = prepare_gigaword(eval=eval)
            dataset = dataset.shuffle(seed=0)
            return dataset
        if  tasks == 'all_tasks' or 'all_tasks' in tasks:
            #selecting only "instruction" and "response" columns
            boolq = prepare_boolq(eval=eval).map(lambda x: {'instruction': x['instruction'], 'response': x['response']})
            webnlg = prepare_webnlg(eval=eval).map(lambda x: {'instruction': x['instruction'], 'response': x['response']})
            samsum = prepare_samsum(eval=eval).map(lambda x: {'instruction': x['instruction'], 'response': x['response']})
            gigaword = prepare_gigaword(eval=eval).map(lambda x: {'instruction': x['instruction'], 'response': x['response']})
            
            dataset = concatenate_datasets([boolq, webnlg, samsum, gigaword])
            dataset = dataset.shuffle(seed=0)
            return dataset

def prepare_webnlg(eval = False):
    dataset = load_dataset('GEM/web_nlg', 'en', split = 'train')
    dataset = dataset.train_test_split(test_size=0.2, seed=0)

    if eval:
        dataset = dataset['test']
    else:
        dataset = dataset['train']
    
    dataset = dataset.map(webnlg_format)

    return dataset

def prepare_boolq(eval = False):
    dataset = load_dataset('google/boolq', split = 'train')
    dataset = dataset.train_test_split(test_size=0.2, seed=0)

    if eval:
        dataset = dataset['test']
    else:
        dataset = dataset['train']
    
    dataset = dataset.map(boolq_format)

    return dataset

def prepare_samsum(eval = False):
    dataset = load_dataset('Samsung/samsum', split = 'train', trust_remote_code=True)
    dataset = dataset.train_test_split(test_size=0.2, seed=0)

    if eval:
        dataset = dataset['test']
    else:
        dataset = dataset['train']

    dataset = dataset.map(samsum_format)

    return dataset

def prepare_gigaword(eval = False):
    dataset = load_dataset('Harvard/gigaword', split = 'train', trust_remote_code=True)
    dataset = dataset.train_test_split(test_size=0.2, seed=0)

    if eval:
        dataset = dataset['test']
    else:
        dataset = dataset['train']
    
    dataset = dataset.shuffle(seed=0)
    dataset = dataset.select(range(30000)) #loading only part (the whole dataset has aroud 4M examples)
    dataset = dataset.map(gigaword_format)

    return dataset

def boolq_format(example):
    example["instruction"] = example['passage'] + " Based on the passage, answer this question:" + example['question']
    example["response"] = str(example['answer'])
    return example

def webnlg_format(example):
    example['input'] = str(example['input'])
    example["instruction"] = "Organize this data into a readable text: " + example['input']
    example["response"] = example['target']
    return example

def samsum_format(example):
    example["instruction"] = "Summarize this conversation: " + example['dialogue']
    example["response"] = example['summary']
    return example

def gigaword_format(example):
    example["instruction"] = "Summarize this text: " + example['document']
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
    inputs = tokenizer(examples["inputs"],  return_tensors="pt")
    targets = tokenizer(examples["targets"],  return_tensors="pt")
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
        input_ids = tokenizer(batch['inputs'], return_tensors='pt')['input_ids'].to(DEVICE)
        

        with torch.no_grad():
            outputs = model.generate(input_ids=input_ids, max_new_tokens=512, do_sample=False, use_cache=True)
            batch_responses = [tokenizer.decode(output, skip_special_tokens=True) for output in outputs]
            model_responses.extend(batch_responses)

    return model_responses

def save_model_responses(dataset, model_responses, path):
    dataset.add_column('model_responses', model_responses)
    dataset.save_to_disk(path)

def calcule_rogue1(model_responses, dataset):
    metric = evaluate.load("rouge")
    references = [dataset[i]['targets'] for i in range(len(dataset))]
    scores = metric.compute(predictions=model_responses, references=references)
    return scores

def calculate_perplexity(instruction, output, model, tokenizer, device):
    # Move model to specified device
    model = model.to(device)

    # Combine instruction and output into one full text
    full_text = instruction + output
    #print(output)

    # Tokenize the full text
    full_encodings = tokenizer(full_text, return_tensors="pt")
    input_ids = full_encodings["input_ids"].to(device)

    # Tokenize just the instruction to determine the cutoff point
    output_encodings = tokenizer(output, return_tensors="pt")
    output_len = output_encodings["input_ids"].shape[1]

    # Create labels and mask out the instruction tokens
    labels = input_ids.clone()
    labels[:, :-output_len] = -100
    #print(labels)

    # Calculate perplexity

    with torch.no_grad():
        outputs = model(input_ids, labels=labels)
        loss = outputs.loss

    return torch.exp(loss).item()

alpaca_template = """Below is an instruction that describes a task. Write a response that appropriately completes the request.

### Instruction:
{} 

### Response: {}{}"""

TEMPLATE_DICT = {
    'alpaca': (alpaca_template, '\n### Response:'),
}

def get_formatting_prompts_func_test(template_name, eos_token):
    if template_name in TEMPLATE_DICT:
        overall_temp, response_temp = TEMPLATE_DICT[template_name]
        def formatting_prompts_func(example):    
            #output_texts = []    
            text = overall_temp.format(example['instruction'], '', '')
            #output_texts.append(text)    
            return text#output_texts

    elif template_name == 'ag_news':

        formatting_prompts_func = None
        response_temp = None
    
    return formatting_prompts_func, response_temp

formatting_prompts_func, response_template = get_formatting_prompts_func_test('alpaca', '\n### Response:')

MODEL_NAME = 'HuggingFaceTB/SmolLM-1.7B'
DATASET_NAME = "databricks/databricks-dolly-15k"
DATASET_NAME = 'multitask'

trained_datasets = ['open_qa', 'classification', 'summarization', 'all_tasks']
test_tasks = ['open_qa', 'classification', 'summarization', 'all_tasks']

trained_datasets = ['all_tasks', 'boolq', 'webnlg', 'samsum', 'gigaword']
test_tasks = ['all_tasks', 'boolq', 'webnlg', 'samsum', 'gigaword']

#trained_datasets = ['english', 'swedish', 'german', 'portuguese', 'spanish', 'all_tasks']
#test_tasks = ['english', 'swedish', 'german', 'portuguese', 'spanish', 'all_tasks']

DEVICE = 'cuda:0'
EVALSET_LEN = 1000

rouge_enabled = False
perplexity_enabled = True

for checkpoint in [500]:
    for task in trained_datasets:

        path = f"output_centralized/experiments_lora32/{task}/{MODEL_NAME.split('-')[-1]}/checkpoint-{checkpoint}"
                
        model, tokenizer = load_model(path, MODEL_NAME, DEVICE)

        for test_task in test_tasks:
            print(f"Training task: {task} - Testing task: {test_task}")

            if test_task == 'all_tasks':
                print("Testing on all tasks")

                dataset = load_data(DATASET_NAME, 'all_tasks', eval=True)
            else:
                print(f"Testing on {test_task}")
                dataset = load_data(DATASET_NAME, [test_task], eval=True)

            dataset = dataset.select(range(EVALSET_LEN))
            dataset = dataset.map(lambda x: {'inputs': formatting_prompts_func(x), 'targets': x['response']})
            
            traintest = task + '_' + test_task

            if rouge_enabled:

                model_responses = get_model_responses(model, tokenizer, dataset)

                path_ds = f"inference_results/experiments/{traintest}/{MODEL_NAME.split('-')[-1]}"
                save_model_responses(dataset, model_responses, path_ds)

                scores = calcule_rogue1(model_responses, dataset)
                print(f"ROUGE scores: {scores}")
                
                # Save rouge scores
                os.makedirs(f"inference_results/experiments/{traintest}/{MODEL_NAME.split('-')[-1]}-{checkpoint}", exist_ok=True)
                with open(f"inference_results/experiments/{traintest}/{MODEL_NAME.split('-')[-1]}-{checkpoint}/rouge.json", 'w') as f:
                    json.dump(scores, f)
                    
            if perplexity_enabled:
                # Calculate perplexity
                perplexity_scores = []
                for i in range(len(dataset)):
                    #print(dataset[i]['inputs'])#, dataset[i]['targets']) 
                    ppx = calculate_perplexity(dataset[i]['inputs'], dataset[i]['targets'], model, tokenizer, DEVICE)
                    perplexity_scores.append(ppx)
                
                avg_perplexity = np.mean(perplexity_scores)
                median_perplexity = np.median(perplexity_scores)
                print(f"Median perplexity: {median_perplexity}")
                print(f"Average perplexity: {avg_perplexity}")
                
                # Save perplexity scores
                os.makedirs(f"inference_results/experiments/{traintest}/{MODEL_NAME.split('-')[-1]}-{checkpoint}", exist_ok=True)
                with open(f"inference_results/experiments/{traintest}/{MODEL_NAME.split('-')[-1]}-{checkpoint}/perplexity.json", 'w') as f:
                    json.dump({"average_perplexity": avg_perplexity, "median_perplexity": median_perplexity, "perplexity_scores": perplexity_scores}, f)

