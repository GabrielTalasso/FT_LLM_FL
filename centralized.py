import os
os.environ["CUDA_VISIBLE_DEVICES"] = "5"

import numpy as np
import os
import sys
import os
from tqdm import tqdm
import torch
sys.path.append(".")
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, TrainingArguments
from datasets import load_dataset
from accelerate import Accelerator
from torch.utils.data import DataLoader
from utils.template import TEMPLATE_DICT
import seaborn as sns
from peft import LoraConfig, get_peft_model
from matplotlib import pyplot as plt
from trl import SFTTrainer, DataCollatorForCompletionOnlyLM
from datasets import concatenate_datasets

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

        languages = ['English', 'Dutch', 'Turkish', 'Portuguese', 'Spanish'] #filtrar nas linguas
        dataset = dataset.filter(lambda x: x['language'] in languages)

        dataset = dataset.train_test_split(test_size=0.2, seed=0)
        if eval:
            dataset = dataset['test']
        else:
            dataset = dataset['train']
        dataset = dataset.filter(lambda x: x['language'] in tasks) 
        dataset = dataset.map(aya_format)
        return dataset

    if DATASET_NAME == 'multitask':
        if tasks == 'boolq':
            dataset = prepare_boolq(eval=eval)
            dataset = dataset.shuffle(seed=0)
            return dataset
        if tasks == 'webnlg':
            dataset = prepare_webnlg(eval=eval)
            dataset = dataset.shuffle(seed=0)
            return dataset
        if tasks == 'samsum':
            dataset = prepare_samsum(eval=eval)
            dataset = dataset.shuffle(seed=0)
            return dataset
        if tasks == 'gigaword':
            dataset = prepare_gigaword(eval=eval)
            dataset = dataset.shuffle(seed=0)
            return dataset
        if tasks == 'all_tasks':
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

def get_formatting_prompts_func(template_name, eos_token):
    if template_name in TEMPLATE_DICT:
        overall_temp, response_temp = TEMPLATE_DICT[template_name]
        def formatting_prompts_func(example):    
            #output_texts = []    
            text = overall_temp.format(example['instruction'], example['response'], eos_token)
            #output_texts.append(text)    
            return text#output_texts

    elif template_name == 'ag_news':

        formatting_prompts_func = None
        response_temp = None
    
    return formatting_prompts_func, response_temp


def main():
    template = TEMPLATE_DICT['alpaca'][0]
    MODEL_NAME = 'HuggingFaceTB/SmolLM-360M'
    #MODEL_NAME = 'HuggingFaceTB/SmolLM-360M'

    #DATASET_NAME = "databricks/databricks-dolly-15k"
    #DATASET_NAME = "CohereForAI/aya_dataset"
    DATASET_NAME = 'multitask'

    max_eval_len = 1000

    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, load_in_4bit=True)

    peft_config = LoraConfig(
            r=32,
            lora_alpha=64,
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM")

    model = get_peft_model(model, peft_config)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    tokenizer.pad_token = tokenizer.unk_token #tokenizer.eos_token
    tokenizer.add_eos_token = True

    if DATASET_NAME == "databricks/databricks-dolly-15k":
        experiments = [
            {'experiment_name': 'open_qa', 'tasks': ['open_qa']},
            {'experiment_name': 'classification', 'tasks': ['classification']},
            {'experiment_name': 'summarization', 'tasks': ['summarization']},
            {'experiment_name': 'all_tasks', 'tasks': ['open_qa', 'classification', 'summarization']}
        ]

    elif DATASET_NAME == "CohereForAI/aya_dataset":
        experiments = [
            {'experiment_name': 'english', 'tasks': ['English']},
            {'experiment_name': 'dutch', 'tasks': ['Dutch']},
            {'experiment_name': 'turkish', 'tasks': ['Turkish']},
            {'experiment_name': 'portuguese', 'tasks': ['Portuguese']},
            {'experiment_name': 'spanish', 'tasks': ['Spanish']},
            {'experiment_name': 'all_languages', 'tasks': ['English', 'Dutch', 'Turkish', 'Portuguese', 'Spanish']}

            #next: Remove: German and Swedish and add Dutch and Turkish
        ]

    elif DATASET_NAME == 'multitask':
        experiments = [
            {'experiment_name': 'boolq', 'tasks': 'boolq'},
            {'experiment_name': 'webnlg', 'tasks': 'webnlg'},
            {'experiment_name': 'samsum', 'tasks': 'samsum'},
            {'experiment_name': 'gigaword', 'tasks': 'gigaword'},
            {'experiment_name': 'all_tasks', 'tasks': 'all_tasks'}
        ]
    
    for experiment in experiments:

        experiment_name = experiment['experiment_name']
        tasks = experiment['tasks']

        data = load_data(DATASET_NAME, tasks = tasks)
        eval_data = load_data(DATASET_NAME, tasks = tasks, eval = True)
        eval_data = eval_data.select(range(min(max_eval_len, len(eval_data))))

        if DATASET_NAME == "databricks/databricks-dolly-15k":
            data = data.map(dolly_format)
            eval_data = eval_data.map(dolly_format)  # Apply same formatting to eval data
        
        elif DATASET_NAME == "CohereForAI/aya_dataset":
            data = data.map(aya_format)
            eval_data = eval_data.map(aya_format)

        formatting_prompts_func, response_temp = get_formatting_prompts_func('alpaca', tokenizer.eos_token)

        class ScriptArgs:
            output_dir = f"./output_centralized/experiments_wo_formatting/{experiment_name}/{MODEL_NAME.split('-')[-1]}"
            batch_size = 16
            logging_steps = 10
            num_train_epochs = 10
            max_steps = 1000 #2000
            save_steps = int(max_steps/5)
            eval_steps = int(max_steps/10)
            save_total_limit = int(max_steps//save_steps)
            push_to_hub = False
            hub_model_id = "my_model"
            gradient_checkpointing = False

        script_args = ScriptArgs()
        new_lr = 5e-4

        training_args = TrainingArguments(
            output_dir=script_args.output_dir,
            per_device_train_batch_size=script_args.batch_size,
            per_device_eval_batch_size=script_args.batch_size,  # Add eval batch size
            learning_rate=new_lr,
            logging_steps=script_args.logging_steps,
            num_train_epochs=script_args.num_train_epochs,
            max_steps=script_args.max_steps,
            save_steps=script_args.save_steps,
            eval_steps=script_args.eval_steps,  # Add evaluation steps
            evaluation_strategy="steps",  # Run evaluation at specific steps
            save_total_limit=script_args.save_total_limit,
            push_to_hub=script_args.push_to_hub,
            hub_model_id=script_args.hub_model_id,
            gradient_checkpointing=script_args.gradient_checkpointing,
            lr_scheduler_type="constant",
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
        )


        # Define the missing variables
        response_temp = '\n### Response:'
        response_temp_ids = tokenizer(response_temp)['input_ids']
        data_collator = DataCollatorForCompletionOnlyLM(response_temp_ids, tokenizer=tokenizer)

        packing = False  # Example value for packing
        dataset_text_field = 'inputs'  # Example field name

        trainer = SFTTrainer(
                    model=model,
                    tokenizer=tokenizer,
                    args=training_args,
                    #max_seq_length=128,
                    train_dataset=data,
                    eval_dataset=eval_data,
                    formatting_func=formatting_prompts_func,
                    data_collator=data_collator,
                    #packing=packing,
                    #dataset_text_field=dataset_text_field,
                )

        trainer.train()

if __name__ == "__main__":
    main()
