import sys
import os
from tqdm import tqdm
import numpy as np
import torch
sys.path.append(".")
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel
from datasets import load_dataset
from accelerate import Accelerator
from torch.utils.data import DataLoader
from utils.template import TEMPLATE_DICT

#configs
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
template = TEMPLATE_DICT['alpaca'][0]
MODEL_NAME = 'TinyLlama/TinyLlama-1.1B-Chat-v1.0'
DATASET_NAME = 'dominguesm/alpaca-data-pt-br'
PATH = 'output/alpaca-data-pt-br_10000_local1_c20s2_i10_b16a1_l512_r8a16_20240905071418'
DEVICE = 'cuda'
NUM_CHECKPOINTS = 20
EVALSET_LEN = 50

all_perplexities = []
for i in [1,20]:#range(1, NUM_CHECKPOINTS+1):

    i = i*10
    print('-------------------------------------- Evaluation checkpoint: Round ', i)

    path = PATH + f'/checkpoint-{i}'
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=torch.float16,
                                                 quantization_config = BitsAndBytesConfig(
                                                                            load_in_4bit=True,
                                                                            bnb_4bit_use_double_quant=True,
                                                                            bnb_4bit_quant_type="nf4",
                                                                            bnb_4bit_compute_dtype=torch.bfloat16,
                                                                        ),
                                                 device_map={"": Accelerator().local_process_index},)
    
    model = PeftModel.from_pretrained(model, path).to(DEVICE)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=False)

    #sample a evaluation set
    dataset = load_dataset(DATASET_NAME)['train']
    dataset = dataset.shuffle(seed = 0)
    eval_set = dataset.select(range(EVALSET_LEN))

    print('Calculating model predictions...')
    #mesure loss in the evaluation set
    def calculate_perplexity(instruction, output):
        # Combine instruction and output
        combined = f"{instruction} [SEP] {output}"
        
        # Tokenize
        encodings = tokenizer(combined, return_tensors="pt", truncation=True, max_length=512)
        input_ids = encodings["input_ids"].to(DEVICE)
        attention_mask = encodings["attention_mask"].to(DEVICE)
    
        # Calculate perplexity
        with torch.no_grad():
            outputs = model(input_ids, attention_mask=attention_mask, labels=input_ids)
            loss = outputs.loss
            
        return torch.exp(loss).item()

    model.eval()
    perplexities = []

    for sample in tqdm(dataset):
        instruction = sample["instruction"] + sample["input"]

        perplexity = calculate_perplexity(instruction, sample["output"])
        perplexities.append(perplexity)

    # 5. Calculate mean perplexity
    mean_perplexity = np.mean(perplexities)
    std_perplexity = np.std(perplexities)

    print(f"Mean Perplexity: {mean_perplexity:.2f}")
    print(f"Standard Deviation of Perplexity: {std_perplexity:.2f}")

    all_perplexities.append((i, mean_perplexity, std_perplexity))

np.save(PATH + '/perplexities.npy', all_perplexities)


