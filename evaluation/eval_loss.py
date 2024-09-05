import sys
sys.path.append("../../")
from tqdm import tqdm
import numpy as np
import torch

from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel
from datasets import load_dataset
from utils.template import TEMPLATE_DICT

template = TEMPLATE_DICT['alpaca'][0]

#load model
MODEL_NAME = 'TinyLlama/TinyLlama-1.1B-Chat-v1.0'
DATASET_NAME = 'dominguesm/alpaca-data-pt-br'
PATH = 'output/alpaca-data-pt-br_10000_fedavg_c20s2_i10_b16a1_l512_r8a16_20240904165038'
DEVICE = 'cuda'
NUM_CHECKPOINTS = 20
EVALSET_LEN = 5000

losses = []
for i in range(NUM_CHECKPOINTS):

    print('Evaluation checkpoint:', i)

    i = i*10

    path = PATH + f'/checkpoint-{i}'
    lora_path = path + 'adapter_model.bin'

    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=torch.float16,
                                                 quantization_config = BitsAndBytesConfig(
                                                                            load_in_4bit=True,
                                                                            bnb_4bit_use_double_quant=True,
                                                                            bnb_4bit_quant_type="nf4",
                                                                            bnb_4bit_compute_dtype=torch.bfloat16,
                                                                        )).to(DEVICE)
    model = PeftModel.from_pretrained(model, lora_path, torch_dtype=torch.float16).to(DEVICE)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=False)

    #sample a evaluation set
    dataset = load_dataset(DATASET_NAME)
    dataset = dataset.shuffle(seed = 42)
    eval_set = dataset.select(range(EVALSET_LEN))


    #mesure loss in the evaluation set
    loss = 0
    for i, example in tqdm(enumerate(eval_set)):

        instruction = template.format(example["instruction"], "", "")[:-1] #format instruction

        input_ids = tokenizer.encode(instruction, return_tensors="pt").to(DEVICE) #encode instruction
        output_ids = tokenizer.encode(example['output'], return_tensors="pt").to(DEVICE) #get labels ids

        #calculate loss
        loss += model(input_ids, labels=output_ids).loss

    losses.append(loss/EVALSET_LEN)

#save losses
np.save(PATH + '/eval_losses.npy', losses)


