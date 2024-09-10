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

losses = []
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
    loss = 0
    for i, example in tqdm(enumerate(eval_set)):
        
        instruction = template.format(example["instruction"], "", "")[:-1]
        encoded = tokenizer(text = instruction, text_target = example['output'],
                     return_tensors="pt",  padding='max_length', truncation=True, max_length=128)

        input_ids = encoded.input_ids.to("cuda")
        output_ids = encoded.labels.to("cuda") # Use the 'labels' field for target

        #calculate loss
        with torch.no_grad():
            output = model(input_ids, labels=output_ids, )
            loss += output.loss.item()

    print('Loss: ', loss/EVALSET_LEN)
    losses.append(loss/EVALSET_LEN)

#save losses
np.save(PATH + '/eval_loss.npy', losses)


