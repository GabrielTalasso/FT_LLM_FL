import copy
import os
from tqdm import tqdm
import numpy as np

from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import DataCollatorForCompletionOnlyLM
from peft import get_peft_model, get_peft_model_state_dict, set_peft_model_state_dict, prepare_model_for_kbit_training

from utils import *
from federated_learning import *
from config import get_config, save_config, get_model_config, get_training_args
import glob

# ===== Define the arguments =====
script_args, fed_args, peft_config = get_config()
training_args = get_training_args(script_args, script_args.learning_rate)
save_config(script_args, fed_args)
print(script_args, fed_args)

# ===== Load the dataset =====
if script_args.train_split < 1:
    dataset, dataset_test = get_dataset(script_args.dataset_name, script_args.local_data_dir, script_args.train_split)
else:
    dataset = get_dataset(script_args.dataset_name, script_args.local_data_dir)

dataset = process_sft_dataset(script_args.dataset_name, dataset, script_args.dataset_sample)

# ===== Split the dataset into clients =====
local_datasets = split_dataset(fed_args, script_args, dataset)
sample_num_list = [len(local_datasets[i]) for i in range(fed_args.num_clients)]

# ===== Get model config =====
device_map, quantization_config, torch_dtype = get_model_config(script_args)

# Load base model (without adapters)
base_model = AutoModelForCausalLM.from_pretrained(
    script_args.model_name_or_path,
    quantization_config=quantization_config,
    device_map=device_map,
    trust_remote_code=script_args.trust_remote_code,
    torch_dtype=torch_dtype,
)
print(f"Model loaded from {script_args.model_name_or_path}")

if script_args.load_in_8bit or script_args.load_in_4bit:
    base_model = prepare_model_for_kbit_training(
                base_model, use_gradient_checkpointing=training_args.gradient_checkpointing
            )

base_model.config.use_cache = False  # silence the warnings. Please re-enable for inference!

if training_args.gradient_checkpointing:
    base_model.enable_input_require_grads()

# ===== Define the tokenizer =====
tokenizer = AutoTokenizer.from_pretrained(script_args.model_name_or_path, use_fast=False, padding_side="right")
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.unk_token   # following vicuna

# ===== Define the formatting function (cater to TRL SFTTrainer)=====
formatting_prompts_func, response_template = get_formatting_prompts_func(script_args.template, tokenizer.eos_token)
if response_template:
    response_template_ids = tokenizer.encode(response_template, add_special_tokens=False)[2:]   # Now we have it like in the dataset texts: `[2277, 29937, 4007, 22137, 29901]` for Llama2
    data_collator = DataCollatorForCompletionOnlyLM(response_template_ids, tokenizer=tokenizer)
    packing = False
else:
    data_collator = None
    packing = True

# Initialize global adapter for first round
temp_model = get_peft_model(base_model, peft_config, adapter_name="global")
temp_model.print_trainable_parameters()
global_dict = copy.deepcopy(get_peft_model_state_dict(temp_model, adapter_name="global"))
del temp_model  # Free up memory

# Initialize storage for client local adapters
local_adapter_dicts = [{} for _ in range(fed_args.num_clients)]

# Setup federated learning components
proxy_dict, opt_proxy_dict = get_proxy_dict(fed_args, global_dict)
global_auxiliary, auxiliary_model_list, auxiliary_delta_dict = get_auxiliary_dict(fed_args, global_dict)

# ===== Create output directory for adapters =====
if not os.path.exists(os.path.join(script_args.output_dir, "global_adapters")):
    os.makedirs(os.path.join(script_args.output_dir, "global_adapters"))
if not os.path.exists(os.path.join(script_args.output_dir, "local_adapters")):
    os.makedirs(os.path.join(script_args.output_dir, "local_adapters"))

# ===== Start federated training =====
training_loss = [[] for i in range(fed_args.num_clients)]
idx = None

# Dictionary to store updated global adapters for each client in current round
client_global_updates = {}

for round in tqdm(range(fed_args.num_rounds)):
    clients_this_round = get_clients_this_round(fed_args, round)

    if round + 1 == fed_args.sim_round:  # return all clients
        clients_this_round = list(range(fed_args.num_clients))

    print(f">> ==================== Round {round+1} : {clients_this_round} ====================")
    
    # Clear the client updates dictionary for this round
    client_global_updates = {}
    
    for client in range(fed_args.num_clients):
        if client not in clients_this_round:
            training_loss[client].append(-1)  # -1 is an indicator of not training
            continue
        
        print(f'Setting parameters for client {client}...')
        
        # 1) Start with the base model
        model = copy.deepcopy(base_model)
        
        # 2) Add the global adapter
        model = get_peft_model(model, peft_config, adapter_name="global")
        
        # 3) Load the global adapter state from the previous round
        set_peft_model_state_dict(model, global_dict, adapter_name="global")
        #model.set_adapter('global')
        
        print('Global adapter layers - Before training:')
        c = 0
        for name, param in model.named_parameters():
            print(name, param[0:10][0][0])
            c += 1
            if c == 4:
                break

        sub_dataset = get_dataset_this_round(local_datasets[client], round, fed_args, script_args)
        
        new_lr = cosine_learning_rate(round, fed_args.num_rounds, script_args.learning_rate, 1e-5)
        training_args = get_training_args(script_args, new_lr)

        # Train only the global adapter
        trainer = get_fed_local_sft_trainer(
            model=model,
            tokenizer=tokenizer,
            training_args=training_args,
            local_dataset=sub_dataset,
            formatting_prompts_func=formatting_prompts_func,
            data_collator=data_collator,
            global_dict=global_dict,
            fed_args=fed_args,
            script_args=script_args,
            local_auxiliary=auxiliary_model_list[client],
            global_auxiliary=global_auxiliary,
            packing=packing
        )

        print(f'Training global adapter for client {client}...')
        results = trainer.train()
        #training_loss[client].append(results.training_loss)

        # Save updated global adapter state for aggregation
        updated_global_dict = copy.deepcopy(get_peft_model_state_dict(model, adapter_name="global"))
        client_global_updates[client] = updated_global_dict
        
        print('Global adapter layers - After training:')
        c = 0
        for name, param in model.named_parameters():
            print(name, param[0:10][0][0])
            c += 1
            if c == 4:
                break

        # Update auxiliary information for SCAFFOLD if used
        if fed_args.fed_alg == 'scaffold':
            auxiliary_model_list[client], auxiliary_delta_dict[client] = trainer.get_auxiliary_param()

        # 3) Freeze global adapter by merging it into the base model
        trained_model_with_global = model.merge_and_unload()
        
        # 4) Add local adapter on top of the merged model (with global adapter knowledge)
        local_adapter_config = copy.deepcopy(peft_config)
        model = get_peft_model(trained_model_with_global, local_adapter_config, adapter_name="local")
        
        # Load previous local adapter state if it exists
        if local_adapter_dicts[client]:  # Check if dict is not empty
            set_peft_model_state_dict(model, local_adapter_dicts[client], adapter_name="local")
        
        print('Local adapter layers - Before training:')
        c = 0
        for name, param in model.named_parameters():
            print(name, param[0:10][0][0])
            c += 1
            if c == 4:
                break

        # Train only the local adapter
        trainer = get_fed_local_sft_trainer(
            model=model,
            tokenizer=tokenizer,
            training_args=training_args,
            local_dataset=sub_dataset,
            formatting_prompts_func=formatting_prompts_func,
            data_collator=data_collator,
            global_dict={},  # No need for global dict here
            fed_args=fed_args,
            script_args=script_args,
            local_auxiliary=None,
            global_auxiliary=None,
            packing=packing
        )

        print(f'Training local adapter for client {client}...')
        results = trainer.train()
        training_loss[client].append(results.training_loss)
        
        # Save the local adapter state
        local_adapter_dicts[client] = copy.deepcopy(get_peft_model_state_dict(model, adapter_name="local"))
        
        print('Local adapter layers - After training:')
        c = 0
        for name, param in model.named_parameters():
            print(name, param[0:10][0][0])
            c += 1
            if c == 4:
                break

        # Save both adapters separately
        trainer.save_model(os.path.join(script_args.output_dir, f"local_adapters/checkpoint-{round+1}_client{client}"))
        
        # Save the global adapter for this client
        temp_model = copy.deepcopy(base_model)
        temp_model = get_peft_model(temp_model, peft_config, adapter_name="global")
        set_peft_model_state_dict(temp_model, updated_global_dict, adapter_name="global")
        
        # Create a temporary trainer to save the model
        temp_trainer = get_fed_local_sft_trainer(
            model=temp_model,
            tokenizer=tokenizer,
            training_args=training_args,
            local_dataset=sub_dataset,  # This doesn't matter for saving
            formatting_prompts_func=formatting_prompts_func,
            data_collator=data_collator,
            global_dict=global_dict,
            fed_args=fed_args,
            script_args=script_args,
            local_auxiliary=None,
            global_auxiliary=None,
            packing=packing
        )
        temp_trainer.save_model(os.path.join(script_args.output_dir, f"global_adapters/checkpoint-{round+1}_client{client}"))
        del temp_model, temp_trainer  # Free up memory
    
    # 6) Server aggregates ONLY the global adapters (after all clients in this round are processed)
    if round < fed_args.sim_round:
        # Aggregate only global adapters - we're not touching local adapters
        global_dict, global_auxiliary = global_aggregate(
            fed_args, script_args, global_dict, client_global_updates, sample_num_list,
            clients_this_round, round, proxy_dict=proxy_dict,
            opt_proxy_dict=opt_proxy_dict,
            auxiliary_info=(global_auxiliary, auxiliary_delta_dict),
            round=round
        )

        # Save the aggregated global model
        if (round+1) % fed_args.save_model_freq == 0:
            # Reset model to update with new aggregated globals
            temp_model = copy.deepcopy(base_model)
            temp_model = get_peft_model(temp_model, peft_config, adapter_name="global")
            set_peft_model_state_dict(temp_model, global_dict, adapter_name="global")
            
            # Create a temporary trainer to save the model
            temp_trainer = get_fed_local_sft_trainer(
                model=temp_model,
                tokenizer=tokenizer,
                training_args=training_args,
                local_dataset=sub_dataset,  # This is just a placeholder
                formatting_prompts_func=formatting_prompts_func,
                data_collator=data_collator,
                global_dict=global_dict,
                fed_args=fed_args,
                script_args=script_args,
                local_auxiliary=None,
                global_auxiliary=None,
                packing=packing
            )
            temp_trainer.save_model(os.path.join(script_args.output_dir, f"global_checkpoint-{round+1}"))
            del temp_model, temp_trainer  # Free up memory
        
        np.save(os.path.join(script_args.output_dir, "training_loss.npy"), np.array(training_loss))
    
    # For clustering if implemented
    if round >= fed_args.sim_round:
        global_dict, global_auxiliary, idx = global_aggregate(
            fed_args, script_args, global_dict, client_global_updates, sample_num_list,
            clients_this_round, round, proxy_dict=proxy_dict,
            opt_proxy_dict=opt_proxy_dict,
            auxiliary_info=(global_auxiliary, auxiliary_delta_dict),
            round=round, idx=idx
        )

        for cluster in range(fed_args.n_clusters):
            temp_model = copy.deepcopy(base_model)
            temp_model = get_peft_model(temp_model, peft_config, adapter_name="global")
            set_peft_model_state_dict(temp_model, global_dict[cluster])
            
            # Create a temporary trainer to save the model
            temp_trainer = get_fed_local_sft_trainer(
                model=temp_model,
                tokenizer=tokenizer,
                training_args=training_args,
                local_dataset=sub_dataset,  # This is just a placeholder
                formatting_prompts_func=formatting_prompts_func,
                data_collator=data_collator,
                global_dict=global_dict[cluster],
                fed_args=fed_args,
                script_args=script_args,
                local_auxiliary=None,
                global_auxiliary=None,
                packing=packing
            )
            temp_trainer.save_model(os.path.join(script_args.output_dir, f"global_cluster_{cluster}_checkpoint-{round+1}"))
            del temp_model, temp_trainer  # Free up memory
    
        np.save(os.path.join(script_args.output_dir, "training_loss.npy"), np.array(training_loss))

print("Federated training with dual adapter solution completed!")