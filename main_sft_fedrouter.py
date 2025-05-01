import copy
import os
from tqdm import tqdm
import numpy as np
from math import floor

from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import DataCollatorForCompletionOnlyLM
from peft import get_peft_model, get_peft_model_state_dict, set_peft_model_state_dict, prepare_model_for_kbit_training

from utils import *
from federated_learning import *
from federated_learning.router_utils import *
from config import get_config, save_config, get_model_config, get_training_args

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

model = AutoModelForCausalLM.from_pretrained(
    script_args.model_name_or_path,
    quantization_config=quantization_config,
    device_map=device_map,
    trust_remote_code=script_args.trust_remote_code,
    torch_dtype=torch_dtype,
)
print(f"Model loaded from {script_args.model_name_or_path}")

if script_args.load_in_8bit or script_args.load_in_4bit:
    model = prepare_model_for_kbit_training(
                model, use_gradient_checkpointing=training_args.gradient_checkpointing
            )

model = get_peft_model(model, peft_config)
model.print_trainable_parameters()

model.config.use_cache = False  # silence the warnings. Please re-enable for inference!

if training_args.gradient_checkpointing:
    model.enable_input_require_grads()

# ===== Define the global and local models =====
global_dict = copy.deepcopy(get_peft_model_state_dict(model))
local_dict_list = [copy.deepcopy(global_dict) for i in range(fed_args.num_clients)]
proxy_dict, opt_proxy_dict = get_proxy_dict(fed_args, global_dict)
global_auxiliary, auxiliary_model_list, auxiliary_delta_dict = get_auxiliary_dict(fed_args, global_dict)

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

# ===== Start federated training =====
training_loss = [[] for i in range(fed_args.num_clients)]
idx = None

client_embeddings_centers = {}
data_cluster_labels = {}

for round in tqdm(range(fed_args.num_rounds)):

    clients_this_round = get_clients_this_round(fed_args, script_args, round)
    
    if round  == 0: #return all clients
        clients_this_round = list(range(fed_args.num_clients))

    print(f">> ==================== Round {round+1} : {clients_this_round} ====================")
    
    for client in range(fed_args.num_clients):

        if client not in clients_this_round:
            training_loss[client].append(-1)            # -1 is an indicator of not training
            continue
        
        print(f"Getting Embeddings for client {client}...")
        if client not in client_embeddings_centers and round == 0:
            client_embeddings_centers[client], data_cluster_labels[client] = cluster_embeddings(
                            get_client_embedding(script_args, fed_args, local_datasets[client]),
                            num_clusters = fed_args.n_clusters
                            )

            local_datasets[client] = local_datasets[client].add_column('cluster_label', data_cluster_labels[client])

        print(f"Client {client} embeddings center shape: {client_embeddings_centers[client].shape}")

        sub_dataset = get_dataset_this_round(local_datasets[client], round, fed_args, script_args)
        #clusters_datasets = separate_data_into_clusters(sub_dataset, data_cluster_labels[client])
        
        local_dict_list[client] = []
        training_loss_aux = []
        
        for c in range(fed_args.n_clusters):
            cluster_dataset = sub_dataset.filter(lambda x: x['cluster_label'] == c)
            
            #starts with global model
            print(f'Setting parameters for client {client}...')
            if round >= 1:
                set_peft_model_state_dict(model, global_dict[get_most_similar_adapter(global_centroids, global_clusters, client_embeddings_centers[client][c])])
            else:
                set_peft_model_state_dict(model, global_dict)

            if not os.path.exists(os.path.join(script_args.output_dir, "clients_adapters")):
                os.makedirs(os.path.join(script_args.output_dir, "clients_adapters"))

            new_lr = cosine_learning_rate(round, fed_args.num_rounds, script_args.learning_rate, 1e-5)
            training_args = get_training_args(script_args, new_lr)
            
            #same amount of computation for each cluster
            #ajusted_max_steps = int(script_args.max_steps / fed_args.n_clusters)
            #training_args.max_steps = max(ajusted_max_steps, 1)

            #ajusted to the number of samples in each cluster subdataset
            ajusted_max_steps = (len(cluster_dataset) / len(sub_dataset)) * script_args.max_steps
            training_args.max_steps = max(floor(ajusted_max_steps), 1)

            trainer = get_fed_local_sft_trainer(
                model=model,
                tokenizer=tokenizer,
                training_args=training_args,
                local_dataset=cluster_dataset,
                formatting_prompts_func=formatting_prompts_func,
                data_collator=data_collator,
                global_dict=global_dict,
                fed_args=fed_args,
                script_args=script_args,
                local_auxiliary=auxiliary_model_list[client],
                global_auxiliary=global_auxiliary,
                packing=packing
                )

            results = trainer.train()
            training_loss_aux.append(results.training_loss)

            local_dict_list[client].append(copy.deepcopy(get_peft_model_state_dict(model)))
            
        training_loss[client].append(np.mean(training_loss_aux))
    
    if round == 0:
        client_embeddings_centers_list = []
        for client in range(fed_args.num_clients):
            for client_embeddings_center in client_embeddings_centers[client]:
                client_embeddings_centers_list.append(client_embeddings_center)

        global_centroids, global_clusters = cluster_clients_centroids(client_embeddings_centers_list, num_clusters = fed_args.global_n_clusters)
        print(f'Global clusters found: {global_clusters}')
    
        #saving local and global centroids
        print(f"Saving centroids...")

        centroids_path = os.path.join(script_args.output_dir, f"centers/centroids_{round+1}.npy")
        os.makedirs(os.path.dirname(centroids_path), exist_ok=True)
        np.save(centroids_path, global_centroids)

        client_embeddings_path = os.path.join(script_args.output_dir, f"centers/client_embeddings_{round+1}.npy")
        os.makedirs(os.path.dirname(client_embeddings_path), exist_ok=True)
        np.save(client_embeddings_path, client_embeddings_centers)

        clusters_path = os.path.join(script_args.output_dir, f"centers/global_clusters_{round+1}.npy")
        os.makedirs(os.path.dirname(clusters_path), exist_ok=True)
        np.save(clusters_path, global_clusters)


    #Flattening the local_dict_list
    all_local_dict_list = []
    for client in range(fed_args.num_clients):
        for adapter in local_dict_list[client]:
            all_local_dict_list.append(adapter)
    
    print("Length of all_local_dict_list: ", len(all_local_dict_list))

    #Getting only the global clusters and adapters for clients this round 
    global_clusters_this_round = []
    local_dict_list_this_round = []
    for i in range(0, len(global_clusters), fed_args.n_clusters):
        if i // fed_args.n_clusters in clients_this_round:
            global_clusters_this_round += global_clusters[i:i+fed_args.n_clusters].tolist()
            local_dict_list_this_round += all_local_dict_list[i:i+fed_args.n_clusters]

    # ===== Server aggregates the local models =====
    
    global_dict, global_auxiliary, idx = global_aggregate(
            fed_args, script_args, global_dict, local_dict_list_this_round, sample_num_list, \
            clients_this_round, round, proxy_dict=proxy_dict, \
            opt_proxy_dict=opt_proxy_dict,
            auxiliary_info=(global_auxiliary, auxiliary_delta_dict),
            round = round, idx = global_clusters_this_round
        )

    for cluster in range(fed_args.global_n_clusters):
        set_peft_model_state_dict(model, global_dict[cluster])
        trainer.save_model(os.path.join(script_args.output_dir, f"cluster_{cluster}_checkpoint-{round+1}"))

    np.save(os.path.join(script_args.output_dir, "training_loss.npy"), np.array(training_loss))
