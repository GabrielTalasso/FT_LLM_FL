#!/bin/bash
# filepath: /home/gabriel.talasso/FT_LLM_FL/run_multiple_simulations.sh

# Base parameters from the original script
max_steps=10
num_train_epochs=1
num_rounds=200
batch_size=16
gradient_accumulation_steps=1
seq_length=1024
num_clients=20
sample_clients=5
lora_r=8
lora_alpha=16
lr=5e-4
dataset_name='multitask'
output_dir="output_multitask/multiple_simulations"
dataset_sample=400000
model_name_or_path='unsloth/Llama-3.2-1B'
gpu='2'
fed_alg="router"

# Run 10 simulations with different seeds
for i in {1..5}; do
    # Set seed and unique simulation name for each run
    seed=$((42 + i))
    sim_alias="ROUTER_iid_tfix1_seed${seed}"
    
    echo "====================================="
    echo "Starting run $i/10 with seed $seed"
    echo "Simulation alias: $sim_alias"
    echo "====================================="
    
    CUDA_VISIBLE_DEVICES=$gpu python main_sft_fedrouter.py \
     --learning_rate $lr \
     --model_name_or_path $model_name_or_path \
     --dataset_name $dataset_name \
     --dataset_sample $dataset_sample \
     --fed_alg $fed_alg \
     --num_clients $num_clients \
     --sample_clients $sample_clients \
     --max_steps $max_steps \
     --num_train_epochs $num_train_epochs \
     --num_rounds $num_rounds \
     --batch_size $batch_size \
     --gradient_accumulation_steps $gradient_accumulation_steps \
     --seq_length $seq_length \
     --peft_lora_r $lora_r \
     --peft_lora_alpha $lora_alpha \
     --use_peft True \
     --load_in_4bit True \
     --output_dir $output_dir \
     --template "alpaca" \
     --sim_round -1 \
     --n_clusters 5 \
     --split_strategy "multitask_clusters" \
     --train_split 0.8 \
     --sim_alias $sim_alias \
     --global_n_clusters 5 \
     --seed $seed
    
    echo "Completed run $i/10"
    echo ""
done

echo "All simulations completed!"