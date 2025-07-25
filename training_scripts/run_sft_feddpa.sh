max_steps=5
num_train_epochs=1
num_rounds=200
batch_size=16
gradient_accumulation_steps=1
seq_length=1024
num_clients=10
sample_clients=2
lora_r=8
lora_alpha=16   # twice of lora_r
lr=5e-4

# local_data_dir=""       # you may uncomment this line if your data is stored locally and include it in the python command

#dataset_name="vicgalle/alpaca-gpt4"
#dataset_name='CohereForAI/aya_dataset'
#dataset_name='databricks/databricks-dolly-15k'
dataset_name="multitask"

output_dir="output_multitask"

dataset_sample=400000

sim_alias='feddpa'

#model_name_or_path='HuggingFaceTB/SmolLM-1.7B'
model_name_or_path='HuggingFaceTB/SmolLM-135M'
#model_name_or_path='unsloth/Llama-3.2-1B'

gpu='0'
fed_alg="clustered"

CUDA_VISIBLE_DEVICES=$gpu python main_sft_feddpa.py \
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
 --sim_round 1000 \
 --n_clusters 1 \
 --split_strategy "multitask_iid" \
 --train_split 0.8 \
 --sim_alias $sim_alias \