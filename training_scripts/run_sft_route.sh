max_steps=1
num_train_epochs=1
num_rounds=3
eval_round=1
batch_size=16
batch_size_eval=128
gradient_accumulation_steps=1
seq_length=1024
num_clients=5
sample_clients=10
lora_r=8
lora_alpha=16  # twice of lora_r
lr=5e-4
# local_data_dir=""       # you may uncomment this line if your data is stored locally and include it in the python command

#dataset_name="vicgalle/alpaca-gpt4"
dataset_name='CohereForAI/aya_dataset'
#dataset_name='databricks/databricks-dolly-15k'
#dataset_name="multitask"

output_dir="output_aya/baselines_big"

dataset_sample=400000

sim_alias='ROUTER'

#model_name_or_path='HuggingFaceTB/SmolLM-1.7B'
#model_name_or_path='HuggingFaceTB/SmolLM-360M'
model_name_or_path='HuggingFaceTB/SmolLM-135M'
#model_name_or_path='unsloth/Llama-3.2-1B'

gpu='4'
fed_alg="router"

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
 --n_clusters 2 \
 --split_strategy "language_multi_domain" \
 --train_split 0.8 \
 --sim_alias $sim_alias \
 --global_n_clusters 5 \
 --evaluation_rounds $eval_round \
 --eval_batch_size $batch_size_eval \