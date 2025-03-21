max_steps=10
num_train_epochs=1
num_rounds=200
batch_size=16
gradient_accumulation_steps=1
seq_length=1024
num_clients=21
sample_clients=2
lora_r=8
lora_alpha=16   # twice of lora_r
lr=5e-5

# local_data_dir=""       # you may uncomment this line if your data is stored locally and include it in the python command

#dataset_name="vicgalle/alpaca-gpt4"
#dataset_name='CohereForAI/aya_dataset'

#dataset_name='fancyzhx/ag_news'
dataset_name='databricks/databricks-dolly-15k'
dataset_sample=400000
#model_name_or_path="TinyLlama/TinyLlama-1.1B-Chat-v1.0"
model_name_or_path="TinyLlama/TinyLlama_v1.1"
output_dir="output_dolly"

gpu='2'
fed_alg="clustered"

CUDA_VISIBLE_DEVICES=$gpu python main_sft_clustered.py \
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
 --sim_round 50 \
 --n_clusters 7 \
 --split_strategy "dolly_clusters" \
 --train_split 0.8 \