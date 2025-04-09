from transformers import AutoModelForCausalLM
from config import get_model_config
from sklearn.cluster import KMeans
import torch
from transformers import AutoTokenizer
import datasets
import numpy as np

def get_embeddings_model(text, model, tokenizer):
    with torch.no_grad():
        inputs = tokenizer(text, return_tensors='pt', padding=True, truncation=True, max_length=512)
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        outputs = model(**inputs, output_hidden_states=True)
        embedding = outputs.hidden_states[-1].mean(dim=1)
    return embedding

def get_client_embedding(script_args, fed_args, client_dataset):

    device_map, quantization_config, torch_dtype = get_model_config(script_args)
    model = AutoModelForCausalLM.from_pretrained(
        script_args.model_name_or_path,
        quantization_config=quantization_config,
        device_map=device_map,
        trust_remote_code=script_args.trust_remote_code,
        torch_dtype=torch_dtype)
    
    tokenizer = AutoTokenizer.from_pretrained(script_args.model_name_or_path, use_fast=False, padding_side="right")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    embeddings = []
    for text in client_dataset['instruction']:
        embedding = get_embeddings_model(text, model, tokenizer)
        embeddings.append(embedding[0])
    
    embeddings = torch.stack(embeddings).to(torch.float16).cpu().numpy()  # convert to float16 and send to CPU
    print(f"Embeddings shape: {embeddings.shape}")
    return embeddings

def cluster_embeddings(embeddings, num_clusters = 1):
    kmeans = KMeans(n_clusters=num_clusters)
    kmeans.fit(embeddings)
    return kmeans.cluster_centers_, kmeans.labels_

def separate_data_into_clusters(sub_dataset, labels):
    clusters_datasets = []
    sub_dataset = sub_dataset.add_column('index', list(range(len(sub_dataset))))
    for i in range(max(labels) + 1):
        cluster_dataset = sub_dataset.filter(lambda x: labels[x['index']] == i)
        clusters_datasets.append(cluster_dataset)
    return clusters_datasets
    
def cluster_clients_centroids(client_embeddings_centers, num_clusters = 1):
    centers = []
    kmeans = KMeans(n_clusters=num_clusters)
    kmeans.fit(client_embeddings_centers)
    return kmeans.cluster_centers_, kmeans.labels_

def get_most_similar_adapter(global_centroids, global_clusters, client_centroid):
    most_similar_adapter = None
    min_distance = float('inf')
    
    for i, centroid in enumerate(global_centroids):
        distance = np.linalg.norm(centroid - client_centroid)
        if distance < min_distance:
            min_distance = distance
            most_similar_adapter = i
            
    return global_clusters[most_similar_adapter]

    

    