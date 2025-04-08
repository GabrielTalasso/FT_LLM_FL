from tyransformers import AutoModelForCausalLM
from config import get_model_config
from sklearn.cluster import KMeans

def get_embeddings_model(text, model):
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
    
    embeddings = []
    for text in client_dataset['instruction']:
        embedding = get_embeddings_model(text, model)
        embeddings.append(embedding)
    
    return embeddings

def cluster_embeddings(embeddings, num_clusters = 1):
    kmeans = KMeans(n_clusters=num_clusters)
    kmeans.fit(embeddings)
    return kmeans.cluster_centers_, kmeans.labels_

def separete_data_into_clusters(embeddings, labels):
    clusters = {}
    for i, label in enumerate(labels):
        if label not in clusters:
            clusters[label] = []
        clusters[label].append(embeddings[i])
    return clusters

def cluster_clients_centroids(client_embeddings_centers, num_clusters = 1):
    centers = []
    for client, embeddings in client_embeddings_centers.items():
        centers.append(embeddings)
    kmeans = KMeans(n_clusters=num_clusters)
    kmeans.fit(client_embeddings_centers)
    return kmeans.cluster_centers_, kmeans.labels_


    

    