import random

def split_dataset(fed_args, script_args, dataset):
    dataset = dataset.shuffle(seed=script_args.seed)        # Shuffle the dataset
    local_datasets = []
    if fed_args.split_strategy == "iid":
        for i in range(fed_args.num_clients):
            local_datasets.append(dataset.shard(fed_args.num_clients, i))
    
    #------------- IID datasets ----------------
    if fed_args.split_strategy == "language_iid":
        
        #languages = ['English', 'Swedish', 'German', 'Portuguese', 'Spanish']
        languages = ['English', 'Dutch', 'Turkish', 'Portuguese', 'Spanish']
        dataset = dataset.filter(lambda x: x['language'] in languages)
        for i in range(fed_args.num_clients):
            local_datasets.append(dataset.shard(fed_args.num_clients, i))
    
    if fed_args.split_strategy == "ag_news_iid":
        for i in range(fed_args.num_clients):
            local_datasets.append(dataset.shard(fed_args.num_clients, i))

    if fed_args.split_strategy == "dolly_iid":
        for i in range(fed_args.num_clients):
            local_datasets.append(dataset.shard(fed_args.num_clients, i))
    
    if fed_args.split_strategy == "multitask_iid":
        for i in range(fed_args.num_clients):
            local_datasets.append(dataset.shard(fed_args.num_clients, i))

    #------------- Non-IID datasets ----------------

    if fed_args.split_strategy == "language_clusters":
        #languages = ['English', 'Swedish', 'German', 'Portuguese', 'Spanish']
        languages = ['English', 'Dutch', 'Turkish', 'Portuguese', 'Spanish']
        n_clients_in_cluster = fed_args.num_clients // len(languages)

        for i in range(fed_args.num_clients):
            language = languages[i // n_clients_in_cluster]
            cluster_dataset = dataset.filter(lambda x: x['language'] == language)
            cluster_dataset = cluster_dataset.shuffle(seed=script_args.seed)

            local_datasets.append(cluster_dataset.shard(n_clients_in_cluster, i % n_clients_in_cluster))
    
    if fed_args.split_strategy == "ag_news_clusters":
        n_clients_in_cluster = fed_args.num_clients // 4
        categories = [0, 1, 2, 3] # 0: World, 1: Sports, 2: Business, 3: Sci/Tech

        for i in range(fed_args.num_clients):
            category = categories[i // n_clients_in_cluster]
            cluster_dataset = dataset.filter(lambda x: x['label'] == category)
            cluster_dataset = cluster_dataset.shuffle(seed=script_args.seed)

            local_datasets.append(cluster_dataset.shard(n_clients_in_cluster, i % n_clients_in_cluster))
    
    if fed_args.split_strategy == "dolly_clusters":
        tasks = ['open_qa', 'general_qa', 'closed_qa', 'classification', 'brainstorming', 'information_extraction', 'summarization']
        n_clients_in_cluster = fed_args.num_clients // len(tasks)

        for i in range(fed_args.num_clients):
            task = tasks[i // n_clients_in_cluster]
            cluster_dataset = dataset.filter(lambda x: x['category'] == task)
            cluster_dataset = cluster_dataset.shuffle(seed=script_args.seed)

            local_datasets.append(cluster_dataset.shard(n_clients_in_cluster, i % n_clients_in_cluster))
    
    if fed_args.split_strategy == "multitask_clusters":
        tasks = ['boolq', 'gigaword', 'samsum', 'webnlg']
        n_clients_in_cluster = fed_args.num_clients // len(tasks)

        for i in range(fed_args.num_clients):
            task = tasks[i // n_clients_in_cluster]
            cluster_dataset = dataset.filter(lambda x: x['task'] == task)
            cluster_dataset = cluster_dataset.shuffle(seed=script_args.seed)

            local_datasets.append(cluster_dataset.shard(n_clients_in_cluster, i % n_clients_in_cluster))

    return local_datasets

def get_dataset_this_round(dataset, round, fed_args, script_args):
    num2sample = script_args.batch_size * script_args.gradient_accumulation_steps * script_args.max_steps
    num2sample = min(num2sample, len(dataset))
    random.seed(round)
    random_idx = random.sample(range(0, len(dataset)), num2sample)
    dataset_this_round = dataset.select(random_idx)

    return dataset_this_round