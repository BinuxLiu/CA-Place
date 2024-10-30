import faiss, os
import logging
import numpy as np
from tqdm import tqdm
import time

import torch
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Subset
from utils import visualizations

def test_efficient_ram_usage(args, eval_ds, model):
    model = model.eval()

    distances = np.empty([eval_ds.queries_num, eval_ds.database_num], dtype=np.float32)
    
    with torch.inference_mode():
        queries_features = np.ones((eval_ds.queries_num, args.features_dim), dtype="float32")
        queries_infer_batch_size = 1
        # queries_infer_batch_size = args.infer_batch_size
        queries_subset_ds = Subset(eval_ds, list(range(eval_ds.database_num, eval_ds.database_num+eval_ds.queries_num)))
        queries_dataloader = DataLoader(dataset=queries_subset_ds, num_workers=args.num_workers,
                                        batch_size=queries_infer_batch_size, pin_memory=True)
        for images, indices in tqdm(queries_dataloader, ncols=100):
            features = model(images.to("cuda"))
            queries_features[indices.numpy()-eval_ds.database_num, :] = features.cpu().numpy()

        queries_features = torch.tensor(queries_features).type(torch.float32).cuda()

        database_subset_ds = Subset(eval_ds, list(range(eval_ds.database_num)))
        database_dataloader = DataLoader(dataset=database_subset_ds, num_workers=args.num_workers,
                                         batch_size=args.infer_batch_size, pin_memory=True)
        database_descriptors_dir = os.path.join(eval_ds.dataset_folder, f"database_{args.aggregation}.npy")
        if os.path.isfile(database_descriptors_dir) == 1:
            database_descriptors = np.load(database_descriptors_dir)
            total_size = database_descriptors.shape[0]
            distances = np.empty((queries_features.shape[0], total_size), dtype=np.float32)

            for batch_start in tqdm(range(0, total_size, 512)):
                batch_end = min(batch_start + 521, total_size)
                batch = database_descriptors[batch_start:batch_end]
                batch = torch.from_numpy(batch).to('cuda')

                for index, pred_feature in enumerate(batch):
                    global_index = batch_start + index
                    distances[:, global_index] = ((queries_features - pred_feature) ** 2).sum(1).cpu().numpy()

                del batch
            del queries_features, database_descriptors
        else: 
            all_descriptors = np.empty((len(eval_ds), args.features_dim), dtype="float32")
            for images, indices in tqdm(database_dataloader, ncols=100):
                descriptors = model(images.to("cuda"))
                descriptors = descriptors.cpu().numpy()
                all_descriptors[indices.numpy(), :] = descriptors
            database_descriptors = all_descriptors[:eval_ds.database_num]
            np.save(database_descriptors_dir, database_descriptors)
            database_descriptors = torch.from_numpy(database_descriptors)
            database_descriptors = database_descriptors.to('cuda')

            for index, pred_feature in enumerate(database_descriptors):
                    distances[:, index] = ((queries_features - pred_feature) ** 2).sum(1).cpu().numpy()
            del features, queries_features, pred_feature, database_descriptors

    predictions = distances.argsort(axis=1)[:, :max(args.recall_values)]
    del distances

    positives_per_query = eval_ds.get_positives()

    recalls = np.zeros(len(args.recall_values))
    for query_index, pred in enumerate(predictions):
        for i, n in enumerate(args.recall_values):
            if np.any(np.in1d(pred[:n], positives_per_query[query_index])):
                recalls[i:] += 1
                break

    recalls = recalls / eval_ds.queries_num * 100
    recalls_str = ", ".join([f"R@{val}: {rec:.1f}" for val, rec in zip(args.recall_values, recalls)])

    # Save visualizations of predictions
    if args.num_preds_to_save != 0:
        logging.info("Saving final predictions")
        # For each query save num_preds_to_save predictions
        visualizations.save_preds(predictions[:, :args.num_preds_to_save], eval_ds,
                                args.save_dir, args.save_only_wrong_preds)
        
    return recalls, recalls_str

def test(args, eval_ds, model):
    """Compute features of the given dataset and compute the recalls."""
    
    if args.efficient_ram_testing:
        return test_efficient_ram_usage(args, eval_ds, model)
    
    model = model.eval()
    with torch.no_grad():
        logging.debug("Extracting database features for evaluation/testing")
        # For database use "hard_resize", although it usually has no effect because database images have same resolution
        database_subset_ds = Subset(eval_ds, list(range(eval_ds.database_num)))
        database_dataloader = DataLoader(dataset=database_subset_ds, num_workers=args.num_workers,
                                         batch_size=args.infer_batch_size, pin_memory=True)
        all_features = np.empty((len(eval_ds), args.features_dim), dtype="float32")

        for inputs, indices in tqdm(database_dataloader, ncols=100):
            embeddings, features = model(inputs.to("cuda"))
            embeddings = embeddings.cpu().numpy()
            all_features[indices.numpy(), :] = embeddings
        
        # print(model.all_time / eval_ds.database_num)
        
        logging.debug("Extracting queries features for evaluation/testing")
        queries_infer_batch_size = args.infer_batch_size
        # queries_infer_batch_size = 1
        queries_subset_ds = Subset(eval_ds, list(range(eval_ds.database_num, eval_ds.database_num+eval_ds.queries_num)))
        queries_dataloader = DataLoader(dataset=queries_subset_ds, num_workers=args.num_workers,
                                        batch_size=queries_infer_batch_size, pin_memory=True)
        for inputs, indices in tqdm(queries_dataloader, ncols=100):
            embeddings, features = model(inputs.to("cuda"))
            embeddings = embeddings.cpu().numpy()            
            all_features[indices.numpy(), :] = embeddings
    
    queries_features = all_features[eval_ds.database_num:]
    database_features = all_features[:eval_ds.database_num]
    
    faiss_index = faiss.IndexFlatL2(args.features_dim)
    faiss_index.add(database_features)
    del database_features, all_features
    
    logging.debug("Calculating recalls")
    distances, predictions = faiss_index.search(queries_features, max(args.recall_values))
    
    # if args.dataset_name != "msls_challenge":
    #     fp = open(f"{args.dataset_name}.txt", "w")
    #     for query in range(eval_ds.queries_num):
    #         query_path = eval_ds.queries_paths[query]
    #         fp.write(query_path.split("@")[-1][:-4]+' ')
    #         for i in range(20):
    #             pred_path = eval_ds.database_paths[predictions[query,i]]
    #             fp.write(pred_path.split("@")[-1][:-4]+' ')
    #         fp.write("\n")
    #     fp.write("\n")
    #     return 0, "0"
    
    if args.dataset_name == "msls_challenge":
        fp = open("msls_challenge.txt", "w")
        for query in range(eval_ds.queries_num):
            query_path = eval_ds.queries_paths[query]
            fp.write(query_path.split("@")[-1][:-4]+' ')
            for i in range(20):
                pred_path = eval_ds.database_paths[predictions[query,i]]
                fp.write(pred_path.split("@")[-1][:-4]+' ')
            fp.write("\n")
        fp.write("\n")

    #### For each query, check if the predictions are correct
    positives_per_query = eval_ds.get_positives()
    # args.recall_values by default is [1, 5, 10, 20]
    recalls = np.zeros(len(args.recall_values))
    for query_index, pred in enumerate(predictions):
        for i, n in enumerate(args.recall_values):
            if np.any(np.in1d(pred[:n], positives_per_query[query_index])):
                recalls[i:] += 1
                break
    # Divide by the number of queries*100, so the recalls are in percentages
    recalls = recalls / eval_ds.queries_num * 100
    recalls_str = ", ".join([f"R@{val}: {rec:.2f}" for val, rec in zip(args.recall_values, recalls)])

    # Save visualizations of predictions
    if args.num_preds_to_save != 0:
        logging.info("Saving final predictions")
        # For each query save num_preds_to_save predictions
        visualizations.save_preds(predictions[:, :args.num_preds_to_save], eval_ds,
                                args.save_dir, args.save_only_wrong_preds)


    return recalls, recalls_str