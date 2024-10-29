import faiss
import logging
import numpy as np
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Subset

from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.signal import savgol_filter

logging.getLogger('matplotlib').setLevel(logging.ERROR)
logging.getLogger('PIL').setLevel(logging.ERROR)

def test(args, eval_ds, model):

    model = model.eval()
    feature_maps = []
    with torch.no_grad():
        logging.debug("Extracting database features for evaluation/testing")
        database_subset_ds = Subset(eval_ds, list(range(eval_ds.database_num)))
        database_dataloader = DataLoader(dataset=database_subset_ds, num_workers=args.num_workers,
                                         batch_size=args.infer_batch_size, pin_memory=True)
        all_attn = []
        all_feat = []
        for inputs, indices in tqdm(database_dataloader, ncols=100):
            B, C, H, W = inputs.shape
            feat, attn_feat = model(inputs.to("cuda"))
            # print(feat.shape, attn_feat.shape)
            all_attn.append(attn_feat.detach().cpu().squeeze())
            all_feat.append(feat.detach().cpu().squeeze())

        all_attn = torch.stack(all_attn)
        mean_attn = all_attn.mean(0)
        std_attn = all_attn.std(0).numpy()

        all_feat = torch.stack(all_feat)
        mean_feat = all_feat.mean(0)
        std_feat = all_feat.std(0).numpy()

        channel_ids = torch.arange(feat.shape[1])

        smooth_mean_attn = savgol_filter(mean_attn, 25, 3)
        smooth_mean_feat = savgol_filter(mean_feat, 25, 3)
        print()
        plt.figure(figsize=(12, 4))
        # plt.plot(channel_ids, smooth_mean_attn, label='G2M', color='salmon', linewidth=1)
        # plt.fill_between(channel_ids, smooth_mean_attn - std_attn, smooth_mean_attn + std_attn, color='salmon', alpha=0.4)
        plt.plot(channel_ids, smooth_mean_feat, label='G2M', color='blue', linewidth=1)
        plt.fill_between(channel_ids, smooth_mean_feat - std_feat, smooth_mean_feat + std_feat, color='blue', alpha=0.2)
        plt.xlabel('Channel ID')
        plt.ylabel('Channel Value')
        plt.legend()
        plt.subplots_adjust(left=0.08, right=0.92)
        plt.savefig(f"attention.png")