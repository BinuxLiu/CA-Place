import math
import torch
import faiss
import logging
import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, SubsetRandomSampler

# based on https://github.com/lyakaap/NetVLAD-pytorch/blob/master/netvlad.py
class NetVLAD(nn.Module):
    """NetVLAD layer implementation"""

    def __init__(self, clusters_num=32, dim=128, normalize_input=True, work_with_tokens=False, linear_dim = 256, work_with_linear = False):
        """
        Args:
            clusters_num : int
                The number of clusters
            dim : int
                Dimension of descriptors
            alpha : float
                Parameter of initialization. Larger value is harder assignment.
            normalize_input : bool
                If true, descriptor-wise L2 normalization is applied to input.
        """
        super().__init__()
        self.clusters_num = clusters_num
        self.dim = dim
        self.alpha = 0
        self.normalize_input = normalize_input
        self.work_with_tokens = work_with_tokens
        self.work_with_linear = work_with_linear
        self.linear_dim = linear_dim
        # if work_with_tokens:
        #     self.conv = nn.Conv1d(dim, clusters_num, kernel_size=1, bias=False)
        # else:
        self.conv = nn.Conv2d(dim, clusters_num, kernel_size=(1, 1), bias=False)
        self.centroids = nn.Parameter(torch.rand(clusters_num, dim))

        if self.work_with_linear:
            self.feat_proj = nn.Linear(self.dim, self.linear_dim)
            
        if self.work_with_tokens:
            self.cls_proj = nn.Linear(self.dim, 256)

    def init_params(self, centroids, descriptors):
        centroids_assign = centroids / np.linalg.norm(centroids, axis=1, keepdims=True)
        dots = np.dot(centroids_assign, descriptors.T)
        dots.sort(0)
        dots = dots[::-1, :]  # sort, descending

        self.alpha = (-np.log(0.01) / np.mean(dots[0,:] - dots[1,:])).item()
        self.centroids = nn.Parameter(torch.from_numpy(centroids))
        # if self.work_with_tokens:
        #     self.conv.weight = nn.Parameter(torch.from_numpy(self.alpha * centroids_assign).unsqueeze(2))
        # else:
        self.conv.weight = nn.Parameter(torch.from_numpy(self.alpha*centroids_assign).unsqueeze(2).unsqueeze(3))
        self.conv.bias = None

    def forward(self, x):
        x, cls_token = x
        N, D, H, W = x.shape[:]
        if self.normalize_input:
            x = F.normalize(x, p=2, dim=1)  # Across descriptor dim
        x_flatten = x.view(N, D, -1)
        soft_assign = self.conv(x).view(N, self.clusters_num, -1)
        soft_assign = F.softmax(soft_assign, dim=1)
        vlad = torch.zeros([N, self.clusters_num, D], dtype=x_flatten.dtype, device=x_flatten.device)
        for D in range(self.clusters_num):  # Slower than non-looped, but lower memory usage
            residual = x_flatten.unsqueeze(0).permute(1, 0, 2, 3) - \
                    self.centroids[D:D+1, :].expand(x_flatten.size(-1), -1, -1).permute(1, 2, 0).unsqueeze(0)
            residual = residual * soft_assign[:,D:D+1,:].unsqueeze(2)
            vlad[:,D:D+1,:] = residual.sum(dim=-1)
        # print(vlad.shape)
        vlad = F.normalize(vlad, p=2, dim=2)  # intra-normalization

        if self.work_with_linear:
            vlad = F.normalize(self.feat_proj(vlad).view(N, -1), p=2, dim=-1) # input: torch.Size([32, 32, 768]) out_put: torch.Size([32, 8192])
            if self.work_with_tokens:
                cls_token_proj = F.normalize(self.cls_proj(cls_token), p=2, dim=-1)
                vlad = torch.cat([cls_token_proj, vlad], dim=-1)
        else:
            vlad = vlad.view(N, -1)  # Flatten
            vlad = F.normalize(vlad, p=2, dim=1)  # L2 normalize

        return vlad

    def initialize_netvlad_layer(self, args, cluster_ds, backbone):
        descriptors_num = 50000
        descs_num_per_image = 100
        images_num = math.ceil(descriptors_num / descs_num_per_image)
        random_sampler = SubsetRandomSampler(np.random.choice(len(cluster_ds), images_num, replace=False))
        random_dl = DataLoader(dataset=cluster_ds, num_workers=args.num_workers,
                                batch_size=args.infer_batch_size, sampler=random_sampler)
        with torch.no_grad():
            backbone = backbone.eval()
            logging.debug("Extracting features to initialize NetVLAD layer")
            descriptors = np.zeros(shape=(descriptors_num, self.dim), dtype=np.float32)
            for iteration, (inputs, _) in enumerate(tqdm(random_dl, ncols=100)):
                inputs = inputs.to("cuda")
                outputs = backbone(inputs)
                norm_outputs = F.normalize(outputs[0], p=2, dim=1)
                image_descriptors = norm_outputs.view(norm_outputs.shape[0], self.dim, -1).permute(0, 2, 1)
                image_descriptors = image_descriptors.cpu().numpy()
                batchix = iteration * args.infer_batch_size * descs_num_per_image
                for ix in range(image_descriptors.shape[0]):
                    sample = np.random.choice(image_descriptors.shape[1], descs_num_per_image, replace=False)
                    startix = batchix + ix * descs_num_per_image
                    descriptors[startix:startix + descs_num_per_image, :] = image_descriptors[ix, sample, :]
        kmeans = faiss.Kmeans(self.dim, self.clusters_num, niter=100, verbose=False)
        kmeans.train(descriptors)
        logging.debug(f"NetVLAD centroids shape: {kmeans.centroids.shape}")
        self.init_params(kmeans.centroids, descriptors)
        self = self.to("cuda")


def print_nb_params(m):
    model_parameters = filter(lambda p: p.requires_grad, m.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])
    print(f'Trainable parameters: {params/1e6:.3}M')


def main():
    import torch.cuda.amp as amp  
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)
    x = (torch.randn(1, 768, 16, 16, device=device), torch.randn(1, 768, device=device))  

    agg = NetVLAD(clusters_num=64, dim=768, normalize_input=True, work_with_tokens=True, linear_dim = 128, work_with_linear = True).to(device)

    import time
    
    print_nb_params(agg)
    start_time = time.time()
    for _ in range(3000): 
        with torch.cuda.amp.autocast():  
            output = agg(x)  
    end_time = time.time()
   
    average_time = (end_time - start_time) / 3000  
    print(f'Average time per pass: {average_time:.6f} seconds')
    print(output.shape)


if __name__ == '__main__':
    main()