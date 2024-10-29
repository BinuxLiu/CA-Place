import os
import numpy as np
from glob import glob
from PIL import Image
from sklearn.neighbors import NearestNeighbors

import torchvision.transforms as transforms
import torch.utils.data as data

IMAGENET_MEAN_STD = {'mean': [0.485, 0.456, 0.406],
                     'std': [0.229, 0.224, 0.225]}

class BaseDataset(data.Dataset):
    """Dataset with images from database and queries, used for inference (testing and building cache).
    """
    def __init__(self, args, split="train"):
        super().__init__()
        self.args = args
        self.dataset_name = args.dataset_name
        self.dataset_folder = os.path.join(args.datasets_folder, self.dataset_name, "images", split)
        self.queries_name = args.queries_name if args.queries_name != None else "queries"
        self.resize = args.resize
        
        #### Read paths and UTM coordinates for all images.
        database_folder = os.path.join(self.dataset_folder, "database")
        queries_folder = os.path.join(self.dataset_folder, self.queries_name)

        self.database_paths = sorted(glob(os.path.join(database_folder, "**", "*.jpg"), recursive=True))
        self.queries_paths = sorted(glob(os.path.join(queries_folder, "**", "*.jpg"),  recursive=True))
        # The format must be path/to/file/@utm_easting@utm_northing@...@.jpg
        self.database_utms = np.array([(path.split("@")[1], path.split("@")[2]) for path in self.database_paths]).astype(float)
        self.queries_utms = np.array([(path.split("@")[1], path.split("@")[2]) for path in self.queries_paths]).astype(float)
        
        # Find soft_positives_per_query, which are within val_positive_dist_threshold (deafult 25 meters)
        knn = NearestNeighbors(n_jobs=-1)
        knn.fit(self.database_utms)
        self.soft_positives_per_query = knn.radius_neighbors(self.queries_utms,
                                                             radius=args.val_positive_dist_threshold,
                                                             return_distance=False)
        
        self.soft_positives_per_database = knn.radius_neighbors(self.database_utms,
                                                             radius=5,
                                                             return_distance=False)
        
        self.images_paths = list(self.database_paths) + list(self.queries_paths)
        
        self.database_num = len(self.database_paths)
        self.queries_num = len(self.queries_paths)

        self.transform = transforms.Compose([
            transforms.Resize(args.resize, interpolation=transforms.InterpolationMode.BILINEAR) if args.resize_test_imgs else lambda x: x,
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN_STD['mean'], std=IMAGENET_MEAN_STD['std'])
        ])
    
    def __getitem__(self, index):
        img = Image.open(self.images_paths[index]).convert("RGB")
        img = self.transform(img)

        return img, index
    
    def __len__(self):
        return len(self.images_paths)
    
    def __repr__(self):
        return f"< {self.__class__.__name__}, {self.dataset_name} - #database: {self.database_num}; #queries: {self.queries_num} >"
    
    def get_positives(self):
        return self.soft_positives_per_query
    
    def get_positives_database(self):
        return self.soft_positives_per_database