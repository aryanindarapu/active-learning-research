import torch
import torch.nn as nn
import os
import numpy as np
import json

from torch.utils.data import Dataset
from skimage.io import imread
from skimage.transform import resize
from skimage.measure import label
from tqdm import tqdm

class BFSEvaluationDataset(Dataset):
    def __init__(self, images_path, gt_path, views, input_shape, output_shape):
        self.images_path = images_path
        self.top_trim = 15
        self.gt_path = gt_path
        self.views = views
        self.input_shape, self.output_shape = input_shape, output_shape 
        initial_file_paths = []
        for v in views:
            view_path = os.path.join(gt_path, v)
            if os.path.exists(view_path):
                for date in os.listdir(view_path):
                    date_path = os.path.join(view_path, date)
                    initial_file_paths += [os.path.join(v, date, n) for n in os.listdir(date_path)]
        # load images and targets
        self.file_paths = []
        self.images, self.targets = [], []
        for f in initial_file_paths:
            full_image_path = os.path.join(self.images_path, f.replace('.png', '.jpg'))
            # input image
            if os.path.exists(full_image_path):
                self.file_paths.append(f)
                input_image = imread(full_image_path)
                if input_image.shape[0] != self.input_shape[0] or input_image.shape[1] != self.input_shape[1]:
                    input_image = resize(input_image, self.input_shape)
                input_image[:self.top_trim] = 0
                self.images.append(torch.from_numpy(input_image).permute(2, 0, 1).float())
                # target image
                target_image = imread(os.path.join(self.gt_path, f))
                target_image[:self.top_trim] = 0
                target_image = resize(target_image, self.output_shape)
                target_image = target_image > 0
                self.targets.append(target_image)
        self.file_paths = np.array(self.file_paths)

    def get_blobs(self, target_image):
        '''
        Extract each connected component or "blob" from target_image. This enables
        per-blob metrics like blob recall.
        '''
        all_blobs = label(target_image, connectivity=1)
        blob_numbers = np.unique(all_blobs)
        blob_numbers = blob_numbers[blob_numbers>0]
        blobs = np.array([all_blobs==n for n in blob_numbers]) # (N_blobs, H, W)
        return blobs
    
    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx): 
        file_path = self.file_paths[idx]
        return self.images[idx], torch.from_numpy(self.targets[idx]).unsqueeze(0).float()

class BFSConcatDataset(Dataset):
    def __init__(self, datasets):
        '''
        datasets is a list of (BFSEvaluationDataset, indices) tuples
        '''
        self.images_path = datasets[0][0].images_path
        self.top_trim = datasets[0][0].top_trim
        self.gt_path = datasets[0][0].gt_path
        self.views = [v for d in datasets for v in d[0].views]
        self.input_shape, self.output_shape = datasets[0][0].input_shape, datasets[0][0].output_shape 
        self.images = [d[0][i][0] for d in datasets for i in d[1]]
        self.targets = [d[0][i][1] for d in datasets for i in d[1]]

    def get_blobs(self, target_image):
        '''
        Extract each connected component or "blob" from target_image. This enables
        per-blob metrics like blob recall.
        '''
        all_blobs = label(target_image, connectivity=1)
        blob_numbers = np.unique(all_blobs)
        blob_numbers = blob_numbers[blob_numbers>0]
        blobs = np.array([all_blobs==n for n in blob_numbers]) # (N_blobs, H, W)
        return blobs
    
    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        return self.images[idx], self.targets[idx]
