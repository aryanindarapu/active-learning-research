import numpy as np
import torch
import json
import os

from argparse import ArgumentParser
from torch.utils.data import DataLoader
from dataset import BFSEvaluationDataset
from tqdm import tqdm
from metrics import *
from skimage.io import imread
from tqdm import tqdm
from copy import deepcopy

def model_metrics(dataset, model, thresholds, device): 
    # iterate through valid indices in dataset to assemble gt-prediction pairs
    gt_targets = np.array([deepcopy(dataset[i][1].squeeze(0)).numpy() for i in range(len(dataset))])
    blob_targets = [dataset.get_blobs(gt) for gt in gt_targets]
    predictions = []
    
    with torch.no_grad():
        for i in range(len(dataset)):
            image = dataset[i][0].unsqueeze(0).to(device)
            pred = model(image).squeeze(0).cpu().numpy()
            predictions.append(pred)
            
    predictions = np.array(predictions)
    
    
    # compute metrics for multiple thresholds
    precisions, recalls, f_measures, blob_recalls = [], [], [], []
    for t in thresholds:
        t_p, t_r, t_b = 0, 0, 0
        n_gt = 0
        for i in range(len(gt_targets)):
            # make sure there is GT
            if np.sum(gt_targets[i]):
                n_gt += 1
                curr_pred = predictions[i] > t
                t_p += precision(curr_pred, gt_targets[i])
                t_r += recall(curr_pred, gt_targets[i])
                t_b += get_blob_recall(curr_pred, blob_targets[i])
        t_p /= n_gt
        t_r /= n_gt
        t_b /= n_gt
        t_f = 2*t_p*t_r/(t_p+t_r+1e-9)
        precisions.append(t_p)
        recalls.append(t_r)
        f_measures.append(t_f)
        blob_recalls.append(t_b)
    return precisions, recalls, f_measures, blob_recalls

def get_blob_recall(pred, blob_target):
    '''
    Compute blob recall
    '''
    n_blobs = blob_target.shape[0]
    n_recovered = 0
    for i in range(n_blobs):
        curr_blob = blob_target[i]
        overlap = curr_blob*pred
        if np.sum(overlap)/np.sum(curr_blob) >= 0.5:
            n_recovered += 1
    return n_recovered/n_blobs
