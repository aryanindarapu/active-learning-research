from torch.utils.tensorboard import SummaryWriter

import torch
import yaml
import json
import os
import numpy as np
import timeit

from argparse import ArgumentParser
from yaml import Loader
from torch.utils.data import DataLoader, ConcatDataset, Subset
from torch import optim
from unet_model import UNet
from dataset import *
from tqdm import tqdm
from quantitative_results import *

def train(model, train_dataset, val_dataset, config, full_dataset):
    # set up dataloaders
    batch_size = config['batch_size']
    train_loader = DataLoader(train_dataset, batch_size=batch_size, num_workers=2, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, num_workers=2) 
    
    thresholds = np.array([0.5])

    # set up criteria
    recon_criterion = nn.BCELoss() 
     
    # set up optimizer
    lr = float(config['lr'])
    n_epochs = config['n_epochs']
    if config['crop_size'] != 'None':
        crop_size = int(config['crop_size'])
        max_x = config['input_shape'][1]-crop_size
        max_y = config['input_shape'][0]-crop_size 
    else:
        crop_size = None

    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    model = model.to(device)
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=1e-4)
    milestones = [int(n_epochs*m) for m in config['milestones']]
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones, gamma=0.1)
 
    train_losses, train_precisions, train_recalls, train_f_measures, train_blob_recalls = [], [], [], [], []
    val_precisions, val_recalls, val_f_measures, val_blob_recalls = [], [], [], []
    for epoch in tqdm(range(n_epochs)):
        train_total = 0
        n_images = 0
        model.train()
        model = model.to(device)
        for images, targets in train_loader:
            optimizer.zero_grad()
            if crop_size is not None:
                x_coord = np.random.choice(np.arange(max_x))
                y_coord = np.random.choice(np.arange(max_y))
                images = images[:, :, y_coord:y_coord+crop_size, x_coord:x_coord+crop_size]
                targets = targets[:, :, y_coord:y_coord+crop_size, x_coord:x_coord+crop_size]
            images, targets = images.to(device), targets.to(device)
            n_images += images.size(0)
            preds = model(images)
            loss = recon_criterion(preds, targets)
            train_total += loss.item()
            loss.backward()
            optimizer.step()
            
        # logging

        # train metrics
        train_losses.append(train_total/n_images) 
        train_precision, train_recall, train_f_measure, train_blob_recall = model_metrics(train_dataset, model, thresholds, device, full_dataset)
        train_precisions.append(train_precision[0])
        train_recalls.append(train_recall[0])
        train_f_measures.append(train_f_measure[0])
        train_blob_recalls.append(train_blob_recall[0]) 
        
        # val metrics
        val_precision, val_recall, val_f_measure, val_blob_recall = model_metrics(val_dataset, model, thresholds, device, val_dataset)
        val_precisions.append(val_precision[0])
        val_recalls.append(val_recall[0])
        val_f_measures.append(val_f_measure[0])
        val_blob_recalls.append(val_blob_recall[0])
        scheduler.step() 

    stat_dict = {'Training': {}, 'Validation': {}}
    stat_dict['Training']['Loss'] = train_losses
    stat_dict['Training']['Precision'] = train_precisions
    stat_dict['Training']['Recall'] = train_recalls
    stat_dict['Training']['F-measure'] = train_f_measures
    stat_dict['Training']['Blob recall'] = train_blob_recalls
    stat_dict['Validation']['Precision'] = val_precisions
    stat_dict['Validation']['Recall'] = val_recalls
    stat_dict['Validation']['F-measure'] = val_f_measures
    stat_dict['Validation']['Blob recall'] = val_blob_recalls
    return model, stat_dict

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--config_path', type=str, help='path to YAML config file')
    args = parser.parse_args()

    with open(args.config_path, 'r') as f:
        config = yaml.load(f, Loader=Loader)
    tag = config['tag'] 
    for seed in config['seeds']:
        config['seed'] = seed
        logging_path = os.path.join(config['log_path'], '{}_seed{}'.format(tag, config['seed']))
        writer = SummaryWriter(logging_path)
        n_channels = 3 # RGB images
        model = UNet(n_channels, 1)
        
        # set up datasets
        input_shape, output_shape = config['input_shape'], config['output_shape']
        np.random.seed(seed) 
        views = config['views'] 

        # make initial datasets for each view
        print('Loading datasets...')
        start = timeit.default_timer()
        datasets = []
        for v in views:
            datasets.append(BFSEvaluationDataset(config['images_path'], config['gt_path'], [v], config['input_shape'], config['output_shape']))
        # split each view into training and validation sets
        training_datasets, validation_datasets = [], []
        n_train = config['n_train'] # number of training images per view, remainder are validation
        save_train_indices = []
        save_val_indices = []
        for d in datasets:
            train_indices = np.random.choice(np.arange(len(d)), size=n_train, replace=False)
            val_indices = np.array([i for i in range(len(d)) if i not in train_indices])
            save_train_indices.append([int(i) for i in train_indices])
            save_val_indices.append([int(i) for i in val_indices])
            training_datasets.append((d, train_indices))
            validation_datasets.append((d, val_indices))
        # combine into one training, one validation dataset
        config['train_indices'] = save_train_indices
        config['validation_indices'] = save_val_indices
        train_dataset = BFSConcatDataset(training_datasets)
        val_dataset = BFSConcatDataset(validation_datasets) 
        with open(os.path.join(logging_path, 'config.json'), 'w') as f:
            json.dump(config, f)
        print('Datasets loaded in {:.1f}s'.format(timeit.default_timer()-start)) 
        writer.add_text('Number Training Images', str(len(train_dataset)))
        writer.add_text('Number Validation Images', str(len(val_dataset)))
        model, writer = train(model, train_dataset, val_dataset, writer, config)
        
        # saving and clean-up
        torch.save(model.cpu(), os.path.join(logging_path, 'model.pt'))
