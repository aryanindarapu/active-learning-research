import numpy as np
import matplotlib.pyplot as plt

from unet_model import UNet
from dataset import *
from quantitative_results import *

config = {
  'seed': 310, # random seed
  'input_shape': (240, 360), # size of input images (height, width)
  'output_shape': (240, 360), # size of output target
  'n_train': 40, # number of training images per view
  'batch_size': 2, # number of images per optimization step
  'lr': 1e-2, # learning rate
  'n_epochs': 50, # number of passes through training data
  # optionally perform random cropping, specify integer < max(H, W) to use cropping for training
  'crop_size': 'None',
  # fraction of training epochs to lower learning rate by 1/10, e.g. [0.6, 0.8]
  # lowers learning rate at epochs 120 and 160 if we have 200 training epochs
  'milestones': [0.8],
  'views': ['Almond at Washington North', 'Almond at Washington East'], # list of views for training  'Almond at Washington East'
  'images_path': 'Annotations-Images', # path to RGB images
  'gt_path': 'Annotations-GT', # path to ground-truth segmentation masks
  'log_path': 'x' # path to directory you make for saving results
}

# set up model
n_channels = 3 # RGB images
model = UNet(n_channels)

# set up datasets
seed = config['seed']
np.random.seed(seed) 
views = config['views']
logging_path = config['log_path']
if not os.path.exists(logging_path):
    os.mkdir(logging_path)

# set up model
n_channels = 3 # RGB images
model = UNet(n_channels)

# set up datasets
seed = config['seed']
np.random.seed(seed) 
views = config['views']
logging_path = config['log_path']
if not os.path.exists(logging_path):
    os.mkdir(logging_path)

# make initial datasets for each view
print('Loading datasets...')
datasets = []
for v in views:
    datasets.append(BFSEvaluationDataset(config['images_path'],
                                         config['gt_path'],
                                         [v],
                                         config['input_shape'],
                                         config['output_shape']))
    
from active_learning import *

act = ActiveLearning(model=model, n_init=20 * len(config['views']), n_train_per_view=40, datasets=datasets, config=config, logname="basicLoss.log")
accuracies = act.run_loop(10)

act = ActiveLearning(model=model, n_init=20 * len(config['views']), n_train_per_view=40, datasets=datasets, config=config, logname="groupLoss.log")
accuracies = act.run_loop(10)