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
  'views': ['Almond at Washington North', 'Almond at Washington East', 'Almond at Washington South', 'Almond at Washington West'], # list of views for training  'Almond at Washington East'
  'images_path': 'Annotations-Images', # path to RGB images
  'gt_path': 'Annotations-GT', # path to ground-truth segmentation masks
  'log_path': 'logs' # path to directory you make for saving results
}

# use buffalo grove as test dataset
# 20 total vs 20 per view vs 80 total
# start at really low inital point
# num images per camera view

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
n_epochs = [50, 100, 150, 200, 250, 300, 350, 400]

for n in n_epochs:
  # set up model
  n_channels = 3 # RGB images
  model = UNet(n_channels)
  
  config['n_epochs'] = n
  act = ActiveLearning(model=model, n_init_per_view=20, n_train_per_view=40, datasets=datasets, config=config, loss_type="basicLoss")
  accuracies = act.run_loop(10)
  act.visualize_losses()

# act2 = ActiveLearning(model=model, n_init=20, n_train_per_view=40, datasets=datasets, config=config, loss_type="basicLossExtra")
# accuracies = act2.run_loop(10)
# act2.visualize_losses()
