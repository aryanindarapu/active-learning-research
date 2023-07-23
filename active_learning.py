import numpy as np
from dataset import *
from supervised_train import *
import torch
from matplotlib import pyplot as plt
import logging
from collections import Counter
import random

class ActiveLearning:
  def __init__(self, model, datasets, test_datasets, config, loss_type='baseline', per_view=False):
    self.model_accuracies = []
    self.config = config
    self.datasets = datasets
    self.model = model
    self.filename = 0
    self.lossType = loss_type
    
    self.training_losses = {}
    self.validation_losses = {}
    self.testing_losses = {}
    
    logging.basicConfig(filename=f"{config['log_path']}/{loss_type}.log",
                        filemode='a',
                        format='%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s',
                        datefmt='%H:%M:%S',
                        level=logging.DEBUG)

    plt.set_loglevel (level = 'warning')
    pil_logger = logging.getLogger('PIL')
    # override the logger logging level to INFO
    pil_logger.setLevel(logging.INFO)

    training_datasets, validation_datasets, testing_datasets = [], [], []

    if per_view:
      self.n_init = config['train_details']['n_init_per_view'] * len(config['views'])
      for d in datasets:
        train_indices = np.random.choice(np.arange(len(d)), size=config['train_details']['n_train_per_view'], replace=False)
        val_indices = np.array([i for i in range(len(d)) if i not in train_indices])

        training_datasets.append((d, train_indices))
        validation_datasets.append((d, val_indices))
        
      for d in test_datasets:
        test_indices = np.arange(len(d))
        testing_datasets.append((d, test_indices))
    else:
      self.n_init = config['train_details']['n_init']
      
      ##### Determine which images to homogenously get from dataset
      self.n_train_views = []
      for idx, d in enumerate(datasets):
        self.n_train_views.extend(len(d) * [idx])
        
      c = Counter(random.sample(self.n_train_views, len(self.n_train_views))[:config['train_details']['n_train']]) # determines how many images to include from each batch in the training set
      #####
      
      for idx, d in enumerate(datasets):
        train_indices = np.random.choice(np.arange(len(d)), size=c[idx], replace=False)
        val_indices = np.array([i for i in range(len(d)) if i not in train_indices])

        training_datasets.append((d, train_indices))
        validation_datasets.append((d, val_indices))
        
      for d in test_datasets:
        test_indices = np.arange(len(d))
        testing_datasets.append((d, test_indices))
      

    self.train_dataset = BFSConcatDataset(training_datasets) # full training dataset, i.e. every image that the model will end up training
    self.val_dataset = BFSConcatDataset(validation_datasets) # full validation dataset; should stay the same throughout training process
    self.test_dataset = BFSConcatDataset(testing_datasets) # full testing dataset; should stay the same throughout training process

    # takes all indices from dataset 
    all_train_indices = np.arange(len(self.train_dataset)) # can't use train_indices because it's per view
    
    np.random.shuffle(all_train_indices)
    self.train_indices = all_train_indices[:self.n_init]
    self.remaining_indices = all_train_indices[self.n_init:]
    # train with first n_init images
    self.model, stat_dict = train(self.model, self.train_dataset_preprocess(), self.val_dataset, self.config, self.train_dataset)
    
    # get current accuracy, recall, and other metrics
    self.evaluate_model(self.n_init)
    
    c = Counter([self.n_train_views[i] for i in self.train_indices]) # determines how many images to include from each batch in the training set
    for view in c:
      logging.debug(f'{self.config["views"][view]}: {c[view]} images')
    
    self.training_losses[self.n_init] = stat_dict['Training']['F-measure']
    self.validation_losses[self.n_init] = stat_dict['Validation']['F-measure']
    # print(self.validation_losses)
    self.epochs = stat_dict['Epochs']

  def train_dataset_preprocess(self):
    '''Trims the full training dataset to just the indices in self.train_indices.'''
    dataset = []
    for i in self.train_indices:
      dataset.append(self.train_dataset[i])

    return dataset 
  
  def acquisition(self, num_new_imgs):
    '''Returns the *indices* of self.remaining_indices that have the highest uncertainty.'''
    uncertains = np.array([], dtype='int32')
    
    self.filename = 0
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    
    # test model and get images
    with torch.no_grad():
      for n in self.remaining_indices:
        image, gt = self.train_dataset[n]
        image_tensor = image.unsqueeze(0).to(device)
        prediction_tensor = self.model(image_tensor)
        test_tensor = prediction_tensor.cpu().data.numpy()[0, 0, :, :]
        
        # plt.imsave(f"./uncertainty_images/{self.current_pass}_{self.filename}_gt.png", gt.squeeze(0).numpy())
        # plt.imsave(f"./uncertainty_images/{self.current_pass}_{self.filename}_original.png", image.permute(1, 2, 0).numpy()) 
        
        if self.lossType == "basicLoss":
          loss = self.basic_loss(test_tensor)
        elif self.lossType == "basicLossExtra":
          loss = self.basic_loss_extra(test_tensor)
        else:
          # randomly select pixels to be uncertain
          return np.random.choice(len(self.remaining_indices), size=num_new_imgs)

        uncertains = np.append(uncertains, loss)

    # print("Uncertains: ", uncertains)
    new_training_indices = np.argpartition(uncertains, -num_new_imgs)[-num_new_imgs:] # indices of remaining indices with highest uncertainty rates
    print("Chosen uncertainties: ", new_training_indices)
    
    return new_training_indices

  def basic_loss(self, tensor):
    uncertain_tensor = np.logical_and(tensor > 0.3, tensor < 0.7)
    # plt.imsave(f"./uncertainty_images/{self.current_pass}_{self.filename}.png", uncertain_tensor) 
    self.filename += 1

    return np.count_nonzero(uncertain_tensor)
  
  def basic_loss_extra(self, tensor):
    uncertain_tensor = np.logical_and(tensor > 0.3, tensor < 0.7) # uncertainty binary image
    # for every uncertain pixel (i.e. uncertain_tensor == 1), sum number of uncertain pixels 3 units away in every direction 
    updated_tensor = np.zeros_like(uncertain_tensor)
    uncertain_idx_list = zip(*uncertain_tensor.nonzero())
    
    for pair in uncertain_idx_list:
      image_size_width = len(uncertain_tensor[0])
      image_size_height = len(uncertain_tensor)
      
      topBorder, botBorder = 0 if pair[0]-3 < 0 else pair[0]-3, image_size_height if pair[0]+3 > image_size_height-1 else pair[0]+4
      leftBorder, rightBorder = 0 if pair[1]-3 < 0 else pair[1]-3, image_size_width if pair[1]+3 > image_size_width-1 else pair[1]+4
      
      updated_tensor[pair[0], pair[1]] = np.count_nonzero(uncertain_tensor[topBorder:botBorder, leftBorder:rightBorder])
      # print(test[topBorder:botBorder, leftBorder:rightBorder])
      
    # self.filename += 1
    # print(updated_tensor)
    return np.count_nonzero(updated_tensor)

  # def nll_loss(self, tensor):

  def get_model_accuracies(self):
    return self.model_accuracies

  def evaluate_model(self, curr_loop):
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    thresholds = [0.5] # only checking one foreground probability threshold
    # val_precision, val_recall, val_f_measure, val_blob_recall = model_metrics(self.val_dataset, self.model, thresholds, device, self.val_dataset)
    # p, r, f, b = val_precision[0], val_recall[0], val_f_measure[0], val_blob_recall[0]
    
    test_precision, test_recall, test_f_measure, test_blob_recall = model_metrics(self.test_dataset, self.model, thresholds, device, self.test_dataset)
    p, r, f, b = test_precision[0], test_recall[0], test_f_measure[0], test_blob_recall[0]

    self.testing_losses[curr_loop] = f
    
    logging.debug(f'Metrics with {curr_loop} training images')
    logging.debug('Precision = {:.3f} | Recall = {:.3f} | F-measure = {:.3f} | Blob recall = {:.3f}'.format(p, r, f, b))

  def run_loop(self, step_size):
    print("running loop")
    # device = 'mps' if torch.backends.mps.is_built() else 'cuda:0' if torch.cuda.is_available() else 'cpu'
    
    # training loop with acquisition function
    for curr_loop in range(self.n_init + step_size, len(self.train_dataset) + 1, step_size):
      self.current_pass = curr_loop
      # run acquisition function
      new_training_indices = self.acquisition(step_size)
      
      # add new images to dataset and retrain
      self.train_indices = np.append(self.train_indices, self.remaining_indices[new_training_indices])
      self.remaining_indices = np.delete(self.remaining_indices, new_training_indices)
      
      self.model, stat_dict = train(self.model, self.train_dataset_preprocess(), self.val_dataset, self.config, self.train_dataset)
      
      self.training_losses[curr_loop] = stat_dict['Training']['F-measure']
      self.validation_losses[curr_loop] = stat_dict['Validation']['F-measure']
      
      # print(self.validation_losses)
      # get current accuracy, recall, and other metrics
      self.evaluate_model(curr_loop)
      
      # Determines which views the selecting indices are from
      c = Counter([self.n_train_views[i] for i in new_training_indices]) # determines how many images to include from each batch in the training set
      for view in c:
        logging.debug(f'{self.config["views"][view]}: {c[view]} images')
        
      logging.debug('')
    
    # self.model_accuracies.append((curr_loop, (p, r, f, b)))
    
  def visualize_losses(self):
    plt.figure()
    plt.xlabel("num epochs")
    plt.ylabel("f-measure")
    
    
    for num_images, data in self.training_losses.items():
      plt.plot(self.epochs, data, label=f'{num_images} images')
    plt.legend()
    plt.ylim(0, 1)
    plt.savefig(f"training_losses_{self.config['n_epochs']}_{self.lossType}.png")
    plt.ylim(0.92, 1)
    plt.savefig(f"training_losses_{self.config['n_epochs']}_{self.lossType}_zoom.png")
    
    plt.cla()
    plt.xlabel("num epochs")
    plt.ylabel("f-measure")
  
    for num_images, data in self.validation_losses.items():
      plt.plot(self.epochs, data, label=f'{num_images} images')
    plt.legend()
    plt.ylim(0, 1)
    plt.savefig(f"validation_losses_{self.config['n_epochs']}_{self.lossType}.png")
    plt.ylim(0.85, 1)
    plt.savefig(f"validation_losses_{self.config['n_epochs']}_{self.lossType}_zoom.png")
    
    plt.cla()
    plt.xlabel("num images")
    plt.ylabel("f-measure")
    
    loop_data, test_accuracies = list(self.testing_losses.keys()), list(self.testing_losses.values())
    plt.scatter(loop_data, test_accuracies, label=f'num images')
    plt.legend()
    plt.savefig(f"testing_losses_{self.config['n_epochs']}_{self.lossType}.png")
    