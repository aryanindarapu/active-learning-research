import numpy as np
from dataset import *
from supervised_train import *
import torch
from matplotlib import pyplot as plt
import logging

class ActiveLearning:
  def __init__(self, model, n_init, n_train_per_view, datasets, config, loss_type):
    self.model_accuracies = []
    self.n_init = n_init
    self.config = config
    self.datasets = datasets
    self.model = model
    self.filename = 0
    self.lossType = loss_type
    
    logging.basicConfig(filename=f"{loss_type}.log",
                        filemode='a',
                        format='%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s',
                        datefmt='%H:%M:%S',
                        level=logging.DEBUG)

    training_datasets, validation_datasets = [], []

    for d in datasets:
      train_indices = np.random.choice(np.arange(len(d)), size=n_train_per_view, replace=False)
      val_indices = np.array([i for i in range(len(d)) if i not in train_indices])

      training_datasets.append((d, train_indices))
      validation_datasets.append((d, val_indices))

    self.train_dataset = BFSConcatDataset(training_datasets) # full training dataset, i.e. every image that the model will end up training
    self.val_dataset = BFSConcatDataset(validation_datasets) # full validation dataset; should stay the same throughout training process

    all_train_indices = np.arange(len(self.train_dataset)) # can't use train_indices because it's per view
    
    np.random.shuffle(all_train_indices)
    self.train_indices = all_train_indices[:n_init]
    self.remaining_indices = all_train_indices[n_init:]

    # train with first n_init images
    self.model, self.stat_dict = train(self.model, self.train_dataset_preprocess(), self.val_dataset, self.config, self.train_dataset)

  def train_dataset_preprocess(self):
    '''Trims the full training dataset to just the indices in self.train_indices.'''
    dataset = []
    for i in self.train_indices:
      dataset.append(self.train_dataset[i])

    return dataset
  
  def acquisition(self, num_new_imgs):
    '''Returns the indices of self.remaining_indices that have the highest uncertainty.'''
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
          return np.random.choice(self.remaining_indices, num_new_imgs, replace=False)

        uncertains = np.append(uncertains, loss)

    # print("Uncertains: ", uncertains)
    print("Uncertains: ", uncertains)
    new_training_indices = np.argpartition(uncertains, -num_new_imgs)[-num_new_imgs:] # indices of remaining indices with highest uncertainty rates
    # print("Chosen uncertainties: ", new_training_indices)
    print("Chosen uncertanties: ", new_training_indices)
    
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

  def evaluate_model(self):
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    thresholds = [0.5] # only checking one foreground probability threshold
    test_precision, test_recall, test_f_measure, test_blob_recall = model_metrics(self.val_dataset, self.model, thresholds, device, self.val_dataset)
    return test_precision[0], test_recall[0], test_f_measure[0], test_blob_recall[0]

  def run_loop(self, step_size):
    print("running loop")
    # device = 'mps' if torch.backends.mps.is_built() else 'cuda:0' if torch.cuda.is_available() else 'cpu'
    
    # training loop with acquisition function
    for curr_loop in range(self.n_init, len(self.train_dataset), step_size):
      # get current accuracy, recall, and other metrics
      p, r, f, b = self.evaluate_model()
      print(f'Metrics with {curr_loop} training images')
      print('Precision = {:.3f} | Recall = {:.3f} | F-measure = {:.3f} | Blob recall = {:.3f}'.format(p, r, f, b))
      print('')
      
      logging.debug(f'Metrics with {curr_loop} training images')
      logging.debug('Precision = {:.3f} | Recall = {:.3f} | F-measure = {:.3f} | Blob recall = {:.3f}'.format(p, r, f, b))
      logging.debug('')

      self.model_accuracies.append((curr_loop, (p, r, f, b)))
      self.current_pass = curr_loop
      # run acquisition function
      new_training_indices = self.acquisition(step_size)
    
      # add new images to dataset and retrain
      self.train_indices = np.append(self.train_indices, self.remaining_indices[new_training_indices])
      self.remaining_indices = np.delete(self.remaining_indices, new_training_indices)      

      self.model, self.stat_dict = train(self.model, self.train_dataset_preprocess(), self.val_dataset, self.config, self.train_dataset)
      # break
      
    p, r, f, b = self.evaluate_model()
    print(f'Metrics with {curr_loop} training images')
    print('Precision = {:.3f} | Recall = {:.3f} | F-measure = {:.3f} | Blob recall = {:.3f}'.format(p, r, f, b))
    print('')
    
    logging.debug(f'Metrics with {curr_loop} training images')
    logging.debug('Precision = {:.3f} | Recall = {:.3f} | F-measure = {:.3f} | Blob recall = {:.3f}'.format(p, r, f, b))
    logging.debug('')
    
    self.model_accuracies.append((curr_loop, (p, r, f, b)))