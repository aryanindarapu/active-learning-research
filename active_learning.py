import numpy as np
from dataset import *
from supervised_train import *
import torch

class ActiveLearning:
  def __init__(self, model, n_init, n_train_per_view, n_val_per_view, datasets, config):
    self.model_accuracies = []
    self.n_init = n_init
    self.config = config
    self.datasets = datasets
    self.model = model

    training_datasets, validation_datasets, testing_datasets = [], [], []

    for d in datasets:
      curr_train_indices = np.random.choice(np.arange(len(d)), size=n_train_per_view, replace=False)
      val_segment = len(curr_train_indices) - n_val_per_view

      train_indices, val_indices = curr_train_indices[:val_segment], curr_train_indices[val_segment:]
      test_indices = np.array([i for i in range(len(d)) if i not in curr_train_indices])

      training_datasets.append((d, train_indices))
      validation_datasets.append((d, val_indices))
      testing_datasets.append((d, test_indices))

    self.train_dataset = BFSConcatDataset(training_datasets) # full training dataset, i.e. every image that the model will end up training
    self.val_dataset = BFSConcatDataset(validation_datasets) # full validation dataset; should stay the same throughout training process
    self.test_dataset = BFSConcatDataset(testing_datasets)   # full testing dataset; should stay the same throughout active learning process

    all_train_indices = np.arange(len(self.train_dataset))
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
  
  def acquisition(self):
    '''Returns the indices of self.remaining_indices that have the highest uncertainty.'''
    uncertains = np.array([], dtype='int32')

    device = 'mps' if torch.backends.mps.is_built() else 'cuda:0' if torch.cuda.is_available() else 'cpu'
    # test model and get images
    with torch.no_grad():
      for n in self.remaining_indices:
        image, gt = self.train_dataset[n]
        # image_numpy = image.permute(1, 2, 0).numpy()
        image_tensor = image.unsqueeze(0).to(device)
        prediction_tensor = self.model(image_tensor)
        
        test_tensor = prediction_tensor.cpu().data.numpy()[0, 0, :, :]

        loss = self.basic_loss(test_tensor)
        uncertains = np.append(uncertains, loss)

    new_training_indices = np.argpartition(uncertains, -5)[-5:] # indices of remaining indices with highest uncertainty rates

    return new_training_indices


  def basic_loss(self, tensor):
    return np.count_nonzero(np.logical_and(tensor > 0.3, tensor < 0.7))

  # def nll_loss(self, tensor):

  def get_model_accuracies(self):
    return self.model_accuracies

  def evaluate_model(self):
    device = 'mps' if torch.backends.mps.is_built() else 'cuda:0' if torch.cuda.is_available() else 'cpu'
    thresholds = [0.5] # only checking one foreground probability threshold
    test_precision, test_recall, test_f_measure, test_blob_recall = model_metrics(self.test_dataset, self.model, thresholds, device, self.test_dataset)
    return test_precision[0], test_recall[0], test_f_measure[0], test_blob_recall[0]

  def run_loop(self, step_size):
    print("running loop")
    device = 'mps' if torch.backends.mps.is_built() else 'cuda:0' if torch.cuda.is_available() else 'cpu'

    # training loop with acquisition function
    for curr_loop in range(self.n_init, len(self.train_dataset), step_size):
      # get current accuracy, recall, and other metrics
      p, r, f, b = self.evaluate_model()
      print(f'Metrics with {curr_loop} training images')
      print('Precision = {:.3f} | Recall = {:.3f} | F-measure = {:.3f} | Blob recall = {:.3f}'.format(p, r, f, b))
      print('')

      self.model_accuracies.append((curr_loop, (p, r, f, b)))

      # run acquisition function
      new_training_indices = self.acquisition()
    
      # add new images to dataset and retrain
      self.train_indices = np.append(self.train_indices, self.remaining_indices[new_training_indices])
      self.remaining_indices = np.delete(self.remaining_indices, new_training_indices)      

      self.model, self.stat_dict = train(self.model, self.train_dataset_preprocess(), self.val_dataset, self.config, self.train_dataset)
      
    p, r, f, b = self.evaluate_model()
    print(f'Metrics with {curr_loop} training images')
    print('Precision = {:.3f} | Recall = {:.3f} | F-measure = {:.3f} | Blob recall = {:.3f}'.format(p, r, f, b))
    print('')
    
    self.model_accuracies.append((curr_loop, (p, r, f, b)))