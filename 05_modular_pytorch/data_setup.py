
import os
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

NUM_WORKERS = os.cpu_count()

def create_dataloaders(train_dir: str,
                       test_dir: str,
                       transform: transforms.Compose,
                       batch_size: int,
                       num_workers: NUM_WORKERS):

  # load image data using ImageFolder
  train_data = datasets.ImageFolder(train_dir,
                                  transform= transform,)

  test_data = datasets.ImageFolder(root=test_dir,
                                 transform= transform)

  # get class names
  class_names = train_data.classes

  #turn image dataset into dataloaders
  train_dataloader = DataLoader(train_data,
                               batch_size = batch_size,
                               num_workers= NUM_WORKERS,
                               shuffle = True)
  test_dataloader = DataLoader(test_data,
                              batch_size = batch_size,
                              shuffle = False,
                               num_workers= NUM_WORKERS)

  return train_dataloader, test_dataloader, class_names
