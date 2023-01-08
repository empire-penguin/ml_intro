############### TRITON NEUROTECH #################
## Example dataset class for pytorch
##################################################
## Author: Gavin Roberts
## Date: Jan-07-2023
## Contributors:
## Version: 1.0.0
## Email: gsroberts@ucsd.edu
## Status: Development
##################################################

import torch
from torch.utils.data import Dataset
from torchvision import transforms
import os
from PIL import Image
import numpy as np
from consts import *

class ExampleDataset(Dataset):
    """ 
        In order for the dataset to be compatible with torch.utils.data.DataLoader,
        we need to implement the following methods: 
        __init__, __getitem__, and __len__
    """
    def __init__(self, imgs_dir, img_size):
        """Set the path for images and targets

        Args:
            imgs_dir: specific images directory.
            targets: target labels.
            img_size: image size after resizing.
        """
        self.imgs_dir = imgs_dir
        self.normalize = transforms.Normalize(
            mean=DATASET_MEAN, # Our dataset mean
            std=DATASET_STD   # Our dataset std
        )
        self.resize = transforms.Compose([
            transforms.Resize(
                img_size, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(img_size)
        ])
    

    def __getitem__(self, index):
        """Returns one data pair (image and target).
        
        Args:
            index: the index of item.
        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        # Read data from files
        image = Image.open(f'{self.imgs_dir}/{index}.png').convert('RGB')
        image = self.resize(image)
        image = self.normalize(np.asarray(image))

        target = np.random.randint(0, 10)

        return image, target

    def __len__(self):
        """Returns the total number of samples."""
        return len(os.listdir(self.root))