############### TRITON NEUROTECH #################
## Constants for the project
##################################################
## Author: Gavin Roberts
## Date: Jan-07-2023
## Contributors: 
## Version: 1.0.0
## Email: gsroberts@ucsd.edu
## Status: Development
##################################################
import torch

OPTIMIZER = {
    'sgd': torch.optim.SGD,
    'adam': torch.optim.Adam,
    'adagrad': torch.optim.Adagrad,
}

LR_SCHEDULER = {
    'steplr': torch.optim.lr_scheduler.StepLR,
    'multi_step': torch.optim.lr_scheduler.MultiStepLR,
    'exponential': torch.optim.lr_scheduler.ExponentialLR,
    'cosine': torch.optim.lr_scheduler.CosineAnnealingLR,
    'reduce_on_plateau': torch.optim.lr_scheduler.ReduceLROnPlateau,
}

DATASET_MEAN = [0.485, 0.456, 0.406] # RGB
DATASET_STD = [0.229, 0.224, 0.225] # RGB