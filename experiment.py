############### TRITON NEUROTECH #################
## Experiment class for running a ML model
##################################################
## Author: Gavin Roberts
## Date: Jan-07-2023
## Contributors: 
## Version: 1.0.0
## Email: gsroberts@ucsd.edu
## Status: Development
##################################################

from consts import *
from utils import *
from dataset_factory import get_datasets
from model_factory import get_model
import matplotlib.pyplot as plt
import numpy as np
import torch
from datetime import datetime
from tqdm import tqdm
from copy import deepcopy

class Experiment(object):
    def __init__(self, exp_name):
        self.exp_name = exp_name
        config_data = get_config_data(exp_name)
        self.config_data = config_data
        self.device = config_data['config']['device']
        self.model = get_model()
        self.train_dataloader, self.test_dataloader, self.val_dataloader = self.get_dataloaders()
        self.optimizer = self.get_optimizer()
        self.lr_scheduler = self.get_lr_scheduler()
        self.loss = self.get_loss()
        self.metrics = self.get_metrics()
        self.best_model = None
        self.best_loss = 1e10
        self.best_acc = 0.0

    def run(self):
        self.train()
        self.test()

    def train(self):
        pass

    def test(self):
        pass