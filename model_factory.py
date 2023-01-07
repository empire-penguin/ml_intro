############### TRITON NEUROTECH #################
## Generates a ML model object from a config file
##################################################
## Author: Gavin Roberts
## Version: 1.0.0
## Email: gsroberts@ucsd.edu
## Status: Development
##################################################

# Import the necessary packages
import torch
import torch.nn as nn
import torch.nn.functional as F

'''
This is an example of how to define a ML model in pytorch.
'''
class example_model(nn.Module):

    def __init__(self, config_data):
        # Call the parent class constructor (this is just a Python thing)
        super(example_model, self).__init__()

        # Save the config data
        self.config_data = config_data
        self.hidden_size = config_data['hidden_size']
        self.num_layers = config_data['num_layers']
        self.num_classes = config_data['num_classes']

