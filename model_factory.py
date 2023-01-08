############### TRITON NEUROTECH #################
## Generates a ML model object from a config file
##################################################
## Author: Gavin Roberts
## Date: Jan-07-2023
## Contributors: 
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
class ExampleModel(nn.Module):

    def __init__(self, config_data):
        '''
            This is the constructor for the ExampleModel class.
            Args:
                config_data (dict): The config data for the model
        '''
        # Call the parent class constructor (this is just a Python thing)
        super(ExampleModel, self).__init__()

        # Save the config data
        self.config_data = config_data
        self.num_classes = config_data['model']['num_classes']
        self.dropout_rate = config_data['model']['dropout_rate']

        # Define the layers (From the AlexNet paper)
        self.conv1 = nn.Conv2d(
            in_channels=1,
            out_channels=48,
            kernel_size=11,
            stride=4,
            padding=1
        )
        self.batchnorm1 = nn.BatchNorm2d(num_features=48)
        self.pool = nn.MaxPool2d(kernel_size=5, stride=2, padding=0)

