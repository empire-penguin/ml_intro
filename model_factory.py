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
            out_channels=96,
            kernel_size=11,
            stride=4,
            padding=1
        )
        self.batchnorm1 = nn.BatchNorm2d(num_features=96)
        self.maxpool1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=0)
        self.conv2 = nn.Conv2d(
            in_channels=96,
            out_channels=256,
            kernel_size=5,
            stride=1,
            padding=2
        )
        self.batchnorm2 = nn.BatchNorm2d(num_features=256)
        self.maxpool2 = nn.MaxPool2d(kernel_size=3, stride=2, padding=0)
        self.conv3 = nn.Conv2d(
            in_channels=256,
            out_channels=384,
            kernel_size=3,
            stride=1,
            padding=1
        )
        self.batchnorm3 = nn.BatchNorm2d(num_features=384)
        self.conv4 = nn.Conv2d(
            in_channels=384,
            out_channels=384,
            kernel_size=3,
            stride=1,
            padding=1
        )
        self.batchnorm4 = nn.BatchNorm2d(num_features=384)
        self.conv5 = nn.Conv2d(
            in_channels=384,
            out_channels=256,
            kernel_size=3,
            stride=1,
            padding=1
        )
        self.batchnorm5 = nn.BatchNorm2d(num_features=256)
        self.maxpool3 = nn.MaxPool2d(kernel_size=3, stride=2, padding=0)
        self.fc1 = nn.Linear(in_features=9216, out_features=4096)
        self.fc2 = nn.Linear(in_features=4096, out_features=4096)
        self.fc3 = nn.Linear(in_features=4096, out_features=self.num_classes)
        self.dp = nn.Dropout(p=self.dropout_rate)
    
    def forward(self, x):
        '''
            This is the forward pass of the model.
            Args:
                x (torch.Tensor): The input tensor
            Returns:
                x (torch.Tensor): The output tensor
        '''
        # First block 227x227x3 -> 55x55x96
        x = self.conv1(x)
        x = self.batchnorm1(x)
        x = F.relu(x)
        x = self.maxpool1(x)

        # Second block 55x55x96 -> 27x27x256
        x = self.conv2(x)
        x = self.batchnorm2(x)
        x = F.relu(x)
        x = self.maxpool2(x)

        # Third block 27x27x256 -> 13x13x384
        x = self.conv3(x)
        x = self.batchnorm3(x)
        x = F.relu(x)

        # Fourth block 13x13x384 -> 13x13x384
        x = self.conv4(x)
        x = self.batchnorm4(x)
        x = F.relu(x)

        # Fifth block 13x13x384 -> 13x13x256
        x = self.conv5(x)
        x = self.batchnorm5(x)
        x = F.relu(x)
        x = self.maxpool3(x)
        
        # Fully connected layers
        # 13x13x256 -> 4096
        x = self.fc1(x.view(len(x), -1))
        x = self.dp(x)
        # 4096 -> 4096
        x = self.fc2(x)
        x = self.dp(x)
        # 4096 -> num_classes
        x = self.fc3(x)
        x = self.dp(x)

        return x
    

def get_model(config_data):
    '''
        This function returns the model object.
        Args:
            config_data (dict): The config data for the model
        Returns:
            model (torch.nn.Module): The model object
    '''
    # Create the model object
    model = ExampleModel(config_data)

    # Return the model object
    return model