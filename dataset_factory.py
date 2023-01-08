############### TRITON NEUROTECH #################
## Generates a dataset object from a config file
##################################################
## Author: Gavin Roberts 
## Date: 01-07-2023
## Contribuors:
## Version: 1.0.0
## Email: gsroberts@ucsd.edu
## Status: Development
##################################################

import os
from torch.utils.data import DataLoader
from example_dataset import ExampleDataset

def get_datasets(config_data):
    # Get the root directories for the train, test, and val datasets
    images_root_dir = config_data['dataset']['images_root_dir']
    root_train = os.path.join(images_root_dir, 'train')
    root_test = os.path.join(images_root_dir, 'test')
    root_val = os.path.join(images_root_dir, 'val')

    # Get the csv files for the train, test, and val datasets
    csv_file_train = config_data['dataset']['csv_file_train']
    csv_file_test = config_data['dataset']['csv_file_test']
    csv_file_val = config_data['dataset']['csv_file_val']

    # Get the batch size, img_size, and num_workers for the train, test, and val datasets
    batch_size = config_data['dataset']['batch_size']
    num_workers = config_data['dataset']['num_workers']
    img_size = config_data['dataset']['img_size']

    # Get the train, test, and val dataloaders
    train_dataloader = get_dataloader(root_train, csv_file_train, batch_size, True, num_workers)
    test_dataloader = get_dataloader(root_test, csv_file_test, batch_size, False, num_workers)
    val_dataloader = get_dataloader(root_val, csv_file_val, batch_size, False, num_workers)



def get_dataloader(img_dir, csv_file, batch_size, shuffle, num_workers):
    
    # Create the dataset object
    dataset = ExampleDataset(img_dir, csv_file)

    # Create the dataloader object
    dataloader = DataLoader(dataset=dataset,
                            batch_size=batch_size,
                            shuffle=shuffle,
                            num_workers=num_workers)

    return dataloader