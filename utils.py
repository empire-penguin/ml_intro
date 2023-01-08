############### TRITON NEUROTECH #################
## Helper functions for the project
##################################################
## Author: Gavin Roberts
## Date: Jan-07-2023
## Contributors: 
## Version: 1.0.0
## Email: gsroberts@ucsd.edu
## Status: Development
##################################################

import os
import json

def read_json(file_path):
    ''' 
    Reads a json file and returns a dictionary
    
    Args:
        file_path (str): The path to the json file
    Returns:
        dict: The dictionary representation of the json file
    '''
    if not os.path.isfile(file_path):
        raise FileNotFoundError(f'File {file_path} does not exist')

    with open(file_path, 'r') as f:
        data = json.load(f)
    return data

def read_json_in(root, file_path):
    ''' 
    Reads a json file in a directory and returns a dictionary
    
    Args:
        root (str): The root directory of the json file
        file_path (str): The name of the json file
    Returns:
        dict: The dictionary representation of the json file
    '''
    return read_json(os.path.join(root, file_path))

def write_json(data, file_path):
    ''' 
    Writes a dictionary to a json file
    
    Args:
        data (dict): The dictionary to write to the json file
        file_path (str): The path to the json file
    '''
    with open(file_path, 'w') as f:
        json.dump(data, f, indent=4)

def write_json_in(root, file_path, data):
    ''' 
    Writes a dictionary to a json file in a directory
    
    Args:
        root (str): The root directory of the json file
        file_path (str): The name of the json file
        data (dict): The dictionary to write to the json file
    '''
    write_json(data, os.path.join(root, file_path))

def get_config_data(exp_name):
    ''' 
    Gets the config data for an experiment
    
    Args:
        exp_name (str): The name of the experiment
    Returns:
        dict: The config data for the experiment
    '''
    # Get the root directory of the project
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    # Get the config file
    config_file = os.path.join(root, 'configs', f'{exp_name}.json')

    # Read the config file
    config_data = read_json(config_file)

    return config_data