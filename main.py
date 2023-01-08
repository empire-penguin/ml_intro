############### TRITON NEUROTECH #################
## Main script for training a ML model
##################################################
## Author: Gavin Roberts
## Date: Jan-07-2023
## Contributors: 
## Version: 1.0.0
## Email: gsroberts@ucsd.edu
## Status: Development
##################################################

# Import the necessary packages
from experiment import Experiment
import sys

# Main Driver for your code. Either run `python main.py` which will run the experiment with default config
# or specify the configuration by running `python main.py task-1-default-config`
if __name__ == "__main__":
    exp_name = 'default'

    if len(sys.argv) > 1:
        exp_name = sys.argv[1]

    print("Running Experiment: ", exp_name)
    exp = Experiment(exp_name)
    exp.run()
