
# Outlining the project 

 # - Importing the modules 
 # - Downloading the data 
 # - Preprocess
 # - Train Test
 # - Build Architecture
 # - Train
 # - Test
 # - Check Acccuracy and get done with tuning and selection of the model
 # - save the model encodings
 # - Deploy



 # Importing modules 
import torch 
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms

import os


# Path to download the data 
dataset_dir = os.path.join(os.getcwd(), "Datasets", "FashionMNIST")


# Loading data 


