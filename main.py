
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


# Hyper Parameters
batch_size = 128


# Path to download the data 
dataset_dir = os.path.join(os.getcwd(), "Datasets", "FashionMNIST")


# Loading data 
train_dataset = torchvision.datasets.MNIST(root = dataset_dir, train = True, transform = transforms.ToTensor(), download = True)
train_loader  = torch.utils.data.DataLoader(train_dataset, batch_size = batch_size,  shuffle = True)

test_dataset = torchvision.datasets.MNIST(root = dataset_dir, train = False, transform = transforms.ToTensor(), download = True)
test_loader  = torch.utils.data.DataLoader(test_dataset, batch_size = batch_size,  shuffle = True)



