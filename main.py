
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
import matplotlib.pyplot as plt
import matplotlib.cm as cm 

#imports
import torch.optim as optim
import torch.nn.functional as F 
from torch.utils.data import DataLoader
import torchvision.datasets as datasets
import torchvision.transforms as transforms 
import sys

from time import time

# Hyper Parameters
batch_size = 100


# Path to download the data 
dataset_dir = os.path.join(os.getcwd(), "Datasets", "FashionMNIST")


# Loading data 


def loader_data(data_set):
    if data_set == "MNIST":
        train_dataset = torchvision.datasets.FashionMNIST(root = dataset_dir, train = True, transform = transforms.ToTensor(), download = True)
        train_loader  = torch.utils.data.DataLoader(train_dataset, batch_size = batch_size,  shuffle = True)

        test_dataset = torchvision.datasets.FashionMNIST(root = dataset_dir, train = False, transform = transforms.ToTensor(), download = True)
        test_loader  = torch.utils.data.DataLoader(test_dataset, batch_size = batch_size,  shuffle = True)
    elif data_set == "Fashion-MNIST":
        train_dataset = torchvision.datasets.FashionMNIST(root = dataset_dir, train = True, transform = transforms.ToTensor(), download = True)
        train_loader  = torch.utils.data.DataLoader(train_dataset, batch_size = batch_size,  shuffle = True)

        test_dataset = torchvision.datasets.FashionMNIST(root = dataset_dir, train = False, transform = transforms.ToTensor(), download = True)
        test_loader  = torch.utils.data.DataLoader(test_dataset, batch_size = batch_size,  shuffle = True)

    return train_dataset, train_loader, test_dataset, test_loader


train_dataset, train_loader, test_dataset, test_loader = loader_data('MNIST')

# Exploring the data 
nsamples = 10
classes_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal','Shirt', 'Sneaker', 'Bag', 'Ankle boot'] 

imgs, labels = next(iter(train_loader))

# fig=plt.figure(figsize=(20,5),facecolor='w')
# for i in range(nsamples):
#     ax = plt.subplot(1,nsamples, i+1)
#     plt.imshow(imgs[i, 0, :, :], vmin=0, vmax=1.0, cmap=cm.gray)
#     ax.set_title("{}".format(classes_names[labels[i]]), fontsize=15)
#     ax.get_xaxis().set_visible(False)
#     ax.get_yaxis().set_visible(False)

# plt.savefig('fashionMNIST_samples.png', bbox_inches='tight')    
# plt.show()


# Implementing Linear Network just to Check the performance 


class LinearNet(nn.Module):
    def __init__(self, input_size,  num_classes):
        super(LinearNet, self).__init__()
        self.input_size = input_size
        self.num_classes = num_classes
        self.classifier = nn.Linear(self.input_size, self.num_classes)

    def forward(self, x):
        # Resizing , keeping batch size separate and making everythng else a one D vector 

        x = x.view(x.size()[0] , -1)
        y = self.classifier(x)
        return y


class FCNN(nn.Module):
    def __init__(self, input_size, num_classes ):
        super(FCNN,self).__init__()
        self.num_classes = num_classes
        self.input_size = input_size
        self.fc1 = nn.Linear(self.input_size, 50)
        self.fc2 = nn.Linear(50, self.num_classes)

    def forward(self, x):
        x = x.view(x.size()[0] , -1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        return x



class FashionCNN(nn.Module):
    
    def __init__(self):
        super(FashionCNN, self).__init__()
        
        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        self.layer2 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        
        self.fc1 = nn.Linear(in_features=64*6*6, out_features=600)
        self.drop = nn.Dropout2d(0.25)
        self.fc2 = nn.Linear(in_features=600, out_features=120)
        self.fc3 = nn.Linear(in_features=120, out_features=10)
        
    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.view(out.size(0), -1)
        out = self.fc1(out)
        out = self.drop(out)
        out = self.fc2(out)
        out = self.fc3(out)
        
        return out




# set Hyper parameters
input_size    = 1*28*28
num_classes   = 10
batch_size    = 100
learning_rate = 0.01
num_epochs    = 10

# check if cuda is available
device = 'cuda'if torch.cuda.is_available() else 'cpu'

# Instantiate the Model
# model = FCNN(input_size, num_classes).to(device)
# model = FashionCNN().to(device)
model = FashionCNN().to(device)


print(model)

# Loss Function and Optimizers
criterion  = nn.CrossEntropyLoss()
optimizer  = optim.Adam(model.parameters(), lr = learning_rate)



def train(epoch):
    # Initializing or setting to the training method
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        # Converting the tensor to be calculated on GPU if available
        data = data.view(100, 1, 28, 28)
        data, target = data.to(device), target.to(device)
        # Make sure optimizer has no gradients,  leaning the exisiting gradients.
        optimizer.zero_grad()
        # Giving model the data it calls the forward function and returns the output Forward pass - compute outputs on input data using the model
        output = model(data)
        # Calculate the loss
        loss = criterion(output, target)
        # back propagating the loss calclulating gradients and adjustmrnts
        loss.backward()
        # Updating Optimizing Parameters
        optimizer.step()
        # Logging the training progress to console.
        print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()), end = '\r')



def test():
    # Setting it up to the Eval ot test mode
    model.eval()
    # While testing we woould not need to calculte gradients hence it will same some computation.
    test_loss = 0
    correct   = 0
    with torch.no_grad():
        for data, target in test_loader:
            print("Evaluating...", end = '\r' )
            data = data.view(100, 1, 28, 28)
            data, target = data.to(device), target.to(device)
            # reshaping the data as it has Color channel things        # 
            # Giving model the data it calls the forward function and returns the output Forward pass - compute outputs on input data using the model
            output = model(data)
            # Calculate the loss and appending the loss to make it a total figure
            test_loss += criterion(output, target).item()
            # Getting the highest probability Output.
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset) 
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))



def main():
    start = time()
    for i in range(num_epochs):
        train(i+1)
    print('\n Time Taken {:.2f} secs'. format(time()-start))
    start = time()
    print('\n Time Taken {:.2f} secs'. format(time()-start))
    test()

if __name__ == '__main__':
    main()
    torch.save(model.state_dict(), "mnist_cnn.pt")






