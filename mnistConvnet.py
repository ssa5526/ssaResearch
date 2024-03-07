import torch
import torchvision
from torchvision import datasets
from torch import optim
from torch.utils.data import DataLoader
from torch.autograd import Variable

import torchvision.transforms as transforms
import torch.nn as nn
import matplotlib.pyplot as plt



trainData = datasets.MNIST(
    root = 'data', #path to where data is 
    train = True, #whether dataset is training or test
    download = True, #downloads the dataset if not in root    
    transform = transforms.ToTensor() #changes the data into a tensor(matrix)
)

testData = datasets.MNIST(
    root = 'data',
    train = False,
    download = True,
    transform = transforms.ToTensor()
)

#print(trainData)
#print(testData)

# we typically want to pass samples in “minibatches”, reshuffle the data at 
# every epoch to reduce model overfitting, and use Python’s multiprocessing to speed up data retrieval.
# DataLoader is the API to do so

trainDataLoader = DataLoader(trainData, batch_size=64, shuffle=True)
testDataLoader = DataLoader(testData, batch_size=64, shuffle=True)

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()

        #define the convolution operations using nn
        self.conv1 = nn.Conv2d(
            in_channels=1, #grayscale image so only 1 channel
            out_channels=16, #can mess around with this number, may allow more features to be seen
            kernel_size=5, #can mess around with this
            padding=2, #how much zero padding there is, since image is focusing on center, we want 0 padding
        )

        self.conv2 = nn.Conv2d(
            in_channels=16,
            out_channels=32, 
            kernel_size=5,
            padding=2 
        )

        self.relu1 = nn.ReLU()
        self.relu2 = nn.ReLU()
        self.maxPool1 = nn.MaxPool2d(2)
        self.maxPool2 = nn.MaxPool2d(2)

        self.output = nn.Linear(
            in_features = 32*7*7, #since we maxpool twice, the dimensions go down by a factor of 4 to 7x7
            out_features = 10 # 
        )

    def forward(self, x):
        #apply layers in order
        x = self.conv1(x)
        x = self.reul1(x)
        x = self.maxPoo1(x)
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.maxPool2(x)

        return self.out(x)

#declares the cnn
cnn = CNN()
#print(cnn)

#makes the loss function, function to be minimized for model to work
lossFunction = nn.CrossEntropyLoss()
#print(lossFunction)

#make the optimizer, lots of different optimization functions, using adam here
#lr determines the step size for the optimizer
optimizer = optim.Adam(cnn.parameters(), lr = 0.01)