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

trainDataLoader = DataLoader(trainData, batch_size=100, shuffle=True)
testDataLoader = DataLoader(testData, batch_size=100, shuffle=True)


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()

        #first convolution with 1 input channel (grayscale)
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1) 
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        #second convolution with more input channels
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        #need to flatten outputs to work
        self.flatten = nn.Flatten()

        #linearize outputs 
        self.linear1 = nn.Linear(64 * 7 * 7, 128)
        self.relu3 = nn.ReLU()
        self.linear2 = nn.Linear(128, 10)  # Output layer with 10 classes (digits 0-9)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.pool1(x)

        x = self.conv2(x)
        x = self.relu2(x)
        x = self.pool2(x)

        x = self.flatten(x)
        x = self.linear1(x)
        x = self.relu3(x)
        x = self.linear2(x)
        return x

#declares the cnn
cnn = CNN()
#print(cnn)

#makes the loss function, function to be minimized for model to work
lossFunction = nn.CrossEntropyLoss()
#print(lossFunction)

#make the optimizer, lots of different optimization functions, using adam here
#lr determines the step size for the optimizer
optimizer = optim.Adam(cnn.parameters(), lr = 0.001)

num_epochs = 10
for epoch in range(num_epochs):
    for images, labels in trainDataLoader:
        # Forward pass
        outputs = cnn(images)
        loss = lossFunction(outputs, labels)

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

cnn.eval()
with torch.no_grad():
    correct = 0
    total = 0
    for images, labels in testDataLoader:
        outputs = cnn(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    print(f'Test Accuracy: {100 * correct / total}%')