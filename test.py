import torch
import torchvision
from torchvision import datasets
import torchvision.transforms as transforms
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

print(trainData)