"""
Simple CNN based on Kazu Terao's and Abhishek Abhishek code .
"""

# PyTorch imports
import torch.nn as nn

# KazuNet class
class SimpleCNN(nn.Module):
    
    # Initializer
    
    def __init__(self, num_input_channels=38, num_classes=3, train=True):
        
        # Initialize the superclass
        super(SimpleCNN, self).__init__()
        
        # Activation functions
        self.relu = nn.ReLU()
                
        # Feature extractor

        self.f_embed = nn.Conv2d(num_input_channels, 32, kernel_size=1, stride=1, padding=0)
        # Convolutions and max-pooling
        self.f_conv1 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.f_max_pool1  = nn.MaxPool2d(2,2)
        
        self.f_conv2a = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.f_conv2b = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.f_max_pool2  = nn.MaxPool2d(2,2)
        
        self.f_conv3a = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.f_conv3b = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)
        self.f_max_pool3 = nn.MaxPool2d(2,2)
        
        self.f_conv4  = nn.Conv2d(128, 128, kernel_size=2, stride=1, padding=0)
        
        # Flattening / MLP
        
        # Fully-connected layers
        self.fc1 = nn.Linear(128, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, num_classes)
        
    # Forward pass
    
    def forward(self, x):
        
        # Convolutions and max-pooling
        x = self.f_max_pool1(self.relu(self.f_conv1(self.relu(self.f_embed(x)))))
        #print("after first max pool shape of the data: {}".format(x.shape))
        x = self.f_max_pool2(self.relu(self.f_conv2b(self.relu(self.f_conv2a(x)))))
        #print("after 2nd max pool shape of the data: {}".format(x.shape))
        x = self.f_max_pool3(self.relu(self.f_conv3b(self.relu(self.f_conv3a(x)))))
        #print("after 3rd max pool shape of the data: {}".format(x.shape))
        x = self.relu(self.f_conv4(x))
        #print("after last convolution {}".format(x.shape))

        
        # Flattening
        x = nn.MaxPool2d(x.size()[2:])(x)
        x = x.view(-1, 128)
        
        # Fully-connected layers
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        
        return x
