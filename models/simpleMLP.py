"""
Simple MLP.
"""

# PyTorch imports
import torch.nn as nn

# 4 hidden layer MLP
class SimpleMLP(nn.Module):
    
    #Define the network layers here
    def __init__(self,num_classes=3):
        
        # Initialize the superclass
        super(SimpleMLP, self).__init__()
        
        # Activation functions
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)
        
        
        # Fully-connected layers
        self.fc1 = nn.Linear(24320, 9728)
        self.fc2 = nn.Linear(9728, 4864)
        self.fc3 = nn.Linear(4864, 1600)
        self.fc4 = nn.Linear(1600, 400)
        self.fc5 = nn.Linear(400, num_classes)
        
    # Forward pass
    
    def forward(self, x):
        
        # Fully-connected layers
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.relu(self.fc3(x))
        x = self.relu(self.fc4(x))
        x = self.fc5(x)
        
        return x


class SimpleMLPSEQ(nn.Module):
    
    #Define the network layers - using a Sequential Module
    def __init__(self,num_classes=3):
        
        # Initialize the superclass
        super(SimpleMLPSEQ, self).__init__()

        self._sequence = nn.Sequential(
            nn.Linear(24320, 9728),nn.ReLU(),
            nn.Linear(9728, 4864),nn.ReLU(),      
            nn.Linear(4864, 1600),nn.ReLU(),      
            nn.Linear(1600, 400),nn.ReLU(),       
            nn.Linear(400, num_classes))
        
        
    # Forward pass
    
    def forward(self, x):
        
        x=self._sequence(x)
        
        return x
