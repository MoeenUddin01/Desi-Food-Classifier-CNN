from torch.nn import Module, Flatten, Sequential,Linear, ReLU
import torch

class CNN(Module):
    def __init__(self):
        super(CNN, self).__init__()
        # height * width * channels
        # Input image 128 * 128 * 3
        self.conv1d = torch.nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1) 
        self.pool1d = torch.nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        # Image Size 64 * 64 * 64
        self.conv2d = torch.nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.pool2d = torch.nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        # Image Size 32 * 32 * 128
        self.flatten_layer = Flatten(1)
        self.hidden_layers = Sequential(
            # 1 x 1,31,072
            Linear(32 * 32 * 128, 256),
            ReLU(),
            Linear(256, 128),
            ReLU(),
            Linear(128, 13)
        )


    def forward(self, x):
        x = self.conv1d(x)
        x = torch.relu(x)
        x = self.pool1d(x)
        x = self.conv2d(x)
        x = torch.relu(x)
        x = self.pool2d(x)
        x = self.flatten_layer(x)
        x = self.hidden_layers(x)
        return x


     
