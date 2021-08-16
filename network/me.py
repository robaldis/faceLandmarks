import torch.nn as nn
from torchvision import models

class Network(nn.Module):

    def __init__(self, num_classes=136):
        super().__init__()
        self.model_name = 'resnet18'
        # ResNet18 framework to build off of
        self.model = models.resnet18()
        # Change the first layer so we can input gray scale images
        self.model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        # change the final layer to ouput 136 values (68 * 2, all x, y points)
        self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)

    def forward(self, x):
        x = self.model(x)
        return x
