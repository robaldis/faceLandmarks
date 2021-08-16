#! env/bin/python3

import torch.nn as nn
import torch.optim as optim

from training.his import train
from network.his import Network
from dataobject.his import FaceLandmarksDataset, split_dataset
from transform.his import Transforms


dataset = FaceLandmarksDataset(Transforms())

network = Network()
network.cuda()
criterian = nn.MSELoss()
optimizer = optim.Adam(network.parameters(), lr=0.0001)
train_loader, valid_loader = split_dataset(dataset)

train(network, criterian, optimizer, train_loader, valid_loader, epoch=20)


