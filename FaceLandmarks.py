#! env/bin/python3

import time
import math
import cv2
import os
import sys
import random
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import imutils
import matplotlib.image as mpimg
from collections import OrderedDict
from skimage import io, transform
from math import radians, cos, sin
import xml.etree.ElementTree as ET


import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from torchvision import datasets, models, transforms
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

from transform import Transform


loss_array = []

# TODO: Refactor all of the code to SoC each major element so they can be tested
# interchangeably to make sure the transforms are the thing thats making it not 
# train properly
# TODO: Clean up all the non-needed code


class FaceLandmarkDataset(Dataset):

    def __init__(self, Transform=None):
        tree = ET.parse('ibug_300W_large_face_landmark_dataset/labels_ibug_300W_train.xml')
        root = tree.getroot()

        self.image_filenames = []
        self.landmarks = []
        self.crops = []
        self.images = []
        self.root_dir = 'ibug_300W_large_face_landmark_dataset'
        self.transform = Transform

        for filename in root[2]:
            self.image_filenames.append(os.path.join(self.root_dir, filename.attrib["file"]))

            self.crops.append(filename[0].attrib)

            landmark = []
            for num in range(68):
                x_coord = filename[0][num].attrib['x']
                y_coord = filename[0][num].attrib['y']
                landmark.append([x_coord, y_coord])
            self.landmarks.append(landmark)

        self.landmarks = np.array(self.landmarks).astype('float32')
        # Make sure we have matching amount of landmarks to images
        assert len(self.image_filenames) == len(self.landmarks)

    def __len__(self):
        return len(self.image_filenames)

    def __getitem__(self, index):
        # This line is cuasing the error we see when we try to read all
        image = cv2.imread(self.image_filenames[index], 0)
        landmarks = self.landmarks[index]
        crops = self.crops[index]

        if (self.transform):
            image, landmarks = self.transform(image, landmarks, crops, self.image_filenames[index])

        # Makes it easier for the nueral network to train from
        landmarks = landmarks - 0.5

        return image, landmarks


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


# Helper functions
def check(dataset, index):
    count = 0
    # batches = torch.utils.data.DataLoader(dataset, batch_size=64, num_workers=4)
    for image, landmarks in dataset:
        count += 1
        if image == None:
            raise ReferenceError()
        if landmarks == None:
            raise ReferenceError()

    print(f"checked {count} items")

    image, landmarks = dataset[index]

    landmarks = (landmarks + 0.5) * 224

    plt.figure(figsize=(10, 10))
    plt.imshow(image.numpy().squeeze(), cmap='gray')
    plt.scatter(landmarks[:, 0], landmarks[:, 1], s=8)
    plt.show()

def download_dataset():
    if not os.path.exists('ibug_300W_large_face_landmark_dataset'):
        os.system("wget http://dlib.net/files/data/ibug_300W_large_face_landmark_dataset.tar.gz")
        os.system("tar -xvzf \'ibug_300W_large_face_landmark_dataset.tar.gz\'")
        os.system("rm -r \'ibug_300W_large_face_landmark_dataset.tar.gz\'")

def show_prediction(data, network, dataset):
    valid_loader = torch.utils.data.DataLoader(dataset, batch_size=8, shuffle=True, num_workers=4)
    with torch.no_grad(): 
        network.eval()

        images, landmarks = next(iter(valid_loader))
        images, landmarks = data

        images = images.cuda()
        landmarks = (landmarks + 0.5) * 224

        pred = (network(images).cpu() + 0.5) * 224
        z = pred.view(-1,68,2)


        plt.figure(figsize=(10, 40))
        plt.imshow(images[0].cpu().numpy().transpose(1,2,0).squeeze(), cmap='gray')
        plt.scatter(z[0,:, 0], z[0,:, 1], c=[[1,0,0]],  s=8)
        plt.scatter(landmarks[0,:, 0], landmarks[0,:, 1], c=[[0,1,0]], s=8)
        plt.figure(2)
        plt.scatter(loss_array, [x for x in range(1,len(loss_array)+1)], c=[[0,0,1]])
        plt.show()


def print_overwrite(step, total_step, another_loss, loss, operation):
    sys.stdout.write('\r')
    if operation == 'train':
        sys.stdout.write('Train Steps: %d/%d Running Loss: %.4f Loss: %.4f' % (step, total_step, loss, another_loss))
    else:
        sys.stdout.write('Eval Steps: %d/%d Loss: %.4f ' % (step, total_step, loss))


def split_dataset(dataset):
    train = []
    test = []

    for item in dataset:
        if (random.random() < 0.8):
            train.append(item)
        else:
            test.append(item)

    print(len(train))
    print(len(test))
    train_batches = torch.utils.data.DataLoader(train, batch_size=64, num_workers=4)
    test_batches = torch.utils.data.DataLoader(test, batch_size=64, num_workers=4)
    return train_batches, test_batches


def save_loss(loss):
    loss_array.append(loss)


def train(dataset): 

    # Things needed to train a machine learning model:
    # - Run the model a sample
    # - Calc cost/loss function
    # - "Backpropigation" / some way of chanigne the wieghts

    network = Network()
    network.cuda()
    critiern = nn.MSELoss()
    optimizer = optim.Adam(network.parameters(), lr=0.0001)
    train_batches, test_batches = split_dataset(dataset)


    loss_min = np.inf
    num_epochs = 10

    for epoch in range(num_epochs):
        # Need to put the network in training mode

        loss_train = 0
        loss_eval = 0
        running_loss = 0



        network.train()
        for step in range(len(train_batches)): 
            images, landmarks = next(iter(train_batches))

            x = images.cuda()
            # I think this flatterns out the landmarks
            y = landmarks.view(landmarks.size(0), -1).cuda()
            # x = image
            # y = landmarks


            # This resets teh grad values the perameters 
            optimizer.zero_grad()

            pred = network(x)
            # This calculates loss based on MeanSquaredError(MSE)
            cost = critiern(pred, y)
            # Performs back propigation to train the network
            cost.backward()
            # updates the weights in the model
            optimizer.step()


            loss_train += cost.item()
            # Average loss over this epoch
            running_loss = loss_train/(step+1)
            save_loss(running_loss)

            print_overwrite(step, len(train_batches), loss_train, running_loss, 'train')

        network.eval()
        with torch.no_grad():
            # Run over the validation dataset, this means split up the dataset
            for step in range(len(test_batches)): 
                images, landmarks = next(iter(test_batches))

                x = images.cuda()
                # I think this flatterns out the landmarks
                y = landmarks.view(landmarks.size(0), -1).cuda()
                # x = image
                # y = landmarks


                pred = network(x)
                # What does this do?
                cost_eval = critiern(pred, y)
                # print(f"\nCost: {cost_eval}")

                loss_eval += cost_eval.item()
                # Average loss over this epoch
                running_loss = loss_eval/(step+1)

                print_overwrite(step, len(test_batches), loss_eval, running_loss, 'eval')
        loss_train /= len(test_batches)
        loss_eval /= len(test_batches)


        print('\n' + '-' *20)
        print('Epoch: {} Train Loss: {:.4f} Eval Loss: {:.4f}'.format(epoch, loss_train, loss_eval))
        print('\n' + '-' *20)

        if (loss_eval < loss_min):
                # Update the min loss
                loss_min = loss_eval
                # Save the net
                torch.save(network.state_dict(), 'content/face_landmarks.pth')
                print('\n Saved a new model\n')

    print("Training Complete")
    show_prediction(next(iter(train_batches)), network, dataset)





if __name__ == "__main__":
    download_dataset()
    dataset = FaceLandmarkDataset(Transform())
    # check(dataset, 0)

    train(dataset)


    # batches = torch.utils.data.DataLoader(dataset, batch_size=64, num_workers=4)
    # network = Network()
    # network.cuda()
    # network.load_state_dict(torch.load('content/face_landmarks.pth'))
    # show_prediction(batches, network, dataset)


