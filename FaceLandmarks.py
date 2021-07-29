#! env/bin/python3

import time
import cv2
import os
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
        # TODO: do the transforms to the image to make it the right format
        # Make a transform class to crop the image by its bounding box,
        # resize the image
        # rotate the image
        image = cv2.imread(self.image_filenames[index], 0)
        landmarks = self.landmarks[index]
        crops = self.crops[index]

        if (self.transform):
            image, landmarks = self.transform(image, landmarks, crops)

        # Makes it easier for the nueral network to train from
        landmarks = landmarks - 0.5

        return image, landmarks


class Network(nn.Module):

    def __init__(self, num_classes=136):
        super().__init__()
        self.model_name = 'resnet18'
        # ResNet18 framework to build off of
        self.model = models.resent18()
        # Change the first layer so we can input gray scale images
        self.model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bais=False)
        # change the final layer to ouput 136 values (68 * 2, all x, y points)
        self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)

    def forward(self, x):
        x = self.model(x)
        return x


# Helper functions
def check_first_index(dataset):
    image, landmarks = dataset[0]

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

if __name__ == "__main__":
    download_dataset()
    dataset = FaceLandmarkDataset(Transform())
    check_first_index(dataset)
