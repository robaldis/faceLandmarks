
from torch.utils.data import Dataset
import xml.etree.ElementTree as ET
import os
import numpy as np
import cv2
import random
import torch

class FaceLandmarksDataset(Dataset):

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
            image, landmarks = self.transform(image, landmarks, crops)

        # Makes it easier for the nueral network to train from
        landmarks = landmarks - 0.5

        return image, landmarks



def split_dataset(dataset):
    len_valid_set = int(0.1*len(dataset))
    len_train_set = len(dataset) - len_valid_set

    train_dataset , valid_dataset,  = torch.utils.data.random_split(dataset , [len_train_set, len_valid_set])

    train_batches = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=4)
    test_batches = torch.utils.data.DataLoader(valid_dataset, batch_size=8, shuffle=True, num_workers=4)
    return train_batches, test_batches
