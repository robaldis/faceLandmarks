#! env/bin/python3

import time
import cv2
import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import imutils

import torch
import torch.nn as nn
from torchvision import models
import torchvision.transforms.functional as TF

image_path = 'ibug_300W_large_face_landmark_dataset/lfpw/trainset/image_0457.png'
image_path = '/home/robert/ownCloud/face.jpg'
weights_path = 'content/face_landmarks.pth'
frontal_face_cascade_path = 'haarcascade_frontalface_default.xml'

class Network(nn.Module):
    def __init__(self,num_classes=136):
        super().__init__()
        self.model_name='resnet18'
        self.model=models.resnet18(pretrained=False)
        self.model.conv1=nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.model.fc=nn.Linear(self.model.fc.in_features,num_classes)
        
    def forward(self, x):
        y=self.model(x)
        return y


def main(Network):
    face_cascade = cv2.CascadeClassifier(frontal_face_cascade_path)

    best_network = Network()
    best_network.load_state_dict(torch.load(weights_path, map_location=torch.device('cuda')))
    best_network.eval()

    image = cv2.imread(image_path)
    image = cv2.resize(image, [350, 464])
    grayscale_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    display_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    height, width,_ = image.shape

    print(height, width)

    # 464 350
    faces = face_cascade.detectMultiScale(grayscale_image, 1.1, 4)

    moustache = cv2.imread("assets/moustache.png")
    m_height, m_width, _ = moustache.shape
    m_height = m_height /10
    m_width = m_width /10

    all_landmarks = []
    for (x, y, w, h) in faces:
        image = grayscale_image[y:y+h, x:x+w]
        image = TF.resize(Image.fromarray(image), size=(224, 224))
        image = TF.to_tensor(image)
        image = TF.normalize(image, [0.5], [0.5])
        image.cuda()


        with torch.no_grad():
            landmarks = best_network(image.unsqueeze(0)) 

        landmarks = (landmarks.view(68,2).detach().numpy() + 0.5) * np.array([[w, h]]) + np.array([[x, y]])
        all_landmarks.append(landmarks)

    plt.figure()
    plt.imshow(display_image)


    for landmarks in all_landmarks:
        x = np.mean(landmarks[48:,0]) - (m_width/2)
        y = np.mean(landmarks[48:,1]) - 25

        plt.imshow(moustache, extent=[m_width + x, x, m_height + y, y])
        for i, landmark in enumerate(landmarks):
            if (i > 48):
                print(landmark)
                plt.scatter(landmark[0], landmark[1], c = 'c', s = 5)
                plt.annotate(i, (landmark[0], landmark[1]))

    plt.show()

if __name__ == "__main__":
    main(Network)
