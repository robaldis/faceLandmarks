#! env/bin/python3

import cv2
import time
from PIL import Image
import numpy as np

import torch
import torch.nn as nn
from torchvision import models
import torchvision.transforms.functional as TF


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


def main():
    weights_path = 'content/face_landmarks.pth'
    frontal_face_cascade_path = 'haarcascade_frontalface_default.xml'

    face_cascade = cv2.CascadeClassifier(frontal_face_cascade_path)

    cap = cv2.VideoCapture(0)

    best_network = Network()
    best_network.load_state_dict(torch.load(weights_path, map_location=torch.device('cpu')))
    best_network.eval()

    # Check if the webcam has opened correctly
    if  not cap.isOpened():
        raise IOError("cannot read camera")

    moustache = cv2.imread("assets/moustache.png", cv2.IMREAD_UNCHANGED)
    moustache = cv2.resize(moustache, (90, 33))
    m_height, m_width, channels = moustache.shape
    print(channels)
    print(m_height, m_width)

    
    while True:
        ret, frame = cap.read();
        grayscale_image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        display_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        height, width,_ = frame.shape

        faces = face_cascade.detectMultiScale(grayscale_image, 1.1, 4)

        # cv2.imshow("webcam", frame)


        all_landmarks = []
        for (x, y, w, h) in faces:
            image = grayscale_image[y:y+h, x:x+w]
            image = TF.resize(Image.fromarray(image), size=(224, 224))
            image = TF.to_tensor(image)
            image = TF.normalize(image, [0.5], [0.5])

            with torch.no_grad():
                landmarks = best_network(image.unsqueeze(0)) 

            landmarks = (landmarks.view(68,2).detach().numpy() + 0.5) * np.array([[w, h]]) + np.array([[x, y]])
            all_landmarks.append(landmarks)

        for landmarks in all_landmarks:
            x = np.mean(landmarks[48:,0]) - (m_width/2)
            y = np.mean(landmarks[48:,1]) - 25

            # Draw the moustache to the screen
            # plt.imshow(moustache, extent=[m_width + x, x, m_height + y, y])
            # cv2.imshow("moustache", moustache)
            # cv2.rectangle(frame, (int(x),int(y)), (int(x) + 100, int(y) + 100), (0,255,0), 5)
            alpha_m = moustache[:,:,3] / 255.0
            alpha_f = 1.0- alpha_m
            x1 = int(x)
            x2 = int(x) + int(m_width)
            y1 = int(y)
            y2 = int(y) + int(m_height)

            for c in range(0,3):

                frame[y1: y2, x1:x2, c] = (alpha_m * moustache[:,:,c] + alpha_f * frame[y1:y2,x1:x2, c])


        cv2.imshow("webcam", frame) 

        c = cv2.waitKey(1)
        if c == 27: #Esc key
            break


    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":

    main()
