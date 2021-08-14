import torch
import torchvision.transforms.functional as TF
import numpy as np
import random
from math import sin, cos, radians
from PIL import Image
import imutils
from skimage import transform


# Transform: perform transforms on the dataset image
# The use for all these transforms is to make the training of the CNN to be
# better. This means a slight rotation, change in brightness, and crop to focus
# on the face.
class Transform():
    def __init__(self):
        pass

    def rotate(self, image, landmarks, angle):
        angle = random.uniform(-angle, +angle)

        transformation_matrix = torch.tensor([
            [+cos(radians(angle)), -sin(radians(angle))], 
            [+sin(radians(angle)), +cos(radians(angle))]
        ])

        image = imutils.rotate(np.array(image), angle)

        landmarks = landmarks - 0.5
        new_landmarks = np.matmul(landmarks, transformation_matrix)
        new_landmarks = new_landmarks + 0.5
        return Image.fromarray(image), new_landmarks

    def resize(self, image, landmarks, img_size):
        image = TF.resize(image, img_size)
        return image, landmarks

    def color_jitter(self, image, landmarks):
        color_jitter = transforms.ColorJitter(brightness=0.3, 
                                              contrast=0.3,
                                              saturation=0.3, 
                                              hue=0.1)
        image = color_jitter(image)
        return image, landmarks

    def crop_face(self, image, landmarks, crops, image_path):
        # TODO: Find out why this line is causing the error
        image = TF.to_tensor(image)
        left = int(crops["left"])
        top = int(crops["top"])
        width = int(crops["width"])
        height = int(crops["height"])

        image = TF.crop(image, top, left, height, width)

        img_shape = np.array(image).shape
        landmarks = torch.tensor(landmarks) - torch.tensor([[left, top]])
        landmarks = landmarks / torch.tensor([img_shape[1], img_shape[2]])

        return image, landmarks

    def __call__(self, image, landmarks, crops, image_path):
        image, landmarks = self.crop_face(image, landmarks, crops, image_path)
        image, landmarks = self.resize(image, landmarks, (224, 224))

        return image, landmarks
