import torch
import torchvision.transforms.functional as TF
import numpy as np


# Transform: perform transforms on the dataset image
# The use for all these transforms is to make the training of the CNN to be
# better. This means a slight rotation, change in brightness, and crop to focus
# on the face.
class Transform():
    def __init__(self):
        pass

    def resize(self, image, landmarks, img_size):
        image = TF.resize(image, img_size)
        return image, landmarks

    def crop_face(self, image, landmarks, crops):
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

    def __call__(self, image, landmarks, crops):
        image, landmarks = self.crop_face(image, landmarks, crops)
        image, landmarks = self.resize(image, landmarks, (224, 224))

        return image, landmarks
