import os

import numpy as np
import cv2
from torchvision import transforms
from PIL import Image

import torch
from torch.utils.data import Dataset


from utils.image import gaussian_noice

class noisyXRayDataset(Dataset):
    def __init__(self, image_paths:list, crop_size:list=[1000,1000]):
        self.image_paths = image_paths
        self.crop = transforms.CenterCrop(crop_size)

    def __getitem__(self, index):
        img = cv2.imread(self.image_paths[index], cv2.IMREAD_GRAYSCALE)
        img = img / 255
        
        noisy_image = gaussian_noice(img,proportion=0.5)
        noisy_image = Image.fromarray(noisy_image)
        cropped_noisy_image = np.asanyarray(self.crop(noisy_image))
        cropped_noisy_image = torch.tensor(cropped_noisy_image.reshape(cropped_noisy_image.shape[0],cropped_noisy_image.shape[1],1))
        
        orig_image = Image.fromarray(img)
        cropped_orig_image = np.asanyarray(self.crop(orig_image))
        cropped_orig_image = torch.tensor(cropped_orig_image.reshape(cropped_orig_image.shape[0],cropped_orig_image.shape[1],1))
    
        return cropped_noisy_image, cropped_orig_image

    def __len__(self):
        return len(self.image_paths)

    def show_image(self, index):
        img = cv2.imread(self.image_paths[index], cv2.IMREAD_GRAYSCALE)
        cv2.imshow(f"{self.image_paths[index]}",img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
