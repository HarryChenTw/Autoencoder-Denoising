import torch
from torch.utils.data import Dataset

class mnistDataset(Dataset):
    def __init__(self, normal_images, noisy_images):
        self.normal_images = normal_images
        self.noisy_images = noisy_images

    def __getitem__(self, index):
        noisy_image = self.noisy_images[index]
        normal_image = self.normal_images[index]
        return torch.tensor(noisy_image.reshape(*noisy_image.shape,1)), torch.tensor(normal_image.reshape(*normal_image.shape,1))

    def __len__(self):
        return len(self.normal_images)

    def get_first_image(self,noise=True,ret_tensor=True):
        first_image = self.normal_images[0]
        if noise:
            first_image = self.noisy_images[0]
        if ret_tensor:
            first_image = torch.tensor(first_image.reshape(*first_image.shape,1))
        return first_image