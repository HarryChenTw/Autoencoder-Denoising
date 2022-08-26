import argparse
import numpy as np
import cv2

import torch

from model.autoencoder import AutoEncoder
from utils.image import gaussian_noise

def test(opt,device):
    # read image and normalize
    test_image = cv2.imread(opt.input_image, cv2.IMREAD_GRAYSCALE)
    test_image = test_image / 255

    # get noisy image
    noisy_image = gaussian_noise(test_image, 0.3)
    cv2.imwrite(f'{opt.output_dir}/noisy_image.png', np.clip(noisy_image * 255,0,255))

    # restore model
    model = AutoEncoder().to(device)
    ckpt = torch.load(opt.model, map_location=device)
    model.load_state_dict(ckpt['model'])
    
    noisy_image = torch.tensor(noisy_image.reshape(1,*noisy_image.shape,1)).to(device,dtype=torch.float)
    output = model(noisy_image).cpu().detach().numpy().reshape(*test_image.shape) * 255
    cv2.imwrite(f'{opt.output_dir}/denoised_image.png', np.clip(output,0,255))



if __name__ == '__main__':
    # parse script options
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_image', type=str, required=True, help='input image to be denoise')
    parser.add_argument('--model', type=str, required=True, help='path of model weights (.pt)')
    parser.add_argument('--output_dir', type=str, default=".", help='output dir for 2 output image (noisy_image, denoised_image)')
    opt = parser.parse_args()

    # device (only use cpu when testing)
    print(f'using cpu\n')
    device = 'cpu'
    
    test(opt,device)