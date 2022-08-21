import argparse
import os
import numpy as np
from tqdm import tqdm
from datetime import datetime, timezone, timedelta
import cv2


import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
from torchsummary import summary

from utils.dataset import mnistDataset
from model.autoencoder import autoencoder
from utils.image import gaussian_noice

def train(opt, device):

    image_size = [28,28]
    channels = 1

    # log
    tz = timezone(timedelta(hours=+8))
    log_name = datetime.now(tz).strftime('%Y-%m-%dT%H-%M-%S')
    os.mkdir(f'log/{log_name}')
    log = open(f'log/{log_name}/{log_name}.txt','a+')
    
    # load data and add noise
    from keras.datasets import mnist
    (train_normal_images, _), (val_normal_images, _) = mnist.load_data()
    log.write(f'training size : {len(train_normal_images)}\n')
    log.write(f'validation size : {len(val_normal_images)}\n')
    train_normal_images = train_normal_images / 255
    val_normal_images = val_normal_images / 255
    train_noisy_images = [gaussian_noice(normal_image,0.3) for normal_image in train_normal_images]
    val_noisy_images = [gaussian_noice(normal_image,0.3) for normal_image in val_normal_images]
    
    train_dataset = mnistDataset(train_normal_images, train_noisy_images)
    train_loader = DataLoader(
                        dataset = train_dataset,
                        batch_size = opt.batch_size,
                        shuffle = True)
    val_dataset = mnistDataset(val_normal_images, val_noisy_images)
    val_loader = DataLoader(
                    dataset = val_dataset,
                    batch_size = opt.batch_size,
                    shuffle = False)

    
    # define model and loss
    model = autoencoder().to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=opt.lr)

    # print model summary
    input_shape = (image_size[0],image_size[1],channels)
    summary(model, input_shape)

    # store the normal image and noisy image
    if opt.s : 
        intermediate_image_dir = f'log/{log_name}/intermediate_image'
        if not os.path.exists(intermediate_image_dir) : os.mkdir(intermediate_image_dir)
        normal_first_image = np.clip(val_dataset.get_first_image(noise=True,ret_tensor=False)*255, 0, 255)
        noisy_first_image = np.clip(val_dataset.get_first_image(noise=False,ret_tensor=False)*255, 0, 255)
        cv2.imwrite(f'{intermediate_image_dir}/noisy_image.png', normal_first_image)
        cv2.imwrite(f'{intermediate_image_dir}/normal_image.png', noisy_first_image)

    # training loop 
    for epoch in range(1, opt.epochs+1):
        optimizer.zero_grad()

        # training iteration
        train_loss = 0
        pbar = tqdm(enumerate(train_loader), total = len(train_loader))
        for i, (noisy_image, orig_image) in pbar:

            noisy_image = noisy_image.to(device,dtype=torch.float)
            orig_image = orig_image.to(device,dtype=torch.float)
            # forward pass
            outputs = model(noisy_image)
            # calculate loss
            loss = criterion(outputs, orig_image)
            train_loss += loss.item()
            # Backward and optimize
            loss.backward()
            optimizer.step()
            
            # update train description
            train_desc = f'Epoch [{epoch}/{opt.epochs}], Loss: {train_loss/(i+1):.4f}'
            pbar.set_description(desc = train_desc)

            # stores intermediate image (every 10 iters)
            if (opt.s) & ((epoch * len(pbar) + i) % 10 == 0):
                first_image = val_dataset.get_first_image(noise=True,ret_tensor=True)
                first_image = torch.reshape(first_image, (1,*first_image.shape)).to(device,dtype=torch.float)
                outputs = model(first_image)[0].cpu().detach().numpy().reshape(*image_size) * 255
                outputs = np.clip(outputs.round(), 0, 255)
                intermediate_image_dir = f'log/{log_name}/intermediate_image'
                if not os.path.exists(intermediate_image_dir) : os.mkdir(intermediate_image_dir)
                cv2.imwrite(f'{intermediate_image_dir}/epoch_{epoch}_iter_{i}.png', outputs)
        log.write(f'{train_desc}')

        # validation
        val_loss = 0
        with torch.no_grad():
            for i, (noisy_image, orig_image) in enumerate(val_loader):
                noisy_image = noisy_image.to(device,dtype=torch.float)
                orig_image = orig_image.to(device,dtype=torch.float)
                # forward pass
                outputs = model(noisy_image)
                # calculate loss
                loss = criterion(outputs, orig_image)
                val_loss += loss.item()

        avg_val_loss = val_loss/(i+1)

        # validation description
        val_desc = f', Val Loss: {avg_val_loss:4f}\n'
        print(val_desc)
        log.write(val_desc)
    
    # save model
    ckpt = {'model': model.state_dict()}
    torch.save(ckpt, f'log/{log_name}/{log_name}_weights.pt')

    log.close()


if __name__ == '__main__':
    # parse script options
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_data', type=str, default='test_image/train', help='image data folder for training')
    parser.add_argument('--val_data', type=str, default='test_image/val', help='image data folder for validation')
    parser.add_argument('--epochs', type=int, default=100, help='num of epochs')
    parser.add_argument('--batch_size', type=int, default=8, help='total batch size')
    parser.add_argument('--lr', type=float, default=1e-4, help='learning rate')
    parser.add_argument('--s', action="store_true", default=False, help='store result image of the 1st val image in training')
    opt = parser.parse_args()

    # device
    device = 'cpu'
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print(f'using {device} : {torch.cuda.get_device_name()}\n')
    else:
        print(f'using {device}\n')
    
    train(opt,device)