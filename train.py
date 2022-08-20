import argparse
import os
import numpy as np
from tqdm import tqdm
from datetime import datetime, timezone, timedelta

import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim

from utils.dataset import noisyXRayDataset
from model.autoencoder import autoencoder

def train(opt, device):

    image_size = [1000,1000]

    # log
    tz = timezone(timedelta(hours=+8))
    log_file_name = datetime.now(tz).strftime('%Y-%m-%dT%H:%M:%S')
    log = open(f'log/{log_file_name}.txt','w+')

    # get all image path
    train_image_paths, val_image_paths = list(), list()
    for image_file in os.listdir(opt.train_data):
        if image_file == '.DS_Store':
            continue
        train_image_paths.append(f"{opt.train_data}/{image_file}")
    for image_file in os.listdir(opt.val_data):
        if image_file == '.DS_Store':
            continue
        val_image_paths.append(f"{opt.val_data}/{image_file}")
    log.write(f'training size : {len(train_image_paths)}\n')
    log.write(f'validation size : {len(val_image_paths)}\n')

    # define dataset and dataloader
    train_dataset = noisyXRayDataset(train_image_paths, image_size)
    train_loader = DataLoader(
                        dataset = train_dataset,
                        batch_size = opt.batch_size,
                        shuffle = True)
    val_dataset = noisyXRayDataset(val_image_paths, image_size)
    val_loader = DataLoader(
                    dataset = val_dataset,
                    batch_size = opt.batch_size,
                    shuffle = False)
    
    # define model and loss
    model = autoencoder(input_size = image_size).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=opt.lr)

    # training loop 
    for epoch in range(1, opt.epochs+1):
        optimizer.zero_grad()

        # training iteration
        train_loss = 0
        pbar = tqdm(enumerate(train_loader), total = len(train_loader))
        for i, (noisy_image, orig_image) in pbar:

            noisy_image, orig_image = noisy_image.to(device), orig_image.to(device)

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
            log.write(f'{train_desc}')

        # validation
        val_loss = 0
        with torch.no_grad():
            for i, (noisy_image, orig_image) in enumerate(val_loader):
                noisy_image, orig_image = noisy_image.to(device), orig_image.to(device)

                # forward pass
                outputs = model(noisy_image)
                
                # calculate loss
                loss = criterion(outputs, orig_image)
                val_loss += loss.item()

        avg_val_loss = val_loss/(i+1)

        # validation description
        val_desc = f'\t Val Loss: {avg_val_loss:4f}'
        print(val_desc)
        log.write(val_desc)
    log.close()


if __name__ == '__main__':
    # parse script options
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_data', type=str, default='test_image/train', help='image data folder for training')
    parser.add_argument('--val_data', type=str, default='test_image/val', help='image data folder for validation')
    parser.add_argument('--val_ratio', type=float, default='0.2', help='ratio of validation')
    parser.add_argument('--epochs', type=int, default=100, help='num of epochs')
    parser.add_argument('--batch_size', type=int, default=8, help='total batch size')
    parser.add_argument('--lr', type=float, default=1e-4, help='learning rate')
    opt = parser.parse_args()

    # device
    device = 'cpu'
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print(f'using {device} : {torch.cuda.get_device_name()}\n')
    else:
        print(f'using {device}\n')
    
    train(opt,device)