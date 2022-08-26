import torch
from torch import nn

class AutoEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels=1,out_channels=128,kernel_size=(2,2),padding='same'),
            nn.BatchNorm2d(num_features=128),
            nn.ReLU(),
            nn.Conv2d(in_channels=128,out_channels=128,kernel_size=(2,2),stride=(2,2)),
            nn.BatchNorm2d(num_features=128),
            nn.ReLU(),
            nn.Conv2d(in_channels=128,out_channels=128,kernel_size=(2,2),padding='same'),
            nn.BatchNorm2d(num_features=128),
            nn.ReLU(),
            nn.Conv2d(in_channels=128,out_channels=128,kernel_size=(2,2),stride=(2,2)),
            nn.BatchNorm2d(num_features=128),
            nn.ReLU(),
            nn.Conv2d(in_channels=128,out_channels=512,kernel_size=(2,2),padding='same'),
            nn.ReLU(),
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(in_channels=512,out_channels=512,kernel_size=(2,2),stride=(2,2)),
            nn.BatchNorm2d(num_features=512),
            nn.ReLU(),
            nn.Conv2d(in_channels=512,out_channels=256,kernel_size=(2,2),padding='same'),
            nn.BatchNorm2d(num_features=256),
            nn.ReLU(),
            nn.Conv2d(in_channels=256,out_channels=256,kernel_size=(2,2),padding='same'),
            nn.BatchNorm2d(num_features=256),
            nn.ReLU(),
            nn.Conv2d(in_channels=256,out_channels=128,kernel_size=(2,2),padding='same'),
            nn.BatchNorm2d(num_features=128),
            nn.ReLU(),
            nn.ConvTranspose2d(in_channels=128,out_channels=128,kernel_size=(2,2),stride=(2,2)),
            nn.ReLU(),
            nn.Conv2d(in_channels=128,out_channels=64,kernel_size=(2,2),padding='same'),
            nn.BatchNorm2d(num_features=64),
            nn.ReLU(),
            nn.Conv2d(in_channels=64,out_channels=1,kernel_size=(2,2),padding='same'),
            nn.ReLU()
        )

    def forward(self, input):
        # input : [batch, img_height, img_weight, channel]
        
        # conv2d input : [batch, channel, img_height, img_weight]
        input = torch.transpose(input,1,3)
        input = torch.transpose(input,2,3)
        compressed = self.encoder(input)
        output = self.decoder(compressed)
        
        output = torch.transpose(output,1,3)
        output = torch.transpose(output,1,2)
        return output
