import torch
from torch import nn

class autoencoder(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.input_size = input_size

        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels=1,out_channels=16,kernel_size=(2,2),stride=(2,2),padding=0),
            nn.Conv2d(in_channels=16,out_channels=32,kernel_size=(2,2),stride=(2,2)),
            nn.Conv2d(in_channels=32,out_channels=64,kernel_size=(2,2),stride=(2,2)),
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(in_channels=64,out_channels=32,kernel_size=(2,2),stride=(2,2)),
            nn.ConvTranspose2d(in_channels=32,out_channels=16,kernel_size=(2,2),stride=(2,2)),
            nn.ConvTranspose2d(in_channels=16,out_channels=1,kernel_size=(2,2),stride=(2,2)),
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