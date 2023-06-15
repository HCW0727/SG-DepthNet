import torch
import torch.nn as nn

# Define the down block for the generator
class Down(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, 2, 1),
            nn.LeakyReLU(),
        )

    def forward(self, x):
        return self.conv(x)

# Define the up block for the generator
class Up(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.up = nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, 3, 2, 1),
            nn.LeakyReLU(),
        )

    def forward(self, x):
        return self.up(x)

# Generator (U-Net style)
class Generator(nn.Module):
    def __init__(self):
        super().__init__()
        self.down1 = Down(3, 64)
        self.down2 = Down(64, 128)
        self.down3 = Down(128, 256)
        self.up1 = Up(256, 128)
        self.up2 = Up(128, 64)

    def forward(self, x):
        x1 = self.down1(x)
        x2 = self.down2(x1)
        x3 = self.down3(x2)
        x = self.up1(x3)
        x = self.up2(x)
        return x

# Discriminator (PatchGAN style)
class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.down1 = Down(3, 64)
        self.down2 = Down(64, 128)
        self.down3 = Down(128, 256)
        self.final = nn.Conv2d(256, 1, 4, 1)

    def forward(self, x):
        x = self.down1(x)
        x = self.down2(x)
        x = self.down3(x)
        return self.final(x)

# Stereo Depth Prediction model (simple style)
class StereoDepthPrediction(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, 3, 1, 1)
        self.conv2 = nn.Conv2d(64, 128, 3, 1, 1)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv3 = nn.Conv2d(128, 1, 1, 1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.pool(x)
        x = self.conv2(x)
        x = self.pool(x)
        return self.conv3(x)
