import torch
import torch.nn as nn


class kernel_3(nn.Module):
    def __init__(self, input, output):
        super().__init__()
        self.conv = nn.Conv2d(input, output, kernel_size=3, stride=1, padding=1)

    def forward(self, x):

        return self.conv(x)


#####################################################################


class double_conv(nn.Module):
    def __init__(self, input, output):
        super().__init__()
        self.Double = nn.Sequential(
            kernel_3(input, output),
            nn.BatchNorm2d(output),
            nn.ReLU(),
            nn.Dropout2d(0.2),

            kernel_3(output, output),
            nn.BatchNorm2d(output),
            nn.ReLU(),
            nn.Dropout2d(0.2)
        )

    def forward(self, x):

        return self.Double(x)


#####################################################################


class down_conv(nn.Module):
    def __init__(self, input, output):
        super().__init__()
        self.down = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2),
            double_conv(input, output)
        )

    def forward(self, x):

        return self.down(x)


#####################################################################


class up_conv(nn.Module):
    def __init__(self, input, output):
        super().__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bicubic'),
            nn.Conv2d(input, input//2, kernel_size=1, stride=1)
        )

        self.down = double_conv(input, output)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        x = torch.cat([x2, x1], dim=1)

        return self.down(x)


#####################################################################


class UNet(nn.Module):
    def __init__(self, input, channels, output):
        super().__init__()
        self.input = input
        self.channels = channels
        self.output = output

        self.first_conv = double_conv(input, channels) #64, 224, 224
        self.down_conv1 = down_conv(channels, 2*channels) # 128, 112, 112
        self.down_conv2 = down_conv(2*channels, 4*channels) # 256, 56, 56
        self.down_conv3 = down_conv(4*channels, 8*channels) # 512, 28, 28
        
        self.middle_conv = down_conv(8*channels, 16*channels) # 1024, 14, 14 
        
        self.up_conv1 = up_conv(16*channels, 8*channels)
        self.up_conv2 = up_conv(8*channels, 4*channels)
        self.up_conv3 = up_conv(4*channels, 2*channels)
        self.up_conv4 = up_conv(2*channels, channels)
        
        self.last_conv = nn.Conv2d(channels, output, kernel_size=1, stride=1)

    def forward(self, x):
        x1 = self.first_conv(x)
        x2 = self.down_conv1(x1)
        x3 = self.down_conv2(x2)
        x4 = self.down_conv3(x3)
        
        x5 = self.middle_conv(x4)
        
        u1 = self.up_conv1(x5, x4)
        u2 = self.up_conv2(u1, x3)
        u3 = self.up_conv3(u2, x2)
        u4 = self.up_conv4(u3, x1)

        x = self.last_conv(u4)

        return x