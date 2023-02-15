import torch
import torch.nn as nn
import torch.nn.functional as F


def conv_block(in_channels, out_channels, pool=False):
    layers = [nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
              nn.BatchNorm2d(out_channels),
              nn.Mish(inplace=True)]
    if pool: layers.append(nn.MaxPool2d(2))
    return nn.Sequential(*layers)


class EncoderDA(nn.Module):
    def __init__(self, d, num_color_channel=3):
        super().__init__()
        self.in_channels = num_color_channel
        self.dim = d
        self.conv1 = conv_block(self.in_channels, 64)
        self.conv2 = conv_block(64, 128, pool=True)
        self.res1 = nn.Sequential(conv_block(128, 128), conv_block(128, 128))

        self.conv3 = conv_block(128, 256, pool=True)
        self.conv4 = conv_block(256, 512, pool=True)
        self.res2 = nn.Sequential(conv_block(512, 512), conv_block(512, 512))

        self.conv5 = conv_block(512, 768, pool=True)
        self.res3 = nn.Sequential(conv_block(768, 768))

        self.conv6 = conv_block(768, 768, pool=True)
        self.res4 = nn.Sequential(conv_block(768, 768))

        self.out = nn.Sequential(
            nn.Flatten(),
            nn.Linear(768, d),
        )

    def forward(self, xb):
        out = self.conv1(xb)
        out = self.conv2(out)
        out = self.res1(out) + out
        out = self.conv3(out)
        out = self.conv4(out)
        out = self.res2(out) + out
        out = self.conv5(out)
        out = self.res3(out) + out
        out = self.conv6(out)
        out = self.res4(out) + out
        zz = self.out(out)
        return zz


class Classifier(nn.Module):

    def __init__(self, d, num_classes):
        super().__init__()

        self.main = nn.Sequential(
            nn.Linear(d, d),
            nn.BatchNorm1d(d),
            nn.Mish(inplace=True),
            nn.Linear(d, num_classes),
        )

    def forward(self, x):
        return self.main(x)
