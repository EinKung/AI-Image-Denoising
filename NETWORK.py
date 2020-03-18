import math
import torch
from torch import nn
import CONFIGURATION as config
from torchvision.models.vgg import vgg16

# *************************************ESRGAN*************************************

class Generator_ESR(nn.Module):
    def __init__(self):
        super(Generator_ESR, self).__init__()
        self.num_upsampleBlock=int(math.log(config.up_scale,2))
        self.input=nn.Sequential(
            nn.Conv2d(in_channels=3,out_channels=64,kernel_size=3,stride=1,padding=1),
            nn.PReLU()
        )

        self.rrdb=nn.Sequential(
            RRDB(64),
            RRDB(64),
            RRDB(64),
            RRDB(64),
            RRDB(64),
            RRDB(64)
        )

        self.up=nn.Sequential(*[UpsampleBlock(64,2) for no in range(self.num_upsampleBlock)])
        self.output=nn.Conv2d(in_channels=64,out_channels=3,kernel_size=3,stride=1,padding=1)

    def forward(self, x):
        input=self.input(x)
        rrdb=self.rrdb(input)
        up=self.up(rrdb)
        return self.output(up)

class Discriminator_ESR(nn.Module):
    def __init__(self):
        super(Discriminator_ESR, self).__init__()
        self.module = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2),

            ConvBlock_DisESR(64,64,3,2,1),
            ConvBlock_DisESR(64,128,3,1,1),
            ConvBlock_DisESR(128,128,3,2,1),
            ConvBlock_DisESR(128,256,3,1,1),
            ConvBlock_DisESR(256,256,3,2,1),
            ConvBlock_DisESR(256,512,3,1,1),
            ConvBlock_DisESR(512,512,3,2,1),

            nn.AdaptiveAvgPool2d(output_size=1),
            nn.Conv2d(512, 1024, kernel_size=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(1024, 1, kernel_size=1)
        )

    def forward(self, x):
        return torch.sigmoid(self.module(x).reshape([x.size()[0]]))

class RRDB(nn.Module):
    def __init__(self,in_channels):
        super(RRDB, self).__init__()
        self.module=nn.Sequential(
            DenseBlock(in_channels),
            DenseBlock(in_channels),
            DenseBlock(in_channels)
        )

    def forward(self, x):
        return self.module(x)+x

class DenseBlock(nn.Module):
    def __init__(self,in_channels):
        super(DenseBlock, self).__init__()
        self.basic01=BasicBlock(in_channels)
        self.basic02=BasicBlock(in_channels)
        self.basic03=BasicBlock(in_channels)
        self.basic04=BasicBlock(in_channels)
        self.conv=nn.Conv2d(in_channels=in_channels,out_channels=in_channels,kernel_size=3,stride=1,padding=1)

    def forward(self, x):
        basic01=self.basic01(x)+x
        basic02=self.basic02(basic01)+basic01+x
        basic03=self.basic03(basic02)+basic02+basic01+x
        basic04=self.basic04(basic03)+basic02+basic01+x
        return self.conv(basic04)+x

class BasicBlock(nn.Module):
    def __init__(self,in_channels):
        super(BasicBlock, self).__init__()
        self.module=nn.Sequential(
            nn.Conv2d(in_channels=in_channels,out_channels=in_channels,kernel_size=3,stride=1,padding=1),
            nn.PReLU()
        )

    def forward(self, x):
        return self.module(x)+x

class UpsampleBlock(nn.Module):
    def __init__(self,in_channels,up_scale):
        super(UpsampleBlock, self).__init__()
        self.module=nn.Sequential(
            nn.Conv2d(in_channels=in_channels,out_channels=in_channels*up_scale**2,kernel_size=3,stride=1,padding=1),
            nn.PixelShuffle(upscale_factor=up_scale),
            nn.PReLU()
        )

    def forward(self, x):
        return self.module(x)

class ConvBlock_DisESR(nn.Module):
    def __init__(self,in_channels,out_channels,ksize,stride=1,padding=0,grad=0.2):
        super(ConvBlock_DisESR, self).__init__()
        self.module=nn.Sequential(
            nn.Conv2d(in_channels=in_channels,out_channels=out_channels,kernel_size=ksize,stride=stride,padding=padding),
            nn.LeakyReLU(grad),
            nn.BatchNorm2d(out_channels)
        )

    def forward(self, x):
        return self.module(x)

# *************************************IDGAN*************************************

class Conv(nn.Module):
    def __init__(self,in_channels,out_channels,ksize,stride,padding):
        super(Conv, self).__init__()
        self.conv=nn.Sequential(
            nn.Conv2d(in_channels=in_channels,out_channels=out_channels,kernel_size=ksize,stride=stride,padding=padding),
            nn.BatchNorm2d(out_channels),
            nn.PReLU()
        )

    def forward(self, x):
        return self.conv(x)

class Generator_ID(nn.Module):
    def __init__(self):
        super(Generator_ID, self).__init__()
        self.firstLayer=Conv(in_channels=3,out_channels=32,ksize=9,stride=1,padding=4)

        self.preLayer=nn.Sequential(
            Conv(in_channels=32,out_channels=64,ksize=3,stride=1,padding=1),
            Conv(in_channels=64,out_channels=128,ksize=3,stride=1,padding=1)
        )

        self.resLayer=nn.Sequential(
            ResBlock_ID(128),
            ResBlock_ID(128),
            ResBlock_ID(128),
        )

        self.upLayer=nn.Sequential(
            DeconvBlock_ID(64),
            DeconvBlock_ID(32)
        )

        self.lastLayer=nn.Sequential(
            nn.Conv2d(in_channels=32,out_channels=3,kernel_size=9,stride=1,padding=4),
            nn.BatchNorm2d(3),
            nn.Tanh()
        )

    def forward(self, x):
        first=self.firstLayer(x)
        pre=self.preLayer(first)
        res=self.resLayer(pre)
        up=self.upLayer(res)
        last=self.lastLayer(up+first)
        return last

class DeconvBlock_ID(nn.Module):
    def __init__(self,in_channels):
        super(DeconvBlock_ID, self).__init__()
        self.up=UpsampleBlock(in_channels*2,2)
        self.conv=nn.Sequential(
            nn.Conv2d(in_channels=in_channels*2,out_channels=in_channels,kernel_size=3,stride=2,padding=1),
            nn.BatchNorm2d(in_channels),
            nn.PReLU()
        )

    def forward(self,x):
        return self.conv(self.up(x))

class ResBlock_ID(nn.Module):
    def __init__(self,in_channels):
        super(ResBlock_ID, self).__init__()
        self.block=nn.Sequential(
            nn.Conv2d(in_channels=in_channels,out_channels=in_channels//2,kernel_size=1,stride=1),
            nn.BatchNorm2d(in_channels//2),
            nn.PReLU(),

            nn.Conv2d(in_channels=in_channels//2,out_channels=in_channels,kernel_size=3,stride=1,padding=1),
            nn.BatchNorm2d(in_channels),
            nn.PReLU()
        )

    def forward(self, x):
        return self.block(x)+x

class Discriminator_ID(nn.Module):
    def __init__(self):
        super(Discriminator_ID, self).__init__()
        self.mainLayer=nn.Sequential(
            Conv(in_channels=3,out_channels=48,ksize=4,stride=2,padding=1),
            Conv(in_channels=48,out_channels=96,ksize=4,stride=2,padding=1),
            Conv(in_channels=96,out_channels=192,ksize=4,stride=2,padding=1),
            Conv(in_channels=192,out_channels=384,ksize=4,stride=1,padding=1)
        )

        self.preLayer=nn.Sequential(
            nn.Conv2d(in_channels=384,out_channels=1,kernel_size=4,stride=1,padding=1),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.preLayer(self.mainLayer(x))

# *************************************VGG16*************************************

class VGG16(nn.Module):
    def __init__(self,num):
        super(VGG16, self).__init__()
        vgg=vgg16(pretrained=True)
        module = nn.Sequential(*list(vgg.features)[:num]).eval()
        for param in module.parameters():
            param.requires_grad = False
        self.module = module

    def forward(self, x):
        return self.module(x)
