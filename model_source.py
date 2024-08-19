import torch
import torch.nn as nn

import numpy as np
import torch.nn.functional as F
from torch.nn.utils import spectral_norm

def snlinear(in_features, out_features):
    return spectral_norm(nn.Linear(in_features=in_features, out_features=out_features))

def snconv2d(in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True):
    return spectral_norm(nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                                   stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias))

class Generator(nn.Module):
    def __init__(self, d_conv_dim=32):
        super(Generator, self).__init__()
        self.d_conv_dim = d_conv_dim
        self.Dropout=nn.Dropout(0.1)
        self.Self_Attn1= Self_Attn(d_conv_dim)
        
        self.generator = nn.Sequential(
            nn.Dropout(0.1),
            spectral_norm(nn.Linear(4608, 256)),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(0.2, True),
            nn.Dropout(0.1),
            spectral_norm(nn.Linear(256, 128)),
            nn.BatchNorm1d(128),
            nn.Tanh()
            )
        

    def forward(self, *input):
        outv1 = forward_FFF(self, input[0])
        outf1 = forward_FFF(self, input[1])
        outf2 = forward_FFF(self, input[2])
        return outv1, outf1, outf2

def forward_FFF(self,x):
    # n x 32 x 12 x 12  
    N,C,H,W = x.size()
    h1=self.Dropout(x)
    h2 = self.Self_Attn1(h1) # n x d_conv_dim x 12 x 12
    h3 = self.generator(h2.view(N,-1))    
    return h3


class Dis(nn.Module):
    def __init__(self, d_conv_dim):
        super(Dis, self).__init__()
        #self.snlinear1 = snlinear(in_features=d_conv_dim*4, out_features=2)
        self.Dtrans = nn.Sequential(
            #nn.Dropout(0.3),
            snlinear(in_features=d_conv_dim*4, out_features=2),
            nn.BatchNorm1d(2),
        )
        
        
    def forward(self, in1, in2, in3):
        out1 = self.Dtrans(in1) # n*2
        out2 = self.Dtrans(in2) # n*2
        out3 = self.Dtrans(in3) # n*2
        return out1, out2, out3


class Class(nn.Module):
    def __init__(self, c_conv_dim):
        super(Class, self).__init__()
        self.trans = nn.Sequential(
            nn.Dropout(0.1),
            snlinear(in_features=c_conv_dim*4*3, out_features=2),
            nn.BatchNorm1d(2)
        )

    def forward(self, in1, in2, in3):
        out = torch.cat([in1, in2, in3], dim=1)
        return self.trans(out)

class class_sex(nn.Module):
    def __init__(self):
        super(class_sex, self).__init__()
        self.trans = nn.Sequential(
            nn.Dropout(0.1),
            spectral_norm(nn.Linear(128, 2)),
            nn.BatchNorm1d(2)
        )

    def forward(self, sex1, sex2, sex3):
        sex1 = self.trans(sex1)
        sex2 = self.trans(sex2)
        sex3 = self.trans(sex3)
        return sex1, sex2, sex3

class class_nation(nn.Module):
    def __init__(self):
        super(class_nation, self).__init__()
        self.trans = nn.Sequential(
            nn.Dropout(0.1),
            spectral_norm(nn.Linear(128, 3)),
            nn.BatchNorm1d(3)
        )

    def forward(self, nat1, nat2, nat3):
        nat1 = self.trans(nat1)
        nat2 = self.trans(nat2)
        nat3 = self.trans(nat3)
        return nat1, nat2, nat3




class feature_extractor(nn.Module):
    def __init__(self):
        super(feature_extractor, self).__init__()
        self._Conv2d1=nn.Conv2d(1, 3, 5, (2, 1), 1)
        self._AvgPool2d1=nn.AvgPool2d(kernel_size=(3, 5), stride=2, ceil_mode=False)
        self._BatchNorm2d1=nn.BatchNorm2d(3)
        self._LeakyReLU1=nn.LeakyReLU(0.2, True)
        self._Conv2d2=nn.Conv2d(3, 6, (3, 6), 2, 1)
        self._BatchNorm2d2=nn.BatchNorm2d(6)
        self._AvgPool2d2=nn.AvgPool2d(kernel_size=3, stride=2, ceil_mode=False)
        self._LeakyReLU2=nn.LeakyReLU(0.2, True)
        self._Conv2d3=nn.Conv2d(6, 12, 3, 1, 1)
        self._BatchNorm2d3=nn.BatchNorm2d(12)
        self._Conv2d4=nn.Conv2d(12, 16, 3, 1, 1)
        self._BatchNorm2d4=nn.BatchNorm2d(16)
        self._Conv2d5=nn.Conv2d(16, 32, 3, 1, 1)
        self._AvgPool2d5=nn.AvgPool2d(kernel_size=(4, 5), stride=1, padding=1, ceil_mode=False)
        self._Self_Attn1= Self_Attn(32)
        self._BatchNorm2d5=nn.BatchNorm2d(32)
        self._LeakyReLU5=nn.LeakyReLU(0.2, True)

        self.Conv2d1=nn.Conv2d(3, 3, 7, 2, 3)
        self.AvgPool2d1=nn.AvgPool2d(kernel_size=2, stride=2)
        self.BatchNorm2d1=nn.BatchNorm2d(3)
        self.LeakyReLU1=nn.LeakyReLU(0.2, True)
        self.Conv2d2=nn.Conv2d(3, 6, 5, 2, 2)
        self.AvgPool2d2=nn.AvgPool2d(kernel_size=2, stride=2, padding=1)
        self.BatchNorm2d2=nn.BatchNorm2d(6)
        self.LeakyReLU2=nn.LeakyReLU(0.2, True)
        self.Conv2d3=nn.Conv2d(6, 12, 2, 1, 4)
        self.BatchNorm2d3=nn.BatchNorm2d(12)
        self.LeakyReLU3=nn.LeakyReLU(0.2, True)
        self.Conv2d4=nn.Conv2d(12, 16, 3, 1, 1)
        self.BatchNorm2d4=nn.BatchNorm2d(16)
        self.Conv2d5=nn.Conv2d(16, 32, 3, 1, 1)
        self.BatchNorm2d5=nn.BatchNorm2d(32)
        self.AvgPool2d5=nn.AvgPool2d(kernel_size=2, stride=2, padding=1)
        self.Self_Attn1= Self_Attn(32)
        self.LeakyReLU5=nn.LeakyReLU(0.2, True)


    def forward(self, f1, v1, v2):
        f = feature_frame(self,f1)
        a1 = feature_audio(self,v1)
        a2 = feature_audio(self,v2)
        return f, a1, a2

def feature_audio(self,x):
    h1=self._Conv2d1(x)
    h1=self._AvgPool2d1(h1)
    h1=self._BatchNorm2d1(h1)
    h1=self._LeakyReLU1(h1)
    h2=self._Conv2d2(h1)
    h2=self._BatchNorm2d2(h2)
    h2=self._AvgPool2d2(h2)
    h2=self._LeakyReLU2(h2)
    h3=self._Conv2d3(h2)
    h3=self._BatchNorm2d3(h3)
    h4=self._Conv2d4(h3)
    h4=self._BatchNorm2d4(h4)
    h5=self._Conv2d5(h4)
    h5=self._AvgPool2d5(h5)
    h5=self._BatchNorm2d5(h5)
    h6=self._LeakyReLU5(h5)

    return h6

def feature_frame(self,x):
    h1=self.Conv2d1(x)
    h1=self.AvgPool2d1(h1)
    h1=self.BatchNorm2d1(h1)
    h1=self.LeakyReLU1(h1)
    h2=self.Conv2d2(h1)
    h2=self.AvgPool2d2(h2)
    h2=self.BatchNorm2d2(h2)
    h2=self.LeakyReLU2(h2)
    h3=self.Conv2d3(h2)
    h3=self.BatchNorm2d3(h3)
    h3=self.LeakyReLU3(h3)
    h4=self.Conv2d4(h3)
    h4=self.BatchNorm2d4(h4)
    h5=self.Conv2d5(h4)
    h5=self.BatchNorm2d5(h5)
    h5=self.AvgPool2d5(h5)
    h6=self.LeakyReLU5(h5)
    
    return h6



class DiscBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DiscBlock, self).__init__()
        self.relu = nn.ReLU()
        self.snconv2d1 = snconv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1)
        self.snconv2d2 = snconv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1)
        self.downsample = nn.AvgPool2d(2)
        self.ch_mismatch = False
        if in_channels != out_channels:
            self.ch_mismatch = True
        self.snconv2d0 = snconv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, x, downsample=True):
        x0 = x

        x = self.relu(x)
        x = self.snconv2d1(x)
        x = self.relu(x)
        x = self.snconv2d2(x)
        if downsample:
            x = self.downsample(x)

        if downsample or self.ch_mismatch:
            x0 = self.snconv2d0(x0)
            if downsample:
                x0 = self.downsample(x0)

        out = x + x0
        return out



class Self_Attn(nn.Module):
    """ Self attention Layer"""

    def __init__(self, in_channels):
        super(Self_Attn, self).__init__()
        #self.in_channels = in_channels
        self.snconv1x1_theta = snconv2d(in_channels=in_channels, out_channels=in_channels//8, kernel_size=1, stride=1, padding=0)
        self.snconv1x1_phi = snconv2d(in_channels=in_channels, out_channels=in_channels//8, kernel_size=1, stride=1, padding=0)
        self.snconv1x1_g = snconv2d(in_channels=in_channels, out_channels=in_channels//2, kernel_size=1, stride=1, padding=0)
        self.snconv1x1_attn = snconv2d(in_channels=in_channels//2, out_channels=in_channels, kernel_size=1, stride=1, padding=0)
        self.maxpool = nn.MaxPool2d(2, stride=2, padding=0)
        self.softmax  = nn.Softmax(dim=-1)
        self.sigma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        """
            inputs :
                x : input feature maps(B X C X W X H)
            returns :
                out : self attention value + input feature 
                attention: B X N X N (N is Width*Height)
        """
        _, ch, h, w = x.size()
        # Theta path
        theta = self.snconv1x1_theta(x)
        theta = theta.view(-1, ch//8, h*w)
        # Phi path
        phi = self.snconv1x1_phi(x)
        phi = self.maxpool(phi)
        phi = phi.view(-1, ch//8, h*w//4)
        # Attn map
        attn = torch.bmm(theta.permute(0, 2, 1), phi)
        attn = self.softmax(attn)
        # g path
        g = self.snconv1x1_g(x)
        g = self.maxpool(g)
        g = g.view(-1, ch//2, h*w//4)
        # Attn_g
        attn_g = torch.bmm(g, attn.permute(0, 2, 1))
        attn_g = attn_g.view(-1, ch//2, h, w)
        attn_g = self.snconv1x1_attn(attn_g)
        # Out
        out = x + self.sigma*attn_g
        return out

