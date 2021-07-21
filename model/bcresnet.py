from PIL.Image import init
import numpy as np
import os
import torch.nn as nn
import torch


import torch
import torch.nn as nn
from torch.nn.modules import padding

class Swish_func(torch.autograd.Function):
    @staticmethod
    def forward(ctx, i):
        result = i * torch.sigmoid(i)
        ctx.save_for_backward(i)
        return result

    @staticmethod
    def backward(ctx, grad_output):
        i = ctx.saved_variables[0]
        sigmoid_i = torch.sigmoid(i)
        return grad_output * (sigmoid_i * (1 + i * (1 - sigmoid_i)))
    

class Swish(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        pass
    def forward(self, input_tensor):
        return Swish_func.apply(input_tensor)



class SubSpectralNorm(nn.Module):
    def __init__(self,S=1):
        super().__init__()
        self.S = S
    def forward(self, x, gamma, beta, eps=1e-5):
        N,C,F,T = x.size()
        x = x.view(N, C*self.S, F//self.S, T)
        mean = x.mean([0,2,3]).view([1, C*self.S, 1, 1])
        var = x.var([0,2,3]).view([1, C*self.S, 1, 1])
        x =  gamma * (x - mean) / (var + eps).sqrt() + beta
        return x.view(N, C, F, T)


class BCResBlock(nn.Module):
    '''
    inputs: (B, F, W) B:batch size. F: frequency, W: time
    '''
    
    def __init__(self, in_channel, out_channel, stride, dilation, S=1):
        super().__init__()
        self.gamma = nn.Parameter(torch.rand([1,S*out_channel,1,1]))
        self.beta = nn.Parameter(torch.rand([1,S*out_channel,1,1]))
        self.init_conv = nn.Conv2d(in_channels=in_channel, out_channels=in_channel, kernel_size=1)
        self.bn1 = nn.BatchNorm2d(in_channel)
        self.bn2 = nn.BatchNorm2d(out_channel)
        self.relu = nn.ReLU()
        self.silu= Swish()
        self.ssn = SubSpectralNorm()
        self.f2 = nn.Conv2d(in_channels=in_channel, out_channels=out_channel, kernel_size=(3,1),stride=stride, padding=(1,0))
        self.avgpool = nn.AdaptiveAvgPool2d((1,None))
        self.f1 = nn.Conv2d(in_channels=out_channel, out_channels=out_channel, kernel_size=(1,3), stride=1,dilation=dilation, padding=(0,1))
        self.final_conv = nn.Conv2d(in_channels=out_channel, out_channels=out_channel, kernel_size=1)
        
    def forward(self, inputs):
        out = self.relu(self.bn1(self.init_conv(inputs)))
        x1 = self.f2(out)
        x1 = self.ssn(x1, self.gamma, self.beta)
        x = self.silu(self.bn2(self.avgpool(x1)))
        x2 = x.repeat([1,1,x1.size(2),1])
        x = x1+x2
        return x




class BCResNet(nn.Module):
    def __init__(self,num_classes=12, in_channel=1, n=[2,2,4,4], c=[8,12,16,20], s=[(1,1),(2,1),(2,1),(1,1)], d=[(1,1), (1,2), (1,4), (1,8)], ):
        super().__init__()
        self.stage1_conv = nn.Conv2d(in_channels=in_channel, out_channels=16,kernel_size=(5,5), stride=(2,1), dilation=1, padding=(2,2))
        self.block1 = self._make_layer(in_channel=16, out_channel=c[0], num_blocks=n[0], stride=s[0], dilation=d[0])
        self.block2 = self._make_layer(in_channel=c[0], out_channel=c[1], num_blocks=n[1], stride=s[1], dilation=d[1])
        self.block3 = self._make_layer(in_channel=c[1], out_channel=c[2], num_blocks=n[2], stride=s[2], dilation=d[2])
        self.block4 = self._make_layer(in_channel=c[2], out_channel=c[3], num_blocks=n[3], stride=s[3], dilation=d[3])
        self.stage2_conv = nn.Conv2d(in_channels=c[3], out_channels=32,kernel_size=(5,5), stride=(1,1), dilation=1, padding=(0,2))
        self.pool = nn.AdaptiveAvgPool2d((1,1))
        self.clasifier = nn.Conv2d(in_channels=32, out_channels=num_classes, kernel_size=(1,1))
    def _make_layer(self, in_channel, out_channel, num_blocks, stride, dilation):
        layers = []
        for i in range(num_blocks):
            if i==0:
                layers.append(BCResBlock(in_channel = in_channel, out_channel = out_channel , stride=stride, dilation=stride))
            else:
                layers.append(BCResBlock(in_channel = out_channel, out_channel = out_channel , stride=(1,1), dilation=(1,1)))
        return nn.Sequential(*layers)
    
    def forward(self,x):
        out = self.stage1_conv(x)
        out = self.block1(out)
        out = self.block2(out)
        out = self.block3(out)
        out = self.block4(out)
        out = self.stage2_conv(out)
        out = self.pool(out)
        out = self.clasifier(out).squeeze(3).squeeze(2)
        return out
        
        
