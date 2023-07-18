import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
Act = nn.ReLU

# Read PGNet, decoder, Grafting modules cop
# All Read Cop:9:53, 2023/4/25


def weight_init(module):
    for n, m in module.named_children():
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, (nn.BatchNorm2d, nn.InstanceNorm2d,nn.BatchNorm1d)):
            nn.init.ones_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Sequential):
            weight_init(m)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, (nn.ReLU,Act,nn.AdaptiveAvgPool2d,nn.Softmax)):
            pass
        else:
            m.initialize()

class SimGrafting(nn.Module):
    def __init__(self, dim, out_dim, num_heads=8, qkv_bias=True, qk_scale=None):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        self.k = nn.Linear(dim, dim , bias=qkv_bias)
        self.qv = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)
        
        self.shortcut = nn.Sequential(
            nn.Conv2d(64,out_dim,kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_dim))
        
        self.lnx = nn.LayerNorm(64)
        self.lny = nn.LayerNorm(64)
        self.conv2 = nn.Sequential(
            nn.Conv2d(64,out_dim,kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_dim,out_dim,kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_dim),
            nn.ReLU(inplace=True)
        )
    def forward(self, x, y):
        batch_size = x.shape[0]
        chanel     = x.shape[1]
        sc = x
        x = x.view(batch_size, chanel, -1).permute(0, 2, 1)
        sc1 = x
        x = self.lnx(x)
        y = y.view(batch_size, chanel, -1).permute(0, 2, 1)
        y = self.lny(y)
        
        B, N, C = x.shape
        y_k = self.k(y).reshape(B, N, 1, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        x_qv= self.qv(x).reshape(B,N,2,self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        x_q, x_v = x_qv[0], x_qv[1] 
        y_k = y_k[0]
        attn = (x_q @ y_k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        x = (attn @ x_v).transpose(1, 2).reshape(B, N, C)

        x = self.proj(x)
        x = (x+sc1)

        x = x.permute(0,2,1)
        x = x.view(batch_size,chanel,*sc.size()[2:])
        sc=self.shortcut(x)
        x = self.conv2(x)+sc
        return x


    def initialize(self):
        weight_init(self)

