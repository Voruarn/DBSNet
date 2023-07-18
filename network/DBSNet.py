import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from .layers import Conv2Module, ConvUp
from .init_weights import init_weights

from .pvt import pvt_tiny, pvt_small, pvt_medium, pvt_large
from .ResNet import *
from .CMG import SimGrafting
from .ASPP import *

class MLP(nn.Module):
    """
    Linear Embedding
    """
    def __init__(self, input_dim=2048, embed_dim=768):
        super().__init__()
        self.proj = nn.Linear(input_dim, embed_dim)

    def forward(self, x):
        x = x.flatten(2).transpose(1, 2)
        x = self.proj(x)
        return x


class ConvModule(nn.Module):
    def __init__(self, c1, c2, k=1, s=1, p=0, g=1, act=True):
        super(ConvModule, self).__init__()
        self.conv   = nn.Conv2d(c1, c2, k, s, p, groups=g, bias=False)
        self.bn     = nn.BatchNorm2d(c2, eps=0.001, momentum=0.03)
        self.act    = nn.ReLU() if act is True else (act if isinstance(act, nn.Module) else nn.Identity())

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

    def fuseforward(self, x):
        return self.act(self.conv(x))
   

class DBSNet(nn.Module):
    def __init__(self, n_channels=3, phi='s', is_deconv=True,
                is_batchnorm=True, dropout_ratio=0.1, aspp_dilate=[12, 24, 36]):
        super(DBSNet, self).__init__()      

        self.in_channels =[64, 128, 320, 512]

        self.Cbackbone  = ResNet()
        self.Tbackbone   = {
            't': pvt_tiny, 's': pvt_small, 'm': pvt_medium, 'l': pvt_large
        }[phi](pretrained=False)
        self.embedding_dim   = {
            't': 256, 's': 256, 'm': 768, 'l': 768
        }[phi]

        
        eout_channels=[64, 256, 512, 1024, 2048]
        filters = [64, 128, 320, 512, 1024] 

        embedding_dim=self.embedding_dim
        c1_in_channels, c2_in_channels, c3_in_channels, c4_in_channels = self.in_channels
        self.n_channels = n_channels
      
        self.is_deconv = is_deconv
        self.is_batchnorm = is_batchnorm
        
        self.sqz_c1=ConvModule(eout_channels[1], c1_in_channels)
        self.sqz_c2=ConvModule(eout_channels[2], c2_in_channels)
        self.sqz_c3=ConvModule(eout_channels[3], 64)
        self.sqz_c4=ConvModule(eout_channels[4], c4_in_channels)

        self.sqz_t3=ConvModule(c3_in_channels, 64)
        
        self.GF=SimGrafting(dim=64, out_dim=c3_in_channels, num_heads=8)
        self.AP=ASPP(c4_in_channels, c4_in_channels, aspp_dilate)


        self.maxpool4 = nn.MaxPool2d(kernel_size=2)
        self.center = Conv2Module(filters[3], filters[4], self.is_batchnorm)
        
        # upsampling
        self.up_concat4 = ConvUp(filters[4], filters[3], self.is_deconv)
        self.up_concat3 = ConvUp(filters[3], filters[2], self.is_deconv)
        self.up_concat2 = ConvUp(filters[2], filters[1], self.is_deconv)
        self.up_concat1 = ConvUp(filters[1], filters[0], self.is_deconv)
      
        self.linear_c4 = MLP(input_dim=c4_in_channels, embed_dim=embedding_dim)
        self.linear_c3 = MLP(input_dim=c3_in_channels, embed_dim=embedding_dim)
        self.linear_c2 = MLP(input_dim=c2_in_channels, embed_dim=embedding_dim)
        self.linear_c1 = MLP(input_dim=c1_in_channels, embed_dim=embedding_dim)

        self.linear_fuse = ConvModule(
            c1=embedding_dim*4,
            c2=embedding_dim,
            k=1,
        )
        
        out_ch=1
        self.dropout        = nn.Dropout2d(dropout_ratio)
        self.linear_pred    = nn.Conv2d(embedding_dim, out_ch, kernel_size=1)
        
        self.sidep1 = nn.Conv2d(embedding_dim, out_ch, kernel_size=1)
        self.sidep2 = nn.Conv2d(embedding_dim, out_ch, kernel_size=1)
        self.sidep3 = nn.Conv2d(embedding_dim, out_ch, kernel_size=1)
        self.sidep4 = nn.Conv2d(embedding_dim, out_ch, kernel_size=1)

        # initialise weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init_weights(m, init_type='kaiming')
            elif isinstance(m, nn.BatchNorm2d):
                init_weights(m, init_type='kaiming')


    def forward(self, inputs):
        H, W = inputs.size(2), inputs.size(3)

        # encoder
        Touts=self.Tbackbone.forward(inputs)
        t1, t2, t3, t4=Touts

        Couts = self.Cbackbone.forward(inputs)

        _, c1, c2, c3, c4 = Couts

        maxpool4 = self.maxpool4(t4)
        center = self.center(maxpool4)  
        
        c1=self.sqz_c1(c1)
        c2=self.sqz_c2(c2)
        c3=self.sqz_c3(c3)
        c4=self.sqz_c4(c4)

        t3=self.sqz_t3(t3)

        ct3=self.GF(c3,t3)
        ct4=c4 + t4
        ct4=self.AP(ct4)

        # decoder
        up4 = self.up_concat4(center, ct4)  
        up3 = self.up_concat3(up4, ct3) 
        up2 = self.up_concat2(up3, c2)  
        up1 = self.up_concat1(up2, c1) 

        n, _, h, w = up4.shape
        
        _c4 = self.linear_c4(up4).permute(0,2,1).reshape(n, -1, up4.shape[2], up4.shape[3])
        _c4 = F.interpolate(_c4, size=t1.size()[2:], mode='bilinear', align_corners=False)

        _c3 = self.linear_c3(up3).permute(0,2,1).reshape(n, -1, up3.shape[2], up3.shape[3])
        _c3 = F.interpolate(_c3, size=t1.size()[2:], mode='bilinear', align_corners=False)

        _c2 = self.linear_c2(up2).permute(0,2,1).reshape(n, -1, up2.shape[2], up2.shape[3])
        _c2 = F.interpolate(_c2, size=t1.size()[2:], mode='bilinear', align_corners=False)

        _c1 = self.linear_c1(up1).permute(0,2,1).reshape(n, -1, up1.shape[2], c1.shape[3])

        _c = self.linear_fuse(torch.cat([_c4, _c3, _c2, _c1], dim=1))

        x = self.dropout(_c)
        fuse = self.linear_pred(x)
        Sfuse = F.interpolate(fuse, size=(H, W), mode='bilinear', align_corners=True)
        
        d1 = self.sidep1(_c1)  
        S1= F.interpolate(d1, size=(H, W), mode='bilinear', align_corners=True)

        d2 = self.sidep2(_c2)  
        S2= F.interpolate(d2, size=(H, W), mode='bilinear', align_corners=True)

        d3 = self.sidep3(_c3)  
        S3= F.interpolate(d3, size=(H, W), mode='bilinear', align_corners=True)

        d4 = self.sidep4(_c4)  
        S4= F.interpolate(d4, size=(H, W), mode='bilinear', align_corners=True)

        return F.sigmoid(Sfuse), F.sigmoid(S1), F.sigmoid(S2), F.sigmoid(S3), F.sigmoid(S4)




