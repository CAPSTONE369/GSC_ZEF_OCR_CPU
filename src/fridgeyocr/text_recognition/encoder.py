import torch
import torch.nn as nn
import torch.nn.functional as F
import os, sys
sys.path.append(os.getcwd()) # text_recognition

from einops import rearrange
from position import PositionEncoding, Adaptive2DPositionalEncoding
from transformer import TransformerEncoderLayer, SeperableTransformerEncoderLayer
from resnet import resnet45

""" Transformer Encoder
- ResNet-50
- Multi-Head Attention
- Add&Norm (=Residual Connection & Layer Norm)
- Feed Forward
- Add&Norm
"""
import torch
import torch.nn as nn


class ShallowCNN(nn.Module):
  def __init__(self, ch_in, ch_mid, model_dim, activation):
    super(ShallowCNN, self).__init__()
    self.conv1 = nn.Sequential(
        nn.Conv2d(ch_in, ch_mid, kernel_size=3, stride=1, padding=1),
        nn.BatchNorm2d(ch_mid), nn.Tanh(),
        nn.MaxPool2d(kernel_size=2, stride=2)
    )
    self.conv2 = nn.Sequential(
        nn.Conv2d(ch_mid, model_dim, kernel_size=3, stride=1, padding=1),
        nn.BatchNorm2d(model_dim), nn.Tanh(),
        nn.MaxPool2d(kernel_size=2, stride=2)
    )
  def forward(self, x):
    x1 = self.conv1(x)
    x2 = self.conv2(x1)
    return x2

class ResTransformer(nn.Module):
    def __init__(self, 
                img_w, img_h,res_in,device,activation,rgb,use_resnet,
                 adaptive_pe, batch_size,seperable_ffn,
                 feedforward_dim=2048,
                 model_dim=512,
                 head_num=8,
                 dropout=0.1,
                 num_layers=5,):
        super(ResTransformer, self).__init__()
        if use_resnet:
            self.resnet = resnet45(res_in, rgb)
        else:
            ch_in = 3 if rgb else 1
            self.resnet = ShallowCNN(ch_in, ch_mid=64, model_dim=model_dim, activation=activation)
        self.img_w = img_w
        self.img_h = img_h
        self.d_model = model_dim
        self.nhead = head_num
        self.inner_dim = feedforward_dim
        self.dropout = dropout
        self.activation = activation# nn.ReLU(inplace=True) # nn.Tanh() # nn.GELU() # nn.ReLU()
        self.num_layers = num_layers
        self.adaptive_pe = adaptive_pe
        if adaptive_pe:
            self.pos_encoder = Adaptive2DPositionalEncoding(
                embedding_dim=model_dim, width=img_w//4, height=img_h//4, batch_size=batch_size,
            )
        else:
            self.pos_encoder = PositionEncoding(embedding_dim=self.d_model, max_length=(img_w//4) * (img_h // 4), dropout_rate = 0.1, device =device)
        
        if seperable_ffn:
            encoder_layer = SeperableTransformerEncoderLayer(model_dim=self.d_model, head_num=self.nhead, 
                dim_feedforward=self.inner_dim, dropout=self.dropout, activation=self.activation)
        else:
            encoder_layer = TransformerEncoderLayer(model_dim=self.d_model, head_num=self.nhead,
                dim_feedforward=self.inner_dim, dropout=self.dropout, activation=self.activation)
        self.transformer = nn.ModuleList([
            encoder_layer for _ in range(self.num_layers)
        ])


    def forward(self, images, batch_size):
        feature = self.resnet(images) ## (B, 512, 8, 32)
        n, c, h, w = feature.shape
        if self.adaptive_pe:
            x, feature = self.pos_encoder(feature, batch_size) ## (B, 512, 8, 32)
            n, w, h, e = x.shape
            # feature = rearrange(x, 'n w h e -> n e (w h)').permute(2, 0, 1).contiguous()
            # feature = x.contiguous().view(n, e, -1).permute(2, 0, 1) ## (8*32, B, 512)

        else:
            n, c, h, w = feature.shape
            feature = rearrange(feature, 'n c h w -> n c (h w)')
            feature = rearrange(feature, 'n c s -> s n c', n=n, c=c)
            # feature = feature.contiguous().view(n, c, -1).permute(2, 0, 1) ## (8*32, B, 512)
            feature = self.pos_encoder(feature,batch_size)
        
        

        for idx, layer in enumerate(self.transformer):
            if self.adaptive_pe:
                feature, attn_weight = layer(feature, height=h, width=w)
            else:
                feature, attn_weight = layer(feature, height=h, width=w)

        # feature = feature.permute(1, 2, 0).contiguous().view(n, c, h, w)
        S, B, E = feature.shape
        feature = rearrange(feature, 's b e -> b e s', b=B, e=E)
        feature = rearrange(feature, 'b e (h w) -> b e h w', h=self.img_h//4, w=self.img_w//4)
        return feature, attn_weight