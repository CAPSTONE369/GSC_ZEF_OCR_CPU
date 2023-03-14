import torch
import torch.nn as nn
import torch.nn.functional as F
import os, sys
from einops import rearrange
sys.path.append(os.getcwd())
from attention import MultiHeadAttention as MultiheadAttention

### Depth Wise Convolution
class DepthWiseConv(nn.Module):
  def __init__(self, nin, kernels_per_layer=1):
    super(DepthWiseConv, self).__init__()
    self.depthwise = nn.Conv2d(nin, nin *kernels_per_layer, kernel_size=3, padding=1, groups=nin)
  def forward(self, x):
    return self.depthwise(x)

class SeperableFeedForward(nn.Module):
  def __init__(self, 
              model_dim, dim_feedforward, activation):
    super(SeperableFeedForward, self).__init__()
    self.conv1 = nn.Sequential(
      nn.Conv2d(model_dim, dim_feedforward, kernel_size=1, padding=0, stride=1),activation)
 
    self.conv2 = nn.Sequential(DepthWiseConv(dim_feedforward, kernels_per_layer=1), activation)

    self.conv3 = nn.Sequential(
      nn.Conv2d(dim_feedforward, model_dim, kernel_size=1, padding=0, stride=1), activation)
  
  def forward(self, x):
    return self.conv3(self.conv2(self.conv1(x)))
  
class TransformerEncoderLayer(nn.Module):
  def __init__(self, 
               model_dim=512,
               head_num=8,
               dim_feedforward=2048,
               dropout=0.0,
               activation=nn.Tanh()): # nn.ReLU()): #nn.Tanh()): # nn.ReLU()): # nn.Tanh()):
    super(TransformerEncoderLayer, self).__init__()
    """ Args
    model_dim: dimension of the model
    head_num: number if heads in the multi head attention layer
    dim_feedforward: middle dimension in the feed forward network
    """

    ## (1) Multi Head Attention
    # self.self_attn = MultiHeadAttention(model_dim, head_num, dropout)
    self.self_attn = MultiheadAttention(
      embed_dim=model_dim, num_heads=head_num, dropout=0., bias=True, add_bias_kv=False,
      add_zero_attn=False, kdim=None, vdim=None, 
    ) ## [B, Seq Len, Embedding Dim]
    ## (2) Feed Forward Network
    self.dropout = nn.Dropout(dropout)
    self.dropout1 = nn.Dropout(dropout)
    self.dropout2 = nn.Dropout(dropout)
    
    ## (3) Add & Norm
    self.norm1 = nn.LayerNorm(model_dim)
    self.norm2 = nn.LayerNorm(model_dim)
    
    self.linear1 = nn.Linear(model_dim, dim_feedforward)
    self.activation = activation
    self.linear2 = nn.Linear(dim_feedforward, model_dim)
  
  
  def forward(self, x, height, width):
    """ Args
    x: input feature map from the ResNet-45
    """
    feature=x
    # print(f"SHAPE OF FEATURE: {feature.shape}")
    _, n, c = feature.shape

    # feature = x.view(n, c, -1).permute(2, 0, 1) ## [Batch, Embed Dim, HxW] -> [HxW, Batch, Embed Dim]
    attn, attn_weight = self.self_attn(x,x,x)
    x = x + self.dropout1(attn)
    x  = self.norm1(x) ## [HxW, Batch, Embed Dim]
    # x = x.view(n, c, -1).permute(2, 0, 1) ## [HxW, Batch, Embed Dim]
    attn = self.linear2(self.dropout(self.activation(self.linear1(x))))
    x = x + self.dropout2(attn)
    x = self.norm2(x)

    return x, attn_weight

class SeperableTransformerEncoderLayer(nn.Module):
  def __init__(self, 
               model_dim=512,
               head_num=8,
               dim_feedforward=2048,
               dropout=0.0,
               activation=nn.Tanh()): # nn.ReLU()): #nn.Tanh()): # nn.ReLU()): # nn.Tanh()):
    super(SeperableTransformerEncoderLayer, self).__init__()
    """ Args
    model_dim: dimension of the model
    head_num: number if heads in the multi head attention layer
    dim_feedforward: middle dimension in the feed forward network
    """

    ## (1) Multi Head Attention
    # self.self_attn = MultiHeadAttention(model_dim, head_num, dropout)
    self.self_attn = MultiheadAttention(
      embed_dim=model_dim, num_heads=head_num, dropout=0., bias=True, add_bias_kv=False,
      add_zero_attn=False, kdim=None, vdim=None, 
    ) ## [B, Seq Len, Embedding Dim]
    ## (2) Feed Forward Network
    self.conv_ffn = SeperableFeedForward(model_dim, dim_feedforward, activation)
    self.norm = nn.LayerNorm(model_dim)
    self.dropout1 = nn.Dropout(dropout)
    self.dropout2 = nn.Dropout(dropout)
    
    ## (3) Add & Norm
    self.norm1 = nn.LayerNorm(model_dim)
    self.norm2 = nn.LayerNorm(model_dim)
    """
    self.linear1 = nn.Linear(model_dim, dim_feedforward)
    self.dropout = nn.Dropout(dropout)
    self.activation = activation
    self.linear2 = nn.Linear(dim_feedforward, model_dim)
    """
  
  def forward(self, x, height, width):
    """ Args
    x: input feature map from the ResNet-45
    """
    feature=x
    # print(f"SHAPE OF FEATURE: {feature.shape}")
    _, n, c = feature.shape

    # feature = x.view(n, c, -1).permute(2, 0, 1) ## [Batch, Embed Dim, HxW] -> [HxW, Batch, Embed Dim]
    attn, attn_weight = self.self_attn(feature, feature, feature) 
    feature = feature + self.dropout1(attn)
    feature = self.norm1(feature) ## [HxW, Batch, Embed Dim]
    s, n, c = feature.shape
    x = rearrange(feature, '(h w) n c -> h w n c', h=height, w=width)
    x = rearrange(x, 'h w n c -> n c h w')
    # x = feature.contiguous().view(height, width, n, c).permute(2, 3, 0, 1) ## [Batch, Embed Dim, H, W]
    attn = self.conv_ffn(x)
    attn = rearrange(attn, 'n c h w -> n c (h w)', h=height, w=width)
    attn = rearrange(attn, 'n c s -> s n c')
    # x = x.view(n, c, -1).permute(2, 0, 1) ## [HxW, Batch, Embed Dim]
    # attn = self.linear2(self.dropout(self.activation(self.linear1(x))))
    x = feature + self.dropout2(attn)
    x = self.norm2(x)

    return x, attn_weight