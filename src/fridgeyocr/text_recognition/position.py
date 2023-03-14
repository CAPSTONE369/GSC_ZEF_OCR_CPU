import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import math
from einops import rearrange

"""Postitional Encoding (L, C)
- Uses the <register_buffer>
  : if you have parameters in the model, which should be saved and restored in the state_dict
    but not trained by the optimizer, you should register them as "buffers"
- The positional encoding module outputs the input feature map tensor added pixel wise with the position encoded vector
- In the paper, the max_length is set to the 75. (Not written on the paper, but is told by the author of the paper)
- In the paper, the embedding dimension is not written.
"""
USE_CUDA=torch.cuda.is_available()
DEVICE=torch.device('cuda' if USE_CUDA else 'cpu')

def get_sinusoid_encoding_table(max_length, embedding_dim):
  def cal_angle(position, dim_i):
    return position / np.power(10000, 2 * (dim_i // 2) / embedding_dim)
  def get_posi_angle_vec(position):
    return [cal_angle(position, dim_i) for dim_i in range(embedding_dim)]
  
  sinusoid_table = torch.FloatTensor([get_posi_angle_vec(length_i) for length_i in range(max_length)])
  sinusoid_table[:, 0::2] = torch.sin(sinusoid_table[:, 0::2]) # 짝수 index
  sinusoid_table[:, 1::2] = torch.cos(sinusoid_table[:, 1::2]) # 홀수 index

  return sinusoid_table

def get_position_encoding(length, hidden_dim, device):
  pe = torch.zeros(length, hidden_dim, device=device)
  pos = torch.arange(length)
  pos = pos.float().unsqueeze(dim=1)
  pe[:, 0::2] = torch.sin(pos / (10000 ** (torch.arange(0, hidden_dim, 2)/hidden_dim)))
  pe[:, 1::2] = torch.sin(pos / (10000 ** (torch.arange(0, hidden_dim, 2)/hidden_dim)))

  return pe

class Adaptive2DPositionalEncoding(nn.Module):
  def __init__(self, embedding_dim, width, height, batch_size, device=DEVICE):
    super(Adaptive2DPositionalEncoding, self).__init__()
    self.inter_fc = nn.Sequential(
        nn.Linear(embedding_dim, embedding_dim//2, bias=False), nn.ReLU(), nn.Linear(embedding_dim//2, 2* embedding_dim, bias=False), nn.Sigmoid()
    )
    pe = torch.zeros(width, embedding_dim, device=device)
    pe.requires_grad=False
    pos = torch.arange(width)
    pos = pos.float().unsqueeze(dim=1)
    pe[:, 0::2] = torch.sin(pos / (10000 ** (torch.arange(0, embedding_dim, 2)/embedding_dim)))
    pe[:, 1::2] = torch.sin(pos / (10000 ** (torch.arange(0, embedding_dim, 2)/embedding_dim)))
    pe_width = pe
    pe = torch.zeros(height, embedding_dim, device=device)
    pe.requires_grad=False
    pos = torch.arange(height)
    pos = pos.float().unsqueeze(dim=1)
    pe[:, 0::2] = torch.sin(pos / (10000 ** (torch.arange(0,embedding_dim, 2)/embedding_dim)))
    pe[:, 1::2] = torch.sin(pos / (10000 ** (torch.arange(0, embedding_dim, 2)/embedding_dim)))
    pe_height = pe
    #pe_width = get_position_encoding(width, embedding_dim, device)
    #pe_height = get_position_encoding(height, embedding_dim, device)
    pe_width = torch.unsqueeze(pe_width, dim=1)
    pe_height = torch.unsqueeze(pe_height, dim=0)
    pe_width = torch.tile(torch.unsqueeze(pe_width, dim=0), [1, 1, 1])
    pe_height = torch.tile(torch.unsqueeze(pe_height, dim=0), [1, 1, 1])

    self.register_buffer('pe_width', pe_width)
    self.register_buffer('pe_height', pe_height)

  def forward(self, x, batch_size):
    # x: output feature map from the Shallow CNN
    # x = [B, C, H, W]
    B, C, H, W = x.shape
    x = rearrange(x, 'b c h w -> b w h c')
    # x = x.permute(0, 3,2,1).contiguous() # [batch_size, width, height, embedding_dim]
    inter = torch.mean(x, dim = (1,2))
    alpha = self.inter_fc(inter)
    n, c = alpha.shape
    alpha = rearrange(alpha, 'n (a b k) -> n a b k', k=C, n=n,a=2, b=1)
    #alpha = alpha.view(-1, 2, 1, C).contiguous()
    
    pos_encoding = alpha[:, 0:1, :, :] * self.pe_height + alpha[:, 1:2, :, :] * self.pe_width
    
    x = x + pos_encoding # x += pos_encoding -> 이렇게 더해주니 inplace error이 backward pass를 할 때 발생하였다.
    B,W, H, E = x.shape
    feature = rearrange(x, 'b w h d -> b d (h w)')
  
    feature = rearrange(feature, 'b d s -> s b d', d=E, b=B)
    # feature = x.contiguous().view(B, E, -1).permute(2, 0, 1).contiguous()
    # feature = x.contiguous().view(B, E, -1).permute(2, 0, 1)

    return x, feature


class PositionEncoding(nn.Module):
  def __init__(self, 
              max_length=75, # +1, ## Additional Stop Token 추가
              embedding_dim=512,
              dropout_rate=0.1,
              device=DEVICE):
    super(PositionEncoding, self).__init__()
    """sin, cos encoding 구현
    max_length: 전체 단어 / 문장의 최대 길이 (단, Hangul Net에서는 3 X 단어의 수이다.)
    embedding_dim: Dimension of the model
    """
    self.dropout = None
    # self.dropout = nn.Dropout(dropout_rate)
    #encoding = torch.zeros(max_length, embedding_dim, device = device)
    #encoding.requires_grad = False
    pe = torch.zeros(max_length, embedding_dim, device=device)
    pe.requires_grad=False
    pos = torch.arange(0, max_length, device=device)
    pos = pos.float().unsqueeze(dim=1)

    _2i = torch.arange(0, embedding_dim, step=2, device=device).float()
    pe[:, 0::2] = torch.sin(pos / (10000**(_2i / embedding_dim)))
    pe[:, 1::2] = torch.cos(pos / (10000**(_2i / embedding_dim)))
    pe = torch.unsqueeze(pe, dim=1)
    # self.project = nn.Linear(embedding_dim, embedding_dim)
    #self.pos_encoding = get_sinusoid_encoding_table(max_length, embedding_dim)
    # self.pe  = nn.Embedding.from_pretrained(self.pos_encoding, freeze=True)
    #positions = torch.arange()
    # pos = torch.arange(0, max_length, device = device)
    # pos = pos.float().unsqueeze(dim = 1)
    # div_term = torch.exp(torch.arange(0, embedding_dim, 2).float() * (-math.log(10000.0) / embedding_dim)).to(device)
    # _2i = torch.arange(0, embedding_dim, step = 2, device = device).float()
    
    #encoding[:, ::2] = torch.sin(pos * div_term) # torch.sin(pos / (1000 ** (_2i / embedding_dim)))
    #encoding[:, 1::2] = torch.cos(pos*div_term) # torch.cos(pos / (1000 ** (_2i / embedding_dim)))
    #encoding = encoding.unsqueeze(0).transpose(0, 1)
    self.register_buffer('pe', pe)
    
  
  def forward(self, x, batch_size):
    if len(x.shape) == 4:
      n, c, h, w = x.shape
      x = rearrange(x, 'n c h w -> (h w) n c', h=h,w=w,c=c)
    """ Args
    별건 아니고 1,2,3.. 순서대로 알아서 위치 정보에 대한 embedded vector을 입력 sequence에 더해 주면 결과를 모델이 알아서 학습을 하게 될 것이다.
    x: (sequence_length, batch_size, embedding_dimension)
    out: (sequence_length, batch_size, embedding_dimension)
    """
    seq_len, batch_size, embed_dim = x.shape
    #positions = torch.arange(seq_len, device = x.device, dtype=x.dtype).expand(seq_len, batch_size, embed_dim).contiguous() + 1
    #pos_mask = x.eq(0)
    #positions.masked_fill_(pos_mask, 0)
    # pos_embs = self.pe(positions)
    #self.pe = self.pe.to(x.device)
    # self.pe = torch.unsqueeze(self.pe, dim=1) 
    """
    신기하게도 여기서 forward pass에서 unsqueeze를 하게 되면 backward pass할 때에
    메모리 초과가 발생하는 것을 확인할 수 있었다.
    미리 unsqueeze를 해 주어야 됬었고, 또 bach내의 각 item마다 똑같은 position encoding vector을 더해 주어야 했다.
    """
    x = x + self.pe[:seq_len, :]
    #print(x.shape)
    # x = x + pos_embs[:seq_len, :] ## input embedding인 x와 position embedding을 더해주면 된다.
    if self.dropout is not None:
      x = self.dropout(x)
    return x



if __name__ == "__main__":
  import os
  import numpy as np
  import matplotlib.pyplot as plt
  FIGURE_PATH='/home/guest/ocr_exp_v2/figures'
  PE = PositionEncoding(
    max_length=75, embedding_dim=512
  ).to(DEVICE)

  pos_encode_vec = PE.pe
  print(np.unique(pos_encode_vec.detach().cpu().numpy()))
  # print(pos_encode_vec.shape)
  sample = torch.zeros((75, 1, 512)).to(DEVICE)
  out = PE(sample)
  # plt.pcolormesh(pos_encode_vec.detach().cpu().numpy()[:,0,:], cmap='RdBu')
  plt.pcolormesh(out.detach().cpu().numpy()[:,0,:], cmap = 'RdBu')
  plt.xlabel('Depth')
  plt.xlim((0, 512))
  plt.ylabel('Position')
  plt.colorbar()
  plt.savefig(os.path.join(FIGURE_PATH, 'postion_encoding_output.png'))

