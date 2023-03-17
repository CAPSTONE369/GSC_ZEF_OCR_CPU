import torch
import torch.nn as nn
import torch.nn.functional as F
import os, sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from modules.position import PositionEncoding
from einops import rearrange
""" Attentional Decoder
- Encoder-Decoder
- Positional Encodings (초성-중성-종성)
"""

""" Attentional Decoder(=Position Attention)
"""

def encoder_layer(in_c, out_c, activation, k=3, s=2, p=1):
    return nn.Sequential(nn.Conv2d(in_c, out_c, k, s, p),
                         nn.BatchNorm2d(out_c),#  nn.LeakyReLU())
                         activation) # nn.ReLU()) # nn.Tanh()) #nn.ReLU(True))

def decoder_layer(in_c, out_c,activation, k=3, s=1, p=1, mode='nearest', scale_factor=None, size=None):
    align_corners = None if mode=='nearest' else True
    return nn.Sequential(nn.Upsample(size=size, scale_factor=scale_factor, 
                                     mode=mode, align_corners=align_corners),
                         nn.Conv2d(in_c, out_c, k, s, p),
                         nn.BatchNorm2d(out_c),# nn.LeakyReLU() )
                         activation) # nn.ReLU()) # nn.Tanh()) # nn.ReLU(True))


class Mini_UNet(nn.Module):
  def __init__(self, 
                h, w, activation,
               in_channels=512,
               num_channels=64,
               mode='nearest'):
    super(Mini_UNet, self).__init__()
    self.k_encoder = nn.Sequential(
            encoder_layer(in_channels, num_channels, activation, s=(1, 2)),
            encoder_layer(num_channels, num_channels, activation,s=(2, 2)),
            encoder_layer(num_channels, num_channels,activation, s=(2, 2)),
            encoder_layer(num_channels, num_channels, activation,s=(2, 2))
        )
    self.k_decoder = nn.Sequential(
            decoder_layer(num_channels, num_channels, activation,scale_factor=2, mode=mode),
            decoder_layer(num_channels, num_channels, activation,scale_factor=2, mode=mode),
            decoder_layer(num_channels, num_channels,activation, scale_factor=2, mode=mode),
            decoder_layer(num_channels, in_channels, activation,size=(h, w), mode=mode) ## Resizing을 하지 않음
        )
  def forward(self, k):
    features = []
    inp = k
    #print(f"UNET: {k.shape}")
    for i in range(0, len(self.k_encoder)):
      k = self.k_encoder[i](k)
      features.append(k)
      #print(k.shape)
    #print('--DECODER--')
    for i in range(0, len(self.k_decoder) - 1):
      k = self.k_decoder[i](k)
      # k = torch.cat([features[len(self.k_decoder)-2-i], k], dim=1)

      k = k + features[len(self.k_decoder) - 2 - i] 
      #print(k.shape)
      

    key = self.k_decoder[-1](k)
    key = key + inp ## residual addition to the unet -> 이게 없이는 마지막에 잘 gradient flow가 전달이 안되었을 수 있다.
    return key

if __name__ == "__main__":
  DEVICE = torch.device('cuda:6') if torch.cuda.is_available() else torch.device('cpu')
  unet = Mini_UNet(8, 48,nn.ReLU()).to(DEVICE)
  sample = torch.rand((1, 512, 8, 48)).to(DEVICE)
  out = unet(sample)
  print(out.shape)

class AttentionalDecoder(nn.Module):
  def __init__(self,
              img_h, img_w, activation,
               in_channel=512,
               unet_channel=64,
               max_seq_length=75, ## 논문에서의 수식으로 따지면 Length of Total Sequence를 의미한다.
               embedding_dim=512):
    super(AttentionalDecoder, self).__init__()
    self.max_length = max_seq_length
    self.unet = Mini_UNet(h=img_h//4, w=img_w//4,activation=activation,\
         in_channels = in_channel, num_channels=unet_channel) # Feature Extraction (key)
    self.project = nn.Linear(in_channel, in_channel)
    ## position encoding vector은 그 자체로 query로 사용되고, 이는 [MAX LENGTH, EMBEDDING DIM]
    self.pos_encoder = PositionEncoding(max_length = max_seq_length, embedding_dim = embedding_dim, dropout_rate = 0.1, device = 'cpu')
    
  def forward(self, x):
    """ Args
    x: input feature map that is reshaped (Batch Size=N, Embedding Dim = E, Height = H, Width = W)
    Outputs
    attn_vecs: (max_length, encoding_dim) the vector that has the attention
    1. Query와 Key^T를 dot production하면 [MAX LENGTH, SEQUENCE 길이 = (H/4  x W/4)]가 된다.
    2. 이후 normalize를 하고 softmax를 취해준 값에 Value를 dot product하면 [MAX LENGTH, EMBEDDING DIM]이 된다.
    """
    N, E, H, W = x.shape  ## [batch, embedding dim, height, width]
    ## (0) Value vector: Original output of the Transformer Encoder
    v = x
    ## (1) Get the Key vector that is the output of the UNet Model
    key = self.unet(x) ## KEY [H/4, W/4, Embedding Dim]

    ## (2) Calculate the Query Vector (=Position Encoding of the first-middle-last graphemes)
    zeros = x.new_zeros((self.max_length, N, E))  # [max length, batch, embedding dim]
    q = self.pos_encoder(zeros, batch_size=N)  # (T, N, E)
    T, N, E = q.shape
    q = rearrange(q, 't n e -> n t e')
    # q = q.permute(1, 0, 2)  # (N, T, E) = [batch, max length, embeedding dim]
    q = self.project(q)  # (N, T, E) -> 이 projection이 없어도 된다고 생각했었으나, 
                    # 작동 원리를 생각해 보면 입력 이미지마다 scale이 다를 수도 있기 때문에 위치를 미세 조정해서 적용하기 위해 있다고 볼수 있다.

    ## (3) Calculate the Attention Matrix 
    b,e,h,w=key.shape
    key = rearrange(key, 'b e h w -> b e (h w)', h=h, w=w)
    attn_scores = torch.bmm(q, key)  # (N, T, (H*W))
    attn_scores = attn_scores / (E ** 0.5) ## attention score normalize하기
    attn_scores = F.softmax(attn_scores, dim=-1)

    N, E, H, W = v.shape
    v = rearrange(v, 'n e h w -> n h w e')
    v = rearrange(v, 'n h w e -> n (h w) e')
    # v = v.permute(0, 2, 3, 1).view(N, -1, E)  # (N, (H*W), E)
    attn_vecs = torch.bmm(attn_scores, v)  # (N, T, E)

    attn_scores = rearrange(attn_scores, 'n e (h w) -> n e h w', h=H, w=W)
    return attn_vecs, attn_scores # .view(N, -1, H, W)
