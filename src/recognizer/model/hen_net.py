import torch
import torch.nn as nn
import torch.nn.functional as F
import sys, os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from transformation import TPS_SpatialTransformerNetwork
from encoder import ResTransformer
from decoder import AttentionalDecoder
from einops import rearrange
DEVICE=torch.device("cuda:6") if torch.cuda.is_available() else torch.device("cpu")
""" Hangul Net
1. ResModel (ResNet45)
2. Transformer Encoder
3. Position Attention based Decoder
4. Linear Classifier (Generates the class of the hangul graphemes)
"""
class HENNet(nn.Module):
  def __init__(self, 
              img_w, img_h, res_in, head_num, encoder_layer_num, tps,
              activation, use_resnet, adaptive_pe, batch_size,seperable_ffn,
              rgb=True,
              max_seq_length=75, ## 논문의 저자들이 지정한 가장 긴 길이의 sequence length인데, 75라는 것은 총 문자의 개수가 25개라는 것이다.
              embedding_dim=512, ## transformer encoder에서 model output의 dimension을 나타냄
              class_n=52, ## 한글에서의 초성-중성-종성의 개수를 나타냄
              ):
    super(HENNet, self).__init__()
    #self.resnet = resnet45()
    if activation.upper() == 'RELU':
      activation = nn.ReLU(inplace=True)
    elif activation.upper() == 'LEAKYRELU':
      activation = nn.LeakyReLU()
    elif activation.upper() == 'GELU':
      activation = nn.GELU()
    elif activation.upper() == 'TANH':
      activation = nn.Tanh()
    
    if rgb:
      in_ch=3
    else:in_ch=1
    if tps:
      self.transformation = TPS_SpatialTransformerNetwork(F=20, I_size=(img_h, img_w), I_r_size=(img_h, img_w), I_channel_num=in_ch)
    self.tps=tps
    self.transformer_encoder = ResTransformer(
      img_w=img_w, img_h=img_h, res_in=res_in, rgb=rgb,use_resnet=use_resnet,
      adaptive_pe=adaptive_pe, batch_size=batch_size,seperable_ffn=seperable_ffn,
      device=torch.device('cuda:6'), activation=activation, head_num=head_num,
      model_dim=embedding_dim, num_layers=encoder_layer_num) # 이 안에 ResNet45가 있음
    
    self.attention_decoder = AttentionalDecoder( 
              img_h=img_h, img_w=img_w,  activation=activation,
               in_channel=embedding_dim,
               unet_channel=64,
               max_seq_length=max_seq_length,
               embedding_dim=embedding_dim) 
    self.cls = nn.Linear(embedding_dim, class_n) # (N, T, Embed Dim) -> (N, T, Grapheme Class #)

  
  def forward(self, x,batch_size, mode='train'):
    #feature = self.resnet(x)
    #logger.info(feature.shape)
    if self.tps:
      x = self.transformation(x)

    encoder_out, attn_weight = self.transformer_encoder(x, batch_size)
    att_vec, att_score = self.attention_decoder(encoder_out)

    pred = self.cls(att_vec) # (Batch #, Seq Length, Grapheme Class #)

    return pred

if __name__ == "__main__":
  net = HENNet(img_w=192, img_h=32, res_in=64, head_num=8, encoder_layer_num=5, tps=False,
              activation='RELU', use_resnet=True, adaptive_pe=True, batch_size=1,seperable_ffn=True, rgb=False).to(DEVICE)
  sample = torch.rand((2, 1, 32,192)).to(DEVICE)

  out = net(sample, mode='train', batch_size=1)
  print(out.shape) ## (Batch Size, Max Sequence Length, Class Number)