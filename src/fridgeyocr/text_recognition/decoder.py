import torch
import torch.nn as nn
import torch.nn.functional as F
import os, sys
# DEVICE = torch.device('cuda:6' if torch.cuda.is_available() else 'cpu')
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
sys.path.append(os.getcwd())
from position import PositionEncoding
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

class ObjectQueryAttentionLayer(nn.Module):
  def __init__(self, query_dim, key_dim, value_dim, head_n, key_len, value_len, query_len, feedforward_dim):
    super(ObjectQueryAttentionLayer, self).__init__()
    """
    query_len = feature_map_size
    query_dim = max_seq_length
    key_dim == value_dim == embedding_dim
    key_len == value_len == feature_map_size
    """
    self.max_length = query_len
    self.feat_size = key_len
    self.embedding_dim = query_dim

    self.key_pe = PositionEncoding(max_length = key_len, embedding_dim = key_dim) ## [#size, #dim]
    self.query_pe = PositionEncoding(max_length = query_len, embedding_dim = query_dim) ## 무조건 position encoding vector => [#max_length, #dim]
    self.value_pe = PositionEncoding(max_length = value_len, embedding_dim = value_dim) ## [#size, #dim]
    
    self.value_project = nn.Linear(value_dim, value_dim)
    self.query_project = nn.Linear(query_dim, query_dim)

    self.attention = nn.MultiheadAttention(embed_dim = query_dim, kdim = key_dim, vdim = value_dim,  num_heads = head_n, batch_first = False)

    self.ln1 = nn.LayerNorm(self.embedding_dim)
    self.ffn = nn.Sequential(
        nn.Conv1d(self.embedding_dim, feedforward_dim, kernel_size=1),
        nn.ReLU(), nn.Conv1d(feedforward_dim, self.embedding_dim, kernel_size=1)
    )
    self.ln2 = nn.LayerNorm(self.embedding_dim)

  def forward(self, x):
    B, E, H, W = x.shape
    query = x.new_zeros((self.max_length, B, E))
    query = self.query_project(self.query_pe(query, batch_size=B)) ## [#max length, #batch, #embedding dim]

    value = x.new_zeros((self.feat_size, B, E))
    value = self.value_project(self.key_pe(value, batch_size=B))

    key = rearrange(x, 'b e h w -> (h w) b e', b=B,e=E,h=H,w=W)

    attn_output, attn_output_weight = self.attention(query, key, value) ## [#max length, #batch, #embedding dim]

    # (1) Add & Norm
    out = self.ln1(attn_output + query)
    # (2) Feed Forward Layer
    L, B, E = out.shape
    out2 = self.ffn(rearrange(out, 'l b e -> b e l'))
    B, E, L = out2.shape
    out2 = rearrange(out2, 'b e l -> l b e')
    # (3) Add & Norm
    out = self.ln2(out + out2)

    return out
    
    


class AttentionalDecoder(nn.Module):
  def __init__(self,
              img_h, img_w, activation, 
               make_object_query: bool, 
               in_channel=512,
               unet_channel=64,
               max_seq_length=75, ## 논문에서의 수식으로 따지면 Length of Total Sequence를 의미한다.
               embedding_dim=512):
    super(AttentionalDecoder, self).__init__()
    self.max_length = max_seq_length
    if make_object_query:
      self.object_query_attention = ObjectQueryAttentionLayer(query_dim=embedding_dim, key_dim=embedding_dim, value_dim=embedding_dim, \
                                                      query_len=max_seq_length, value_len=(img_h * img_w) //16, key_len=(img_h * img_w) // 16, \
                                                      head_n=8, feedforward_dim=1024)
    else:
      self.object_query_attention = None
    self.make_object_query = make_object_query
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
    if self.make_object_query:
      q = self.object_query_attention(x)
      T, N, E = q.shape # [#max length, #batch, #embedding dim]
      q = rearrange(q, 't n e -> n t e')
    
    else:
      zeros = x.new_zeros((self.max_length, N, E))  # [max length, batch, embedding dim]
      q = self.pos_encoder(zeros, batch_size=N)  # (T, N, E)
      T, N, E = q.shape
      q = rearrange(q, 't n e -> n t e')
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


# ---------------->> 여기부터는 Attentional Transformer Decoder ---------------- ## 
import torch
import torch.nn as nn
import torch.nn.functional as F
# !pip install einops
from einops import rearrange
""" Attentional Transformer Decoder
- 우선은 object query를 제대로 만든다. Transformer Decoder에 넣어주려고 하는 Embedding Vector이다.
  -> 목적이 Position Encoding Vector을 학습 시키는 것이다. -> [#seq, #dim]
  Query: Position Encoding Vector [#seq, #dim]
  Key: Encoder output (앞서 Transformer Encoder의 output이다.) [#size, #dim]
  Value: Position Encoding Vector [#size, #dim]

Object Query: [#seq, #dim]

  QK^T * V -> [#seq, #size] * V -> [#seq, #dim]
  - QK^T를 하면 각 sequence 위치가 "이미지에서 어떤 부분에 관심을 가져야하는지 제대로 학습"이 된다.
  - 이후 V를 곱해주면 한층 발전한 Position Encoding Vector이 만들어진다. (Skip Connection으로는 Value를 더해준다.)

- 이후 attention block에서는 목적이 학습된 위치 정보가 반영이 된 Feature Map을 학습 시키는 것이다. -> [#seq, #dim]
  1. 첫번째 attention은 self attention
  2. 두번째 attention은 encoder-decoder attention
  Query: 앞선 attention block의 output [#seq, #size] -> 단순히 Linear Projection을 한 HangulNet과 달리 이미지의 숫자|영어|한글의 차이에 따른 차별적으로 입력되는 position encoding이 될 것이다. -> 얘는 그냥 넣어줌 ㅋㅋ
  Key, Value: Transformer Encoder의 output feature map [#size, #dim] -> 여기서 UNet을 태운 feature map을 Key로 넣어줄지 말지는 고민 중 -> 얘네만 Linear Layer을 거침

"""
class MultiHeadAttentionLayer(nn.Module):
  def __init__(self, 
               query_dim: int,
               key_dim: int,
               value_dim: int,
               hidden_dim: int = 512,
               head_n: int = 5,
               dropout = 0.1):
    super(MultiHeadAttentionLayer, self).__init__()
    self.device = torch.device('cuda:6' if torch.cuda.is_available() else 'cpu')
    assert hidden_dim % head_n == 0
    self.hidden_dim = hidden_dim
    self.head_n = head_n
    self.head_dim = self.hidden_dim // self.head_n
  
    self.fc_q = nn.Linear(query_dim, hidden_dim) if query_dim > 0 else None
    self.fc_k = nn.Linear(key_dim, hidden_dim)
    self.fc_v = nn.Linear(value_dim, hidden_dim)

    self.fc_o = nn.Linear(hidden_dim, hidden_dim)
    self.dropout = nn.Dropout(dropout)

    self.scale = torch.sqrt(torch.FloatTensor([self.head_dim])).to(DEVICE)

  def forward(self, query, key, value):
    batch_size = query.shape[0]
    Q = self.fc_q(query) if self.fc_q is not None else query
    K = self.fc_k(key)
    V = self.fc_v(value)

    Q = Q.view(batch_size, -1, self.head_n, self.head_dim).permute(0, 2, 1, 3)
    K = K.view(batch_size, -1, self.head_n, self.head_dim).permute(0, 2, 1, 3)
    V = V.view(batch_size, -1, self.head_n, self.head_dim).permute(0, 2, 1, 3)

    energy = torch.matmul(Q, K.permute(0, 1, 3, 2)) / self.scale
    attention = torch.softmax(energy, dim=-1)
    x = torch.matmul(self.dropout(attention), V)
    x = x.permute(0, 2, 1, 3).contiguous()
    x = x.view(batch_size, -1, self.hidden_dim)
    x = self.fc_o(x)

    return x



class FeedForwardLayer(nn.Module):
  def __init__(self, 
               hidden_dim: int = 512,
               feedforward_dim: int=2048,
               dropout: float = 0.1):
    super(FeedForwardLayer, self).__init__()
    self.hidden_dim = hidden_dim
    self.fc1 = nn.Conv1d(hidden_dim, feedforward_dim, kernel_size =1) # nn.Linear(hidden_dim, feedforward_dim)
    self.fc2 = nn.Conv1d(feedforward_dim, hidden_dim, kernel_size=1) # nn.Linear(feedforward_dim, hidden_dim)
    self.dropout = nn.Dropout(dropout)
  
  def forward(self, x):
    B, L, C = x.shape
    transpose = False
    if (C == self.hidden_dim):
      transpose = True
      x = rearrange(x, 'b l c -> b c l')
    x = self.dropout(F.relu(self.fc1(x)))
    x = self.fc2(x)
    if transpose:
      x = rearrange(x, 'b c l -> b l c')
    return x

class AttentionalTransformerDecoderLayer(nn.Module):
  def __init__(self, 
               hidden_dim: int = 512,
               head_n: int = 8,
               feedforward_dim: int=2048
               ):
    super(AttentionalTransformerDecoderLayer, self).__init__()
    self.first_attention_ln = nn.LayerNorm(hidden_dim)
    self.second_attention_ln = nn.LayerNorm(hidden_dim)
    self.ffn_attention_ln = nn.LayerNorm(hidden_dim)
    
    self.first_attention = MultiHeadAttentionLayer(query_dim=hidden_dim, key_dim=hidden_dim, value_dim=hidden_dim, hidden_dim=hidden_dim, head_n=head_n)
    self.second_attention = MultiHeadAttentionLayer(query_dim=-1, key_dim=hidden_dim, value_dim=hidden_dim, hidden_dim=hidden_dim, head_n=head_n)
    self.ffn = FeedForwardLayer(hidden_dim, feedforward_dim, dropout=0.1)
  
    self.dropout = nn.Dropout(0.1)
  
  def forward(self, query, key, value):
    first_out = self.first_attention(query, query, query) ## Self Attention 
    target = self.first_attention_ln(query + self.dropout(first_out)) ## Add and Norm

    second_out = self.second_attention(target, key, key) ## Encoder-Decoder Attention
    target = self.second_attention_ln(target + self.dropout(second_out)) ## Add and Norm

    last_out = self.ffn(target)
    target = self.ffn_attention_ln(target + self.dropout(last_out))

    return target



class AttentionalTransformerDecoder(nn.Module):
  def __init__(self, 
               img_w: int,
               img_h: int,
               layer_n: int = 1,
               hidden_dim: int = 512,
               head_n: int = 8,
               max_seq_length: int = 75,
               feedforward_dim: int = 2048,
               ):
    super(AttentionalTransformerDecoder, self).__init__()

    self.img_w = img_w
    self.img_h = img_h
    self.max_seq_length = max_seq_length
    self.size = (img_h // 4) * (img_w // 4)

    self.size_pe = PositionEncoding(max_length=self.size, embedding_dim=hidden_dim)
    self.seq_pe = PositionEncoding(max_length=max_seq_length, embedding_dim=hidden_dim)
    self.size_project = nn.Linear(hidden_dim, hidden_dim)
    self.seq_project = nn.Linear(hidden_dim, hidden_dim)

    self.object_query_attention = MultiHeadAttentionLayer(query_dim=hidden_dim, key_dim=hidden_dim, value_dim=hidden_dim, head_n=4)
    self.layers = nn.ModuleList([
        AttentionalTransformerDecoderLayer(
            hidden_dim=hidden_dim, head_n=head_n, feedforward_dim=feedforward_dim
        )
     for _ in range(layer_n)])
  
  def _make_pe(self, x, size, mode='size'):
    N, E, H, W = x.shape
    zeros = x.new_zeros((size, N, E))
    if mode == 'size':
      pe = self.size_pe(zeros, batch_size=N)
      T, N, E = pe.shape
      pe = rearrange(pe, 't n e -> n t e')
      pe = self.size_project(pe)
    else:
      pe = self.seq_pe(zeros, batch_size=N)
      T, N, E = pe.shape
      pe = rearrange(pe, 't n e -> n t e')
      pe = self.seq_project(pe)
    return pe


  def forward(self, x):
    N, E, H, W = x.shape # batch size, embedding dim, height, width -> 얘는 계속해서 동일한 값을 넣어준다.
    seq_pe = self._make_pe(x, self.max_seq_length, mode = 'seq') ## [B, #seq, #dim] = [B, 75, 512]
    size_pe = self._make_pe(x, self.size, mode = 'size') ## [B, #size, #dim] = [B, 384, 512]

    key = rearrange(x, 'b e h w -> b (h w) e', h=H, w=W)

    object_query = self.object_query_attention(query=seq_pe, key=key, value=size_pe) ## [B, #seq, #dim]

    out = object_query
    for idx, layer in enumerate(self.layers):
      out = layer(query=out, key=key, value=key)
    
    return out, None



