### Multi-Head Self Attention
import torch
import torch.nn as nn
import torch.nn.functional as F
import os, sys
sys.path.append(os.getcwd())
from mha_formula import multi_head_attention_forward

class MultiHeadAttention(nn.Module):
  def __init__(self,
               embed_dim,
               num_heads,
               dropout=0.0, kdim=None, vdim=None, bias=True, add_bias_kv=False, add_zero_attn=False):
    super(MultiHeadAttention, self).__init__()
    self.embed_dim = embed_dim
    self.kdim = embed_dim if kdim is None else kdim
    self.vdim = embed_dim if vdim is None else vdim

    self._qkv_same_embed_dim = self.kdim == embed_dim and self.vdim == embed_dim

    self.head_num = num_heads
    self.dropout = dropout
    self.head_dim = embed_dim // num_heads
    assert self.head_dim * num_heads == self.embed_dim, "embedding dimension must be divisble by number of heads"
    
    if self._qkv_same_embed_dim is False:
      self.q_proj_weight = nn.Parameter(torch.Tensor(embed_dim, embed_dim))
      self.k_proj_weight = nn.Parameter(torch.Tensor(embed_dim, self.kdim))
      self.v_proj_weight = nn.Parameter(torch.Tensor(embed_dim, self.vdim))
      self.register_parameter('in_proj_weight', None)
    
    else:
      self.in_proj_weight = nn.Parameter(torch.empty(3 * embed_dim, embed_dim))
      self.register_parameter('q_proj_weight', None)
      self.register_parameter('k_proj_weight', None)
      self.register_parameter('v_proj_weight', None)
    
    if bias:
      self.in_proj_bias = nn.Parameter(torch.empty(3 * embed_dim))
    else:
      self.register_parameter('in_proj_bias', None)
    self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
    
    if add_bias_kv:
      self.v_proj_bias = nn.Parameter(torch.empty(1, 1, embed_dim))
      self.k_proj_bias = nn.Parameter(torch.empty(1, 1, embed_dim))
    else:
      self.v_proj_bias = self.k_proj_bias = None
    
    self.add_zero_attn = add_zero_attn

    self._reset_parameters()
  
  def _reset_parameters(self):
    if self._qkv_same_embed_dim:
      nn.init.xavier_uniform_(self.in_proj_weight)
    else:
      nn.init.xavier_uniform_(self.q_proj_weight)
      nn.init.xavier_uniform_(self.k_proj_weight)
      nn.init.xavier_uniform_(self.v_proj_weight)
    
    if self.in_proj_bias is not None:
      nn.init.constant_(self.in_proj_bias, 0.)
      nn.init.constant_(self.out_proj.bias, 0.)
    if self.k_proj_bias is not None:
      nn.init.xavier_uniform_(self.k_proj_bias)
    if self.v_proj_bias is not None:
      nn.init.xavier_uniform_(self.v_proj_bias)
    
  def forward(self, query, key, value, key_padding_mask=None, need_weights=True, attn_mask=None):
    """ Args
    need_weights: attention weight도 return할지 말지
    attn_mask: mask처리가 된 부분을 제외하고 attention을 적용하도록 한다. => 그렇다면 가능하다면 한글이 아닌 영어 부분에 대한 masking처리도 가능?
    """
    if not self._qkv_same_embed_dim:
      return multi_head_attention_forward(
          query, key, value, self.embed_dim, self.head_num,
          self.in_proj_weight, self.in_proj_bias, 
          self.k_proj_bias, self.v_proj_bias, self.add_zero_attn,
          self.dropout, self.out_proj.weight, self.out_proj.bias,
          key_padding_mask=key_padding_mask, need_weights=need_weights, attn_mask=attn_mask,
          q_proj_weight=self.q_proj_weight, k_proj_weight=self.k_proj_weight, v_proj_weight=self.v_proj_weight
      )
    else:
      return multi_head_attention_forward(
          query, key, value, self.embed_dim, self.head_num,
          self.in_proj_weight, self.in_proj_bias, 
          self.k_proj_bias, self.v_proj_bias, self.add_zero_attn,
          self.dropout, self.out_proj.weight, self.out_proj.bias,
          key_padding_mask=key_padding_mask, need_weights=need_weights, attn_mask=attn_mask
      )
