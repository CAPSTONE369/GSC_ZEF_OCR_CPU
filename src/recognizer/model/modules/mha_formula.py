### Compute the Forward Operation of MHA (Multi Head Attention)
import torch
import torch.nn as nn
import torch.nn.functional as F

def multi_head_attention_forward(query,
                                 key,
                                 value,
                                 embed_dim_to_check,
                                 head_num,
                                 in_proj_weight,
                                 in_proj_bias,
                                 k_proj_bias,
                                 v_proj_bias,
                                 add_zero_attn,
                                 dropout_p,
                                 out_proj_weight,
                                 out_proj_bias,
                                 key_padding_mask=None, need_weights=True, attn_mask=None,
                                 use_seperate_proj_weight=False, q_proj_weight=None, k_proj_weight=None, v_proj_weight=None
                                 ):
  seq_length, batch_size, embed_dim = query.size()
  assert embed_dim == embed_dim_to_check
  assert key.size() == value.size()

  head_dim = embed_dim // head_num
  scaling = float(head_dim) ** -0.5 ## scaled dot product attention에서 QK^T / (루트(dim of K)) 계산할 때 필요함

  if not use_seperate_proj_weight: ## 모두 같은 projection weight를 사용해야 하는 경우우
    if torch.equal(query, key) and torch.equal(key, value):
      ## self-attention
      q, k, v = F.linear(query, in_proj_weight, in_proj_bias).chunk(3, dim=-1)
  
  q = q * scaling

  if attn_mask is not None:
    if attn_mask.dtype == torch.uint8: ## byte type사용하면 안됨
      attn_mask = attn_mask.to(torch.bool)
    
    if attn_mask.dim() == 2:
      attn_mask = attn_mask.unsqueeze(0)
      if list(attn_mask.size()) != [1, query.size(0), key.size(0)]:
        raise RuntimeError('Size of the 2D attention mask is not correct')
    elif attn_mask.dim() == 3:
      if list(attn_mask.size()) != [batch_size * head_num, query.size(0), key.size(0)]:
        raise RuntimeError('Size of the 3D attention mask is not correct')
  if k_proj_bias is not None and v_proj_bias is not None:
    k = torch.cat([k, k_proj_bias.repeat(1, batch_size, 1)])
    v = torch.cat([v, v_proj_bias.repeat(1, batch_size, 1)])
    if attn_mask is not None:
      attn_mask = F.pad(attn_mask, (0, 1))
    if key_padding_mask is not None:
      key_padding_mask = F.pad(key_padding_mask, (0, 1))
  
  else:
    assert k_proj_bias is None
    assert v_proj_bias is None
  
  q = q.contiguous().view(seq_length, batch_size * head_num, head_dim).transpose(0, 1)
  k = k.contiguous().view(-1, batch_size * head_num, head_dim).transpose(0, 1)
  v = v.contiguous().view(-1, batch_size * head_num, head_dim).transpose(0, 1)

  out_seq_length = k.size(1) ## 왜냐면 Q와 K를 사용해서 행렬 연산을 계산하면 K의 1st dimension인 sequence의 길이 (Q와 같은수도 있고 다를수도 있음)


  if add_zero_attn:
    out_seq_length += 1
    k = torch.cat([k, torch.zeros((k.size(0), 1) + k.size()[2:], dtype = k.dtype, device = k.device)], dim=1)
    v = torch.cat([v, torch.zeros((v.size(0), 1) + v.size()[2:], dtype = v.dtype, device = v.device)], dim=1)
    if attn_mask is not None:
      attn_mask = F.pad(attn_mask, (0, 1))
    if key_padding_mask is not None:
      key_padding_mask = F.pad(key_padding_mask, (0, 1))
  
  attn_output_weights = torch.bmm(q, k.transpose(1, 2)) ## q: [Batch Size * Head Num, Seq Length, Head Dim] k^T: [Batch Size * Head Num, Head Dim, Seq Length]
  ## => Weight : [Batch Size * Head Num, Seq Length of Q, Seq Length of K]
  if attn_mask is not None:
    if attn_mask.dtype == torch.bool:
      attn_output_weights.masked_fill_(attn_mask, float('-inf'))
    else:
      attn_output_weights += attn_mask
  
  if key_padding_mask is not None:
    attn_output_weights = attn_output_weights.view(batch_size, head_num, out_seq_length, seq_length)
    attn_output_weights = attn_output_weights.masked_fill(
        key_padding_mask.unsqueeze(1).unsqueeze(2), float('-inf')
    )
    attn_output_weights = attn_output_weights.view(batch_size * head_num, out_seq_length, seq_length)
  
  attn_output_weights = F.softmax(attn_output_weights, dim=-1)
  attn_output = torch.bmm(attn_output_weights, v)
  attn_output = attn_output.transpose(0, 1).contiguous().view(out_seq_length, batch_size, embed_dim)
  attn_output = F.linear(attn_output, out_proj_weight, out_proj_bias)

  if need_weights:
    # average attention weights over heads
    attn_output_weights=  attn_output_weights.view(batch_size, head_num, out_seq_length, seq_length)
    return attn_output, attn_output_weights.sum(dim=1)/head_num
  else:
    return attn_output, None

