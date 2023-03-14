import torch
import torch.nn as nn
import torch.nn.functional as F
import os, sys
sys.path.append(os.getcwd())

from encoder import ResTransformer
from decoder import AttentionalDecoder, AttentionalTransformerDecoder
from einops import rearrange

DEVICE=torch.device('cuda' if torch.cuda.is_available() else torch.device('cpu'))

class HENNet(nn.Module):
    def __init__(self,
                 img_w, img_h, res_in, encoder_layer_num, attentional_transformer,
                 class_n, adaptive_pe, batch_size, seperable_ffn, head_num,
                 use_conv=True, make_object_query=False, activation='relu',
                 use_resnet=True, rgb=False, max_seq_length=75, embedding_dim=512,
                 **kwargs):
        super(HENNet, self).__init__()
        activation = nn.ReLU(inplace=True)
        in_ch = 3 if rgb else 1
        self.transformer_encoder = ResTransformer(
            img_w=img_w, img_h=img_h, res_in=res_in, rgb=rgb, use_resnet=use_resnet,
            adaptive_pe=adaptive_pe, batch_size=batch_size, seperable_ffn=seperable_ffn,
            device=DEVICE, activation=activation, head_num=head_num,
            model_dim=embedding_dim, num_layers=encoder_layer_num
        )

        if attentional_transformer:
            self.attention_decoder = AttentionalTransformerDecoder(
                img_w=img_w, img_h=img_h, layer_n=3, hidden_dim=embedding_dim,
                head_n=8, max_seq_length=max_seq_length, feedforward_dim=2048
            )
        else:
            self.attention_decoder = AttentionalDecoder(
                img_h=img_h, img_w=img_w, activation=activation,
                make_object_query=make_object_query,
                in_channel=embedding_dim, unet_channel=64,
                max_seq_length=max_seq_length, embedding_dim=embedding_dim
            )
        self.use_conv = use_conv
        if use_conv:
            self.cls = nn.Conv1d(embedding_dim, class_n, kernel_size=1)
        else:
            self.cls = nn.Linear(embedding_dim, class_n)
    
    def forward(self, x, batch_size=1, mode='test'):
        encoder_out, attn_weight = self.transformer_encoder(x, batch_size)
        att_vec, att_score = self.attention_decoder(encoder_out)

        if self.use_conv:
            B, L, E = att_vec.shape
            att_vec = rearrange(att_vec, 'b l e -> b e l')
            pred = self.cls(att_vec)
            B, C, L = pred.shape
            pred = rearrange(pred, 'b c l -> b l c')
        else:
            pred = self.cls(att_vec)
        
        return pred
    
    def get_device(self):
        return next(self.parameters()).device