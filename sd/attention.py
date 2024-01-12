import torch
from torch import nn
from torch.nn import functional as F
import math

class SelfAttention(nn.Module):
    def __init__(self, n_heads: int, d_embed: int, in_proj_bias = True,out_proj_bias = True):
        super().__init__()
        self.in_proj_bias = nn.Linear(d_embed,3*d_embed,bias = in_proj_bias)
        self.out_proj_bias = nn.Linear(d_embed,d_embed, bias = out_proj_bias)
        self.n_heads = n_heads
        self.d_head = d_embed // n_heads

    def forward(self, x: torch.Tensor, casual_mask = False):
        # x: (Batch_size, seq_len, d_embed)

        input_shape = x.shape
        batch_size, sequence_length, d_embed = input_shape

        intermim_shape = (batch_size, sequence_length,self.n_heads,self.d_head)

        # (Batch_size, Seq_len, d_embed) -> (Batch_size, Seq_len, 3*d_embed) -> 3 tensors of shape (Batch_size, Seq_len, d_embed)
        q,k,v = self.in_proj(x).chunck(3,dim= -1)

        # (Batch_size, Seq_len, d_embed) -> (Batch_size, Seq_len, Heads, D_head/Heads) -> (Batch_size, Heads, Seq_len, D_head/Heads)
        q = q.view(intermim_shape).transpose(1,2)
        k = k.view(intermim_shape).transpose(1,2)
        v = v.view(intermim_shape).transpose(1,2)

        # (Batch_size, Heads, Seq_len)
        weight = q @ k.transpose(-1,-2) / math.sqrt(self.d_head)


        if casual_mask:
            # mask where the upper triangle(above the principal diagonal) is made up of 1
            mask = torch.ones_like(weiht, dtype = torch.bool).triu(1)
            weight.masksed_fill_(mask, -torch.inf)

        weight /= math.sqrt(self.d_head)

        weight = F.softmax(weight, dim = -1)

        # (Batch_size, H, Seq_len,Seq_len) @ (Batch_size, H, seq_len, Dim/H) -> (Batch_size, H, Seq_len, Dim/H)
        output = output.transpose(1,2)

        output = output.reshape(input_shape)

        # (Batch_size, Seq_len, d_embed) -> (Batch_size, Seq_len, d_embed)
        return output
        