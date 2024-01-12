import torch
from torch import nn
from torch.nn import functional as F
from attention import SelfAttention

class CLIPEmbedding(nn.Module):

    def __init__(self, n_vocab: int, n_embd: int, n_tokens= int):
        super().__init__()
        self.token_embedding = nn.Embedding(n_vocab, n_embd)
        self.position_embedding = nn.Parameter(torch.zeros(n_tokens, n_embd))

    def forward(self, tokens):
        # (batch_size,seq_len) -> (batch_size, seq_len, d_embed)
        x = self.token_embedding(tokens)
        x += self.position_embedding

        return x
    
class CLIPLayer(nn.Module):

    def __init__(self, n_heads: int, n_embd: int):
        super().__init__()

        self.layernorm_1 = nn.Layernorm(n_embd)
        self.attention = SelfAttention(n_heads, n_embd)
        self.layernorm_2 = nn.Layernorm(n_embd)
        self.linear_1 = nn.Linear(n_embd, 4*n_embd)
        self.linear_2 = nn.Linear(4*n_embd, n_embd)
    def forward(self,x: torch.Tensor) -> torch.Tensor:
        
        # (batch_size, seq_len, dim)
        residue = x

        ## Self attention

        x = self.layernorm_1(x)

        x = self.attention(x, casual_mask = True)

        x += residue

        ## Feed forward

        residue = x

        x = self.layernorm_2(x)

        x = self.linear_1(x)

        x = x * torch.sigmoid(1.702 *x) # quick gelu

        x = self.linear_2(x)

        x += residue

        return x
    
        



class CLIP(nn.Module):
    def __init__(self):
        self.embedding =CLIPEmbedding(49408,768,77)

        self.layers = nn.Module([
            CLIPLayer(12,768) for i in range(12)
        ])
        self.layernorm = nn.Layernorm(768)
    def forward(self, tokens: torch.Longtensor) -> torch.FloatTensor:
        tokens = tokens.type(torch.long)

        # (batch_size, seq_len) -> (batch_size, seq_len, d_embed)
        state = self.embedding(tokens)

        for layer in self.layers:
            state = layer(state)
        output = self.layernorm(state)

        return output