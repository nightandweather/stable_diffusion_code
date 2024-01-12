import torch
from torch import nn
from torch.nn import functional as F
from attention import SelfAttention, CrossAttention

class TimeEmbedding(nn.Module):
    def __init__(self, n_embd: int):
        super().__init__()
        self.linear_1 = nn.Linear(320, 4*n_embd)
        self.linear_2 = nn.Linear(4*n_embd, n_embd)
    
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:

        # x: (1,320)

        x = self.linear_1(x)

        x = F.silu(x)

        x = self.linear_2(x)

        return x

class UNET_ResidualBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, n_time = 1200):
        super().__init__()
        self.groupnrom_feature = nn.GroupNorm(32, in_channels)
        self.conv_feature = nn.Conv2d(in_channels, out_channels, kernel_size = 3, padding = 1)
        self.linear_time = nn.Linear(n_time, out_channels)

        self.groupnorm_merged = nn.GroupNorm(32, out_channels)
        self.conv_merged = nn.Conv2d(out_channels, out_channels,kernel_size = 3, padding = 1)

        if in_channels = out_channels:
            self.residual_layer = nn.Identity()
        else:
            self.residual_layer = nn.Conv2d(in_channels, out_channels, kernel_size = 1, padding = 0)
    def forward(self, feature, time):
        # feature: (Batch_size, in_channels, height, width)
        # time: (1, 1200)

        residue = feature

        feature = self.groupnrom_feature(feature)

        feature = F.silu(feature)

        feature = self.conv_feature(feature)

        time = F.silu(time)

        time = self.linear_time(time)

        merged = feature + time.unsqueeze(-1).unsqueeze(-1)

        merged = self.groupnorm_merged(merged)

        merged = F.silu(merged)

        merged = self.conv_merged(merged)

        return merged + self.residual_layer(residue)

class Upsample(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        
        self.conv = nn.Conv2d(channels, channels, kernel_size = 3, padding = 1)
    def forward(self, x):
        # x: (Batch_size, channels, height, width) -> (Batch_size, Features, height*2, width*2)
        x = F.interpolate(x, scale_factor = 2, mode = "nearest")

        return self.conv(x)

class SwitchSequential(nn.Sequential):
    def forward(self, x: torch.Tensor, context: torch.Tensor, time: torch.Tensor) -> torch.Tensor:
        for layer in self:
            if isinstance(layer, UNET_AttentionBlock):
                x = layer(x, context)
            elif isinstance(layer, UNET_ResidualBlock):
                x = layer(x,time)
            else:
                x = layer(x)
        return x
    

class UNET(nn.Module):
    def __init__(self):
        super().__init__()

        self.encoder = nn.Module([
            # (Batch_size, 4, height/8, width/8) 
            SwitchSequential(nn.Conv2d(4, 320, kernel_size = 3, padding=1)),
            
            SwitchSequential(UNET_ResidualBlock(320,320), UNET_AttentionBlock(8,40)),
            
            SwitchSequential(UNET_ResidualBlock(320,320), UNET_AttentionBlock(8,40)),
            
            # (Batch_size, 4, height/8, width/8) -> (Batch_size, 4, height/16, width/16) 
            SwitchSequential(nn.Conv2d(320,320, kernel_size = 3, stride = 2, padding = 1)),

            SwitchSequential(UNET_ResidualBlock(320,640), UNET_AttentionBlock(8,80)),

            SwitchSequential(UNET_ResidualBlock(640,640), UNET_AttentionBlock(8,80)),

            # (Batch_size, 4, height/16, width/16) -> (Batch_size, 4, height/32, width/32) 
            SwitchSequential(nn.Conv2d(640,640, kernel_size = 3, stride = 2, padding = 1)),

            SwitchSequential(UNET_ResidualBlock(640,1280), UNET_AttentionBlock(8,160)),

            SwitchSequential(UNET_ResidualBlock(1280,1280), UNET_AttentionBlock(8,160)),

            # (Batch_size, 4, height/32, width/32) -> (Batch_size, 4, height/64, width/64)
            SwitchSequential(nn.Conv2d(1280,1280, kernel_size = 3, stride = 2, padding = 1)),

            SwitchSequential(UNET_ResidualBlock(1280,1280)),

            # (Batch_size, 1280,Height / 64, width / 64) -> (Batch_size, 1280, Height / 64, width / 64)
            SwitchSequential(UNET_ResidualBlock(1280,1280)),

        ])
        self.Bottleneck = nn.Module([
            UNET_ResidualBlock(1280,1280),

            UNET_AttentionBlock(8,160),

            UNET_ResidualBlock(1280,1280),

        ])

        self.decoder = nn.ModuleList([
            # (Batch_size, 2560, height/ 64, width / 64) -> (Batch_size, 1280, height/ 64, width / 64)
            SwitchSequential(UNET_ResidualBlock(2560,1280)),

            SwitchSequential(UNET_ResidualBlock(2560,1280)),

            SwitchSequential(UNET_ResidualBlock(2560,1280),Upsample(1280)),

            SwitchSequential(UNET_ResidualBlock(2560,1280), UNET_AttentionBlock(8,160)),

            SwitchSequential(UNET_ResidualBlock(1920,1280), UNET_AttentionBlock(8,160),Upsample(1280)),

            SwitchSequential(UNET_ResidualBlock(1920,1280), UNET_AttentionBlock(8,80)),

            SwitchSequential(UNET_ResidualBlock(1280,640), UNET_AttentionBlock(8,80)),

            SwitchSequential(UNET_ResidualBlock(960,640), UNET_AttentionBlock(8,80),Upsample(640)),

            SwitchSequential(UNET_ResidualBlock(960,320), UNET_AttentionBlock(8,40)),

            SwitchSequential(UNET_ResidualBlock(640,320), UNET_AttentionBlock(8,80)),

            SwitchSequential(UNET_ResidualBlock(640,320), UNET_AttentionBlock(8,40)),



        ])
class UNET_OutputLayer(nn.Module):
    def __init__(self,in_channels: int, out_channels: int):
        super().__init__()
        self.groupnorm = nn.GroupNorm(32, in_channels)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size = 3, padding = 1)

    def forward(self, x):
        # x: (Batch_size, 320, height / 8, height / 8)

        x = self.groupnorm(x)

        x = F.silu(x)

        x = self.conv(x)

        # (Batch_size, 4, height / 8, height / 8)
        return x
class Diffusion(nn.Module):
    def __init__(self)
        self.time_embedding = TimeEmbedding(320)
        self.unet = UNET()
        self.final = UNET_OutputLayer(320,4)

    def forward(self, x: torch.Tensor, context: torch.Tensor, time: torch.Tensor):
        # latent: (Batch_size, 4, Height / 8 , width / 8)
        # context: (Batch_size seq_len, Dim)
        # time: (1,320)

        # (1,320) -> (1, 1200)
        time = self.time_embedding(time)

        # (Batch, 4, height /8 width / 8) -> (Batch, 320 , height/8, width/8)
        output = self.unet(latent, context, time)

        # (Batch, 320, height/8, width/8) -> (Batch, 4, height/8, width/8)
        output = self.final(output)

        # (batch,,4,height/8,width/8)
        return output