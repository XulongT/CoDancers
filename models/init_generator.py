import numpy as np
import torch
import torch.nn as nn



class PositionEncoder(nn.Module):
    def __init__(self):
        super().__init__()

        input_dim = 1462
        hidden_dim = 512
        num_heads = 2
        dropout = 0.1
        encoder_layer = nn.TransformerEncoderLayer(d_model=input_dim, nhead=num_heads, dim_feedforward=hidden_dim, dropout=dropout)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)


    def forward(self, music):

        x = self.transformer(music)
        x = torch.mean(x, dim=1, keepdim=True)

        return x


class PositionEncoderGPT(nn.Module):
    def __init__(self):
        super().__init__()
        self.lr = nn.Sequential(
            nn.Linear(3, 128),
            nn.Linear(128, 768),
        )
        input_dim = 768
        hidden_dim = 512
        num_heads = 2
        dropout = 0.1
        layers = 2
        encoder_layer = nn.TransformerEncoderLayer(d_model=input_dim, nhead=num_heads, dim_feedforward=hidden_dim, dropout=dropout)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=layers)

    def forward(self, x):
        bk, t, _ = x.shape
        x = self.lr(x)

        x1 = x.reshape(-1, 7, t, 768).permute(0, 2, 1, 3).reshape(-1, 7, 768)
        cross_group = self.transformer(x1)
        person_xyz = cross_group.reshape(bk//7, t, 7, 768).permute(0, 2, 1, 3).reshape(-1, t, 768)
        x = person_xyz + x

        return x