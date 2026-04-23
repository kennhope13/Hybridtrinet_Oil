import torch
import torch.nn as nn
import torch.nn.functional as F

class WaveletKANBlock(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(1, 1, in_dim))
        self.translation = nn.Parameter(torch.zeros(1, 1, in_dim))
        self.weight1 = nn.Linear(in_dim, out_dim)
        self.weight2 = nn.Linear(in_dim, out_dim)

    def forward(self, x):
        x_norm = (x - self.translation) / (self.scale + 1e-8)
        wavelet = (1 - x_norm ** 2) * torch.exp(-0.5 * x_norm ** 2)
        base = F.silu(self.weight1(x))
        wav  = self.weight2(wavelet)
        return base + wav

class GUMNet(nn.Module):
    def __init__(self, seq_len=30, input_dim=11, output_dim=1, horizon=5, d_feat=64, num_quantiles=3):
        super().__init__()
        self.seq_len = seq_len
        self.horizon = horizon
        self.output_dim = output_dim
        self.num_quantiles = num_quantiles
        
        self.cnn = nn.Conv1d(in_channels=input_dim, out_channels=d_feat, kernel_size=3, padding=2, dilation=2)
        self.cnn_pool = nn.AdaptiveAvgPool1d(1)
        
        self.gru = nn.GRU(input_dim, d_feat, num_layers=2, batch_first=True, dropout=0.1)
        self.attention = nn.MultiheadAttention(embed_dim=d_feat, num_heads=4, batch_first=True)
        
        self.kan = WaveletKANBlock(input_dim, d_feat)
        self.kan_pool = nn.AdaptiveAvgPool1d(1)

        self.gate = nn.Sequential(
            nn.Linear(d_feat * 3, 32),
            nn.ReLU(),
            nn.Linear(32, 3),
            nn.Softmax(dim=-1)
        )
        
        self.out_layer = nn.Linear(d_feat, horizon * output_dim * num_quantiles)

    def forward(self, x):
        B, L, D = x.shape
        
        x_cnn = x.transpose(1, 2)
        f_cnn = self.cnn_pool(F.relu(self.cnn(x_cnn))).squeeze(-1)
        
        gru_out, _ = self.gru(x)
        attn_out, _ = self.attention(gru_out, gru_out, gru_out)
        f_gru = attn_out[:, -1, :] 
        
        kan_out = self.kan(x)
        f_kan = self.kan_pool(kan_out.transpose(1, 2)).squeeze(-1)
        
        f_concat = torch.cat([f_cnn, f_gru, f_kan], dim=-1)
        weights = self.gate(f_concat)
        
        w_cnn = weights[:, 0].unsqueeze(1)
        w_gru = weights[:, 1].unsqueeze(1)
        w_kan = weights[:, 2].unsqueeze(1)
        
        f_fused = (w_cnn * f_cnn) + (w_gru * f_gru) + (w_kan * f_kan)
        
        out = self.out_layer(f_fused)
        out = out.view(B, self.horizon, self.output_dim, self.num_quantiles)
        return out, weights