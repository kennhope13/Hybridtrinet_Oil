import math

import torch
import torch.nn as nn


class RBFExpansion(nn.Module):
    def __init__(self, in_features: int, M: int = 8):
        super().__init__()
        self.in_features = in_features
        self.M = M
        centers = torch.linspace(-1.0, 1.0, M).repeat(in_features, 1)
        self.centers = nn.Parameter(centers)
        self.gamma = nn.Parameter(torch.ones(in_features, M))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B = x.shape[0]
        x_ = x.unsqueeze(-1)
        c = self.centers.unsqueeze(0)
        g = torch.abs(self.gamma).unsqueeze(0) + 1e-6
        phi = torch.exp(-g * (x_ - c) ** 2)
        return phi.view(B, self.in_features * self.M)


class KANBlock(nn.Module):
    def __init__(self, in_features: int, hidden: int = 96, M: int = 8, p: float = 0.2):
        super().__init__()
        self.rbfe = RBFExpansion(in_features, M)
        self.lin1 = nn.Linear(in_features * M, hidden)
        self.act = nn.GELU()
        self.drop = nn.Dropout(p)
        self.lin2 = nn.Linear(hidden, in_features)
        self.proj = nn.Linear(in_features, in_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.rbfe(x)
        z = self.drop(self.act(self.lin1(z)))
        z = self.lin2(z)
        return self.act(z + self.proj(x))


class PosEnc(nn.Module):
    def __init__(self, d_model: int, max_len: int = 4096):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, : x.size(1), :]


class Patchify(nn.Module):
    def __init__(self, patch_len: int = 16, stride: int = 8, d_in: int = 1, d_model: int = 48):
        super().__init__()
        self.patch_len = patch_len
        self.stride = stride
        self.proj = nn.Linear(d_in * patch_len, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, D = x.shape
        n = 1 + (T - self.patch_len) // self.stride
        patches = []
        for i in range(n):
            s = i * self.stride
            e = s + self.patch_len
            patches.append(x[:, s:e, :].reshape(B, -1))
        P = torch.stack(patches, dim=1)
        return self.proj(P)


class HybridTriNet(nn.Module):
    def __init__(
        self,
        k: int,
        D: int,
        H: int,
        d_feat: int = 96,
        kan_M: int = 8,
        kan_depth: int = 2,
        kan_drop: float = 0.2,
        gru_hidden: int = 128,
        gru_layers: int = 1,
        gru_drop: float = 0.1,
        attn_dmodel: int = 48,
        attn_heads: int = 3,
        attn_layers: int = 2,
        attn_drop: float = 0.2,
        patch_len: int = 16,
        stride: int = 8,
    ):
        super().__init__()
        self.k = k
        self.D = D
        self.H = H

        self.kan_in = nn.Linear(k * D, D)
        self.kan_blocks = nn.Sequential(
            *[KANBlock(D, hidden=d_feat, M=kan_M, p=kan_drop) for _ in range(kan_depth)]
        )
        self.kan_head = nn.Sequential(nn.LayerNorm(D), nn.Linear(D, d_feat))

        self.gru = nn.GRU(
            D,
            gru_hidden,
            num_layers=gru_layers,
            batch_first=True,
            dropout=(gru_drop if gru_layers > 1 else 0.0),
        )
        self.gru_head = nn.Sequential(nn.LayerNorm(gru_hidden), nn.Linear(gru_hidden, d_feat))

        self.patch = Patchify(patch_len, stride, d_in=D, d_model=attn_dmodel)
        self.pos = PosEnc(attn_dmodel)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=attn_dmodel,
            nhead=attn_heads,
            dim_feedforward=4 * attn_dmodel,
            dropout=attn_drop,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=attn_layers)
        self.attn_head = nn.Sequential(nn.LayerNorm(attn_dmodel), nn.Linear(attn_dmodel, d_feat))

        self.out_kan = nn.Linear(d_feat, H * D)
        self.out_gru = nn.Linear(d_feat, H * D)
        self.out_att = nn.Linear(d_feat, H * D)

        self.gate = nn.Sequential(
            nn.Linear(3 * d_feat, 2 * d_feat),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(2 * d_feat, H * 3),
        )

    def forward(self, x: torch.Tensor):
        B = x.size(0)

        zk = self.kan_in(x.reshape(B, -1))
        zk = self.kan_blocks(zk)
        fk = self.kan_head(zk)

        hg, _ = self.gru(x)
        fg = self.gru_head(hg[:, -1, :])

        za = self.patch(x)
        za = self.pos(za)
        za = self.encoder(za)
        fa = self.attn_head(za.mean(1))

        yk = self.out_kan(fk).view(B, self.H, self.D)
        yg = self.out_gru(fg).view(B, self.H, self.D)
        ya = self.out_att(fa).view(B, self.H, self.D)

        fcat = torch.cat([fk, fg, fa], dim=1)
        w = self.gate(fcat).view(B, self.H, 3)
        w = torch.softmax(w, dim=-1)

        y = (
            w[..., 0, None] * yk
            + w[..., 1, None] * yg
            + w[..., 2, None] * ya
        )
        return y.view(B, -1), (yk, yg, ya, w)
