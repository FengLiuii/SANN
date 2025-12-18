from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import HypergraphConv


class HyperGCN(nn.Module):
    def __init__(self, in_channels: int, hidden_channels: int, out_channels: int, dropout: float = 0.5, num_layers: int = 2):
        super().__init__()
        self.convs = nn.ModuleList()
        self.convs.append(HypergraphConv(in_channels, hidden_channels, use_attention=False))
        for _ in range(num_layers - 1):
            self.convs.append(HypergraphConv(hidden_channels, hidden_channels, use_attention=False))
        self.lin = nn.Linear(hidden_channels, out_channels)
        self.dropout = dropout

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        for conv in self.convs:
            x = F.relu(conv(x, edge_index))
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lin(x)
        return x


class HypergraphAttnConv(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.linear = nn.Linear(in_channels, out_channels, bias=False)
        self.att = nn.Parameter(torch.Tensor(1, 2 * out_channels))
        self.leaky_relu = nn.LeakyReLU(0.2)
        nn.init.xavier_uniform_(self.linear.weight)
        nn.init.xavier_uniform_(self.att)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        x = self.linear(x)
        row, col = edge_index
        a_input = torch.cat([x[row], x[col]], dim=-1)
        alpha = self.leaky_relu((a_input * self.att).sum(dim=-1))
        alpha = torch.exp(alpha)
        denom = torch.zeros(x.size(0), device=x.device).scatter_add_(0, col, alpha)
        alpha = alpha / (denom[col] + 1e-16)
        out = torch.zeros_like(x).scatter_add_(0, col.unsqueeze(-1).expand(-1, x.size(1)), alpha.unsqueeze(-1) * x[row])
        return out


class HyperGAT(nn.Module):
    def __init__(self, in_channels: int, hidden_channels: int, out_channels: int, dropout: float = 0.5, num_layers: int = 2):
        super().__init__()
        self.convs = nn.ModuleList()
        self.convs.append(HypergraphAttnConv(in_channels, hidden_channels))
        for _ in range(num_layers - 1):
            self.convs.append(HypergraphAttnConv(hidden_channels, hidden_channels))
        self.lin = nn.Linear(hidden_channels, out_channels)
        self.dropout = dropout

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        for conv in self.convs:
            x = F.relu(conv(x, edge_index))
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lin(x)
        return x


class TemporalTransformer(nn.Module):
    def __init__(self, feature_dim: int, model_dim: int = 128, num_heads: int = 4, num_layers: int = 2, dropout: float = 0.1):
        super().__init__()
        self.input_proj = nn.Linear(feature_dim, model_dim)
        self.layers = nn.ModuleList([
            _TemporalEncoderLayer(model_dim, num_heads, model_dim * 4, dropout)
            for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(model_dim)
        self.last_attn: torch.Tensor | None = None

    def forward(self, x_seq: torch.Tensor) -> torch.Tensor:
        # x_seq: [N, T, F]
        h = self.input_proj(x_seq)
        last_attn = None
        for layer in self.layers:
            h, attn = layer(h)
            last_attn = attn
        h = self.norm(h)
        self.last_attn = last_attn  # shape: [N, heads, T, T]
        # pool over time (mean)
        h = h.mean(dim=1)
        return h


class _TemporalEncoderLayer(nn.Module):
    def __init__(self, d_model: int, nhead: int, dim_feedforward: int, dropout: float):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(embed_dim=d_model, num_heads=nhead, dropout=dropout, batch_first=True)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, src: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # src: [N, T, C]
        attn_output, attn_weights = self.self_attn(src, src, src, need_weights=True, average_attn_weights=False)
        src = src + self.dropout1(attn_output)
        src = self.norm1(src)
        ff = self.linear2(self.dropout(F.relu(self.linear1(src))))
        src = src + self.dropout2(ff)
        src = self.norm2(src)
        # attn_weights: [N, heads, T, T]
        return src, attn_weights


class HyperTemporalModel(nn.Module):
    def __init__(self, base_model: nn.Module, feature_dim: int, temporal_dim: int = 128, fuse_dim: int = 128, dropout: float = 0.5):
        super().__init__()
        self.base_model = base_model
        self.temporal = TemporalTransformer(feature_dim=feature_dim, model_dim=temporal_dim)
        self.fuse = nn.Linear(temporal_dim + feature_dim, fuse_dim)
        self.classifier = nn.Linear(fuse_dim, getattr(base_model, 'lin').out_features if hasattr(base_model, 'lin') else 0)
        self.dropout = dropout

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, x_seq: torch.Tensor) -> torch.Tensor:
        # spatial path uses x and edge_index to produce logits-like embedding before final layer
        spatial_logits = self.base_model(x, edge_index)
        # temporal path encodes x_seq
        temporal_emb = self.temporal(x_seq)
        fused = torch.cat([x, temporal_emb], dim=1)
        fused = F.relu(self.fuse(fused))
        fused = F.dropout(fused, p=self.dropout, training=self.training)
        # project to same number of classes
        logits = self.classifier(fused) + spatial_logits
        return logits


__all__ = ["HyperGCN", "HyperGAT", "TemporalTransformer", "HyperTemporalModel"]


