"""DGCNN-style EdgeConv model for edge-affinity prediction."""

from __future__ import annotations

import torch
from torch import nn


class EdgeConvBlock(nn.Module):
    """EdgeConv block with max aggregation over incoming neighbours."""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(2 * in_channels, out_channels),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True),
            nn.Linear(out_channels, out_channels),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        src, dst = edge_index[0], edge_index[1]
        msg_input = torch.cat([x[dst], x[src] - x[dst]], dim=1)
        msg = self.mlp(msg_input)
        out = torch.full((x.shape[0], msg.shape[1]), -1.0e9, device=x.device, dtype=msg.dtype)
        out.scatter_reduce_(0, dst[:, None].expand_as(msg), msg, reduce="amax", include_self=False)
        out = torch.where(out < -1.0e8, torch.zeros_like(out), out)
        return out


class EdgeAffinityDGCNN(nn.Module):
    """DGCNN/EdgeConv encoder plus edge-affinity classifier."""

    def __init__(self, point_channels: int = 7, edge_attr_channels: int = 15, hidden: int = 48, emb: int = 64):
        super().__init__()
        self.conv1 = EdgeConvBlock(point_channels, hidden)
        self.conv2 = EdgeConvBlock(hidden, emb)
        self.node_proj = nn.Sequential(
            nn.Linear(point_channels + hidden + emb, emb),
            nn.ReLU(inplace=True),
            nn.Linear(emb, emb),
            nn.ReLU(inplace=True),
        )
        self.edge_head = nn.Sequential(
            nn.Linear(emb * 3 + edge_attr_channels, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.10),
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 1),
        )

    def encode(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        h1 = self.conv1(x, edge_index)
        h2 = self.conv2(h1, edge_index)
        return self.node_proj(torch.cat([x, h1, h2], dim=1))

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        classify_edges: torch.Tensor,
        classify_edge_attr: torch.Tensor,
    ) -> torch.Tensor:
        h = self.encode(x, edge_index)
        src = classify_edges[:, 0]
        dst = classify_edges[:, 1]
        pair = torch.cat([h[src], h[dst], torch.abs(h[src] - h[dst]), classify_edge_attr], dim=1)
        return self.edge_head(pair).squeeze(1)
