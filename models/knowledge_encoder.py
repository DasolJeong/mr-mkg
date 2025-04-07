"""
Knowledge Encoder: RGAT implementation for MMKG entity subgraph using PyTorch Geometric.
"""

import torch
import torch.nn as nn
from torch_geometric.nn import GATConv

class RGATLayer(nn.Module):
    def __init__(self, in_dim, out_dim, heads=4):
        super().__init__()
        self.gat = GATConv(in_dim, out_dim // heads, heads=heads, add_self_loops=True)

    def forward(self, x, edge_index):
        return self.gat(x, edge_index)


class KnowledgeEncoder(nn.Module):
    def __init__(self, in_dim=768, hidden_dim=768, heads=4):
        super().__init__()
        self.rgat1 = RGATLayer(in_dim, hidden_dim, heads=heads)
        self.rgat2 = RGATLayer(hidden_dim, hidden_dim, heads=heads)
        self.relu = nn.ReLU()

    def forward(self, x, edge_index):
        x = self.rgat1(x, edge_index)
        x = self.relu(x)
        x = self.rgat2(x, edge_index)
        return x
