"""
Visual Adapter: Projects CLIP image embeddings into LLM embedding space.
"""

import torch.nn as nn

class VisualAdapter(nn.Module):
    def __init__(self, input_dim=512, output_dim=768):
        super().__init__()
        self.linear = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        return self.linear(x)


# models/knowledge_encoder.py
"""
Knowledge Encoder: Graph encoder (e.g., GAT) for MMKG entity subgraph.
"""

import torch.nn as nn

class KnowledgeEncoder(nn.Module):
    def __init__(self, in_dim=768, hidden_dim=768):
        super().__init__()
        self.encoder = nn.Identity()  # placeholder for RGAT/GAT/GNN

    def forward(self, graph):
        # Return entity embeddings (currently no-op)
        return self.encoder(graph)