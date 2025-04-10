import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel
from models.rgat import RGATEncoder


class LanguageEncoder(nn.Module):
    def __init__(self, model_name="google/flan-t5-base"):
        super().__init__()
        base_model = AutoModel.from_pretrained(model_name)
        self.embedding = base_model.get_input_embeddings()
        self.embedding.requires_grad_(False)  # Freeze

    def forward(self, input_ids):
        """
        Args:
            input_ids: Tensor of shape [B, L]

        Returns:
            Tensor of shape [B, L, D] – token embeddings
        """
        return self.embedding(input_ids)


class KGEncoderRGAT(nn.Module):
    def __init__(self, in_dim=512, hidden_dim=512, out_dim=768, num_relations=10, num_layers=2):
        """
        Relation-aware GAT encoder for knowledge graphs.
        """
        super().__init__()
        self.rgat = RGATEncoder(
            in_dim=in_dim,
            hidden_dim=hidden_dim,
            out_dim=out_dim,
            num_relations=num_relations,
            num_layers=num_layers
        )

    def forward(self, graph, node_features, relation_types):
        """
        Args:
            graph: DGLGraph
            node_features: Tensor of shape [N, in_dim]
            relation_types: Tensor of shape [E] – edge relation types

        Returns:
            Tensor of shape [N, out_dim] – encoded node features
        """
        return self.rgat(graph, node_features, relation_types)