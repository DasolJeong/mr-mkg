import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel
from models.rgat import RGATEncoder


class LanguageEncoder(nn.Module):
    def __init__(self, model_name="google/flan-t5-base"):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        base_model = AutoModel.from_pretrained(model_name)
        self.embedding_layer = base_model.get_input_embeddings()
        self.embedding_layer.requires_grad_(False)  # Freeze

    def forward(self, input_ids):
        """
        Args:
            input_ids: [B, L] - tokenized input from T5 tokenizer
        Returns:
            embeddings: [B, L, D]
        """
        return self.embedding_layer(input_ids)


class KGEncoderRGAT(nn.Module):
    def __init__(self, in_dim=512, hidden_dim=512, out_dim=768, num_rels=10, num_layers=2):
        """
        DGL-based Relation-aware Graph Attention Network encoder.
        Args:
            in_dim: input node feature dimension (e.g., from CLIP)
            hidden_dim: hidden dim in RGAT
            out_dim: final output dimension for each node
            num_rels: number of relation types in the KG
        """
        super().__init__()
        self.rgat = RGATEncoder(in_dim, hidden_dim, out_dim, num_rels=num_rels, num_layers=num_layers)

    def forward(self, g, node_feat, rel_types):
        """
        Args:
            g: DGLGraph
            node_feat: [N, in_dim]
            rel_types: [E] - relation type ID for each edge
        Returns:
            node embeddings: [N, out_dim]
        """
        return self.rgat(g, node_feat, rel_types)
