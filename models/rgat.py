import torch
import torch.nn as nn
from dgl.nn import GATConv


class RGATLayer(nn.Module):
    def __init__(self, in_dim, out_dim, num_relations, num_heads=1, dropout=0.1):
        super().__init__()
        self.num_relations = num_relations
        self.rel_embeddings = nn.Embedding(num_relations, in_dim)  # last index reserved for self-loop
        self.gat = GATConv(
            in_feats=in_dim,
            out_feats=out_dim // num_heads,
            num_heads=num_heads,
            feat_drop=dropout,
            attn_drop=dropout
        )

    def forward(self, g, node_features, relation_types):
        """
        Args:
            g: DGLGraph (with self-loops added)
            node_features: Tensor of shape [N, in_dim]
            relation_types: Tensor of shape [E_original] (relation IDs per edge)

        Returns:
            Tensor of shape [N, out_dim]
        """
        with g.local_scope():
            num_edges = g.num_edges()
            device = node_features.device

            # Append self-loop relation IDs if not included in rel_types
            if relation_types.shape[0] < num_edges:
                num_extra = num_edges - relation_types.shape[0]
                self_loop_id = self.num_relations - 1  # last ID reserved for self-loop
                extra_rel_ids = torch.full((num_extra,), self_loop_id, dtype=torch.long, device=device)
                relation_types = torch.cat([relation_types, extra_rel_ids], dim=0)

            g.ndata['h'] = node_features
            g.edata['rel_emb'] = self.rel_embeddings(relation_types)

            def edge_attention(edges):
                return {'e': edges.src['h'] + edges.data['rel_emb']}  # element-wise sum

            g.apply_edges(edge_attention)
            h_out = self.gat(g, g.ndata['h'])  # [N, num_heads, out_dim // num_heads]
            return h_out.mean(1)  # average over attention heads


class RGATEncoder(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, num_relations, num_layers=2):
        """
        Multi-layer RGAT encoder for knowledge graph reasoning.
        """
        super().__init__()
        self.layers = nn.ModuleList()
        dims = [in_dim] + [hidden_dim] * (num_layers - 1) + [out_dim]

        # Reserve one extra relation ID for self-loops
        for i in range(num_layers):
            self.layers.append(
                RGATLayer(dims[i], dims[i + 1], num_relations + 1)
            )

    def forward(self, g, node_features, relation_types):
        """
        Args:
            g: DGLGraph
            node_features: Tensor of shape [N, in_dim]
            relation_types: Tensor of shape [E]

        Returns:
            Tensor of shape [N, out_dim]
        """
        for layer in self.layers:
            node_features = layer(g, node_features, relation_types)
        return node_features
