import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl
from dgl.nn import GATConv

class RGATLayer(nn.Module):
    def __init__(self, in_dim, out_dim, num_rels, num_heads=1, dropout=0.1):
        super().__init__()
        self.num_rels = num_rels
        self.rel_emb = nn.Embedding(num_rels, in_dim)  # 마지막 index는 self-loop용
        self.gat = GATConv(in_dim, out_dim // num_heads, num_heads, feat_drop=dropout, attn_drop=dropout)

    def forward(self, g, h, rel_types):
        """
        g: DGLGraph (possibly with self-loops)
        h: [N, in_dim] node features
        rel_types: [E_original] edge type (relation id)
        """
        with g.local_scope():
            E = g.num_edges()
            device = h.device

            # self-loop으로 늘어난 edge 수만큼 self-rel ID로 채움
            if rel_types.shape[0] < E:
                num_extra = E - rel_types.shape[0]
                self_rel_id = self.num_rels - 1  # 마지막 ID는 self-loop
                extra = torch.full((num_extra,), self_rel_id, dtype=torch.long, device=device)
                rel_types = torch.cat([rel_types, extra], dim=0)

            g.edata['rel_emb'] = self.rel_emb(rel_types)
            g.ndata['h'] = h

            def edge_attention(edges):
                return {'e': edges.src['h'] + edges.data['rel_emb']}

            g.apply_edges(edge_attention)
            h_out = self.gat(g, g.ndata['h'])  # [N, num_heads, out_dim // num_heads]
            h_out = h_out.mean(1)  # mean over heads → [N, out_dim]
            return h_out


class RGATEncoder(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, num_rels, num_layers=2):
        super().__init__()
        # num_rels + 1: 마지막은 self-loop relation
        self.layers = nn.ModuleList()
        dims = [in_dim] + [hidden_dim] * (num_layers - 1) + [out_dim]

        for i in range(num_layers):
            self.layers.append(RGATLayer(dims[i], dims[i+1], num_rels + 1))

    def forward(self, g, feat, rel_types):
        for layer in self.layers:
            feat = layer(g, feat, rel_types)
        return feat  # [N, out_dim]