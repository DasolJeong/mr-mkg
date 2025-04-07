"""
Convert NetworkX MMKG to PyTorch Geometric Data object.
"""

import torch
from torch_geometric.data import Data
import networkx as nx

def nx_to_pyg(graph: nx.MultiDiGraph, node_dim=768) -> Data:
    """
    Converts NetworkX MMKG to PyTorch Geometric Data format.

    Args:
        graph (nx.MultiDiGraph): MMKG graph with 'entity' nodes
        node_dim (int): Dimensionality of node features (random init)

    Returns:
        Data: PyG graph with x, edge_index
    """
    node_list = [n for n, d in graph.nodes(data=True) if d.get("type") == "entity"]
    node_id_map = {nid: i for i, nid in enumerate(node_list)}
    num_nodes = len(node_list)

    x = torch.randn((num_nodes, node_dim))  # TODO: replace with CLIP or RGAT init

    edge_index = []
    for src, tgt, d in graph.edges(data=True):
        if src in node_id_map and tgt in node_id_map:
            edge_index.append([node_id_map[src], node_id_map[tgt]])

    if not edge_index:
        edge_index = torch.empty((2, 0), dtype=torch.long)
    else:
        edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()

    return Data(x=x, edge_index=edge_index)
