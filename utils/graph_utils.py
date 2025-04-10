import torch
import dgl
import networkx as nx


def convert_nx_to_dgl(nx_graph: nx.MultiDiGraph):
    """
    Convert a NetworkX MMKG graph into a DGLGraph.

    Args:
        nx_graph: NetworkX MultiDiGraph

    Returns:
        dgl_graph: DGLGraph
        node2id: Dict[str, int]
        rel2id: Dict[str, int]
        rel_types: Tensor of shape [E] – relation type index per edge
    """
    if nx_graph.number_of_nodes() == 0:
        print("[Warning] Empty graph skipped.")
        return None, {}, {}, torch.tensor([])

    node2id = {node: i for i, node in enumerate(nx_graph.nodes)}
    src, dst, rel = [], [], []
    rel_set = set()

    for u, v, data in nx_graph.edges(data=True):
        r = data["relation"]
        src.append(node2id[u])
        dst.append(node2id[v])
        rel.append(r)
        rel_set.add(r)

    rel2id = {r: i for i, r in enumerate(sorted(rel_set))}
    rel_types = torch.tensor([rel2id[r] for r in rel], dtype=torch.long)

    dgl_graph = dgl.graph((src, dst), num_nodes=len(node2id))
    dgl_graph.edata["rel_type"] = rel_types

    # Assign node types
    node_type_vocab = {"text": 0, "image": 1, "entity": 2}
    node_type_ids = []

    for node in nx_graph.nodes:
        node_type_str = nx_graph.nodes[node].get("type", "entity")
        node_type_id = node_type_vocab.get(node_type_str, 2)
        node_type_ids.append(node_type_id)

    if len(node_type_ids) != dgl_graph.num_nodes():
        print(f"[Error] Node count mismatch: {len(node_type_ids)} vs {dgl_graph.num_nodes()}")
        return None, {}, {}, torch.tensor([])

    dgl_graph.ndata["ntype"] = torch.tensor(node_type_ids, dtype=torch.long)

    if dgl_graph.num_nodes() > 0:
        dgl_graph = dgl.add_self_loop(dgl_graph)

    return dgl_graph, node2id, rel2id, rel_types


def extract_mrmkg_subgraph(nx_graph: nx.MultiDiGraph) -> nx.MultiDiGraph:
    """
    Extract MR-MKG style subgraph: question + entities + image + 1-hop neighbors.

    Args:
        nx_graph: Full MMKG NetworkX graph

    Returns:
        subgraph: extracted NetworkX subgraph (or None if empty)
    """
    center_nodes = set()

    # Add question node
    question_node = next((n for n, d in nx_graph.nodes(data=True) if d.get("type") == "text"), None)
    if question_node:
        center_nodes.add(question_node)

    # Add image node(s)
    image_nodes = [n for n, d in nx_graph.nodes(data=True) if d.get("type") == "image"]
    center_nodes.update(image_nodes)

    # Add entity nodes
    entity_nodes = [n for n, d in nx_graph.nodes(data=True) if d.get("type") == "entity"]
    center_nodes.update(entity_nodes)

    # Add 1-hop neighbors of entity nodes
    undirected = nx_graph.to_undirected()
    for ent in entity_nodes:
        if ent in undirected:
            neighbors = nx.single_source_shortest_path_length(undirected, ent, cutoff=1)
            center_nodes.update(neighbors.keys())

    subgraph = nx_graph.subgraph(center_nodes).copy()

    if subgraph.number_of_nodes() == 0:
        print("[Warning] Skipping empty subgraph.")
        return None

    return subgraph


def get_node_initial_embeddings(nx_graph, node2id):
    """
    Initialize node embeddings for input graph.

    Args:
        nx_graph: NetworkX graph
        node2id: Dict[node_key, int] mapping

    Returns:
        Tensor of shape [N, 512] with random initialization
    """
    num_nodes = len(nx_graph.nodes)
    return torch.randn(num_nodes, 512)
