import dgl
import torch
import networkx as nx

def convert_nx_to_dgl(nx_g: nx.MultiDiGraph):
    if nx_g.number_of_nodes() == 0:
        print("[Warning] Empty graph skipped.")
        return None, {}, {}, torch.tensor([])

    node2id = {node: i for i, node in enumerate(nx_g.nodes)}
    rel_set = set()
    
    src, dst, rel = [], [], []

    for u, v, data in nx_g.edges(data=True):
        r = data['relation']
        src.append(node2id[u])
        dst.append(node2id[v])
        rel.append(r)
        rel_set.add(r)

    rel2id = {r: i for i, r in enumerate(sorted(rel_set))}
    rel_types = torch.tensor([rel2id[r] for r in rel], dtype=torch.long)

    g = dgl.graph((src, dst), num_nodes=len(node2id))
    g.edata['rel_type'] = rel_types

    node_type_vocab = {"text": 0, "image": 1, "entity": 2}
    node_type_tensor = []
    for n in nx_g.nodes:
        t_str = nx_g.nodes[n].get("type", "entity")
        t_idx = node_type_vocab.get(t_str, 2)
        node_type_tensor.append(t_idx)

    # ✅ 노드 수와 feature 수가 일치하는지 확인
    if len(node_type_tensor) != g.num_nodes():
        print(f"[Error] Node feature mismatch: {len(node_type_tensor)} != {g.num_nodes()}")
        return None, {}, {}, torch.tensor([])

    g.ndata['ntype'] = torch.tensor(node_type_tensor, dtype=torch.long)

    # ✅ self-loop 추가는 비어있지 않을 때만
    if g.num_nodes() > 0:
        g = dgl.add_self_loop(g)

    return g, node2id, rel2id, rel_types

def extract_mrmkg_subgraph(G: nx.MultiDiGraph) -> nx.MultiDiGraph:
    """
    MR-MKG 논문 방식에 따라 질문 + 엔티티 중심 서브그래프 추출

    포함 요소:
    - 질문 노드
    - 이미지 노드 (있는 경우)
    - 엔티티 노드들
    - 엔티티로부터 1-hop 이웃 노드들 + 연결된 triple

    Returns:
        subgraph: nx.MultiDiGraph
    """
    center_nodes = set()
    
    # 1. 질문 노드
    question_node = next((n for n, d in G.nodes(data=True) if d.get("type") == "text"), None)
    if question_node:
        center_nodes.add(question_node)

    # 2. 이미지 노드 (있을 경우)
    for n, d in G.nodes(data=True):
        if d.get("type") == "image":
            center_nodes.add(n)

    # 3. 엔티티 노드들
    entity_nodes = [n for n, d in G.nodes(data=True) if d.get("type") == "entity"]
    center_nodes.update(entity_nodes)

    # 4. 엔티티에서 1-hop 이웃 노드들
    undirected = G.to_undirected()
    for ent in entity_nodes:
        if ent in undirected:
            neighbors = nx.single_source_shortest_path_length(undirected, ent, cutoff=1)
            center_nodes.update(neighbors.keys())

    # 서브그래프 생성
    subgraph = G.subgraph(center_nodes).copy()
    
    if subgraph.number_of_nodes() == 0:
        print(f"[Warning] Skipping empty subgraph.")
        return None
    
    return subgraph


def get_node_initial_embeddings(graph, node2id):
    """
    Create initial node embeddings for a given graph.

    Args:
        graph: The networkx graph object
        node2id: A dictionary mapping node ids to node index

    Returns:
        node_embeddings: Initial node embeddings (tensor)
    """
    # 노드 수만큼 임베딩 초기화 (512 차원으로 예시 설정)
    num_nodes = len(graph.nodes)
    node_embeddings = torch.randn(num_nodes, 512)  # [N, 512] 임베딩 초기화
    return node_embeddings