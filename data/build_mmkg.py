"""
Construct a multimodal knowledge graph (MMKG) for a ScienceQA question using image, text, and extracted entities.
"""

import networkx as nx
from typing import List, Optional
from pathlib import Path
import torch


def build_mmkg(question_id: int, question_text: str, image_path: Optional[str], entities: List[str]) -> nx.MultiDiGraph:
    """
    Build an MMKG graph for a ScienceQA sample.

    Args:
        question_id (int): ID of the question
        question_text (str): Original question
        image_path (str or None): Path to the image file (can be None)
        entities (List[str]): Extracted entity strings

    Returns:
        nx.MultiDiGraph: Heterogeneous MMKG graph
    """
    G = nx.MultiDiGraph()

    # Add question node
    qid = f"question_{question_id}"
    G.add_node(qid, type="text", text=question_text)

    # Add image node (if exists)
    if image_path and Path(image_path).exists():
        img_id = f"image_{question_id}"
        G.add_node(img_id, type="image", path=image_path)
        G.add_edge(qid, img_id, relation="illustrated_by")

    # Add entity nodes
    for idx, ent in enumerate(entities):
        ent_id = f"entity_{question_id}_{idx}"
        G.add_node(ent_id, type="entity", label=ent)
        G.add_edge(qid, ent_id, relation="mentions")

    return G


# if __name__ == "__main__":
#     question = "Why do rabbits eat their poop?"
#     image_path = "data/scienceqa/images/example.jpg"
#     entities = ["rabbits", "poop"]
#     G = build_mmkg(1234, question, image_path, entities)
#     print(f"Nodes: {G.nodes(data=True)}")
#     print(f"Edges: {list(G.edges(data=True))}")