import os
from PIL import Image
import torch
from transformers import CLIPProcessor, CLIPModel

# Load CLIP model once (shared across functions)
clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

@torch.no_grad()
def get_image_embedding(image_path: str) -> torch.Tensor:
    """
    Load image and return CLIP image feature.
    """
    image = Image.open(image_path).convert("RGB")
    inputs = clip_processor(images=image, return_tensors="pt")
    outputs = clip_model.get_image_features(**inputs)
    return outputs.squeeze(0)  # [512]

@torch.no_grad()
def get_text_embedding(text: str) -> torch.Tensor:
    """
    Use CLIP text encoder to extract embedding from a single string.
    """
    inputs = clip_processor(text=[text], return_tensors="pt", padding=True)
    outputs = clip_model.get_text_features(**inputs)
    return outputs.squeeze(0)  # [512]

def get_node_initial_embeddings(nx_g, node2id) -> torch.Tensor:
    """
    Generate initial node features for all nodes in MMKG using CLIP.
    
    Args:
        nx_g: networkx.MultiDiGraph
        node2id: dict mapping node names to index
    Returns:
        torch.Tensor of shape [num_nodes, 512]
    """
    num_nodes = len(node2id)
    node_feats = torch.zeros(num_nodes, 512)

    for node_name, idx in node2id.items():
        node_data = nx_g.nodes[node_name]
        node_type = node_data.get("type", "entity")

        try:
            if node_type == "image" and "path" in node_data and os.path.exists(node_data["path"]):
                node_feats[idx] = get_image_embedding(node_data["path"])

            elif node_type == "text" and "text" in node_data:
                node_feats[idx] = get_text_embedding(node_data["text"])

            elif node_type == "entity" and "label" in node_data:
                node_feats[idx] = get_text_embedding(node_data["label"])

        except Exception as e:
            print(f"[Warning] Skipping node {node_name} due to error: {e}")

    return node_feats  # [N, 512]