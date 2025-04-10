import torch
import random
from models.cross_modal_align import CrossModalAlignLoss
from utils.visual_feature import get_text_embedding

def compute_image_entity_alignment_loss(graph, image_embedding, dataset, device, aligner=None, align_loss_weight=0.1):
    """
    Compute triplet alignment loss between image and entity text.

    Args:
        graph (nx.MultiDiGraph): current MMKG sample
        image_embedding (Tensor): [768]
        dataset (Dataset): MMKGDataset instance
        device (torch.device)
        aligner (CrossModalAligner): alignment module
        align_loss_weight (float)

    Returns:
        align_loss: weighted triplet loss (float Tensor)
    """
    if image_embedding is None:
        return torch.tensor(0.0, device=device)

    if aligner is None:
        aligner = CrossModalAlignLoss(margin=0.2).to(device)

    anchor = image_embedding.to(device)  # [768]
    

    # Positive entity embedding (mean of all entity labels)
    entity_labels = [d["label"] for _, d in graph.nodes(data=True) if d.get("type") == "entity"]
    if not entity_labels:
        return torch.tensor(0.0, device=device)

    pos_embeddings = [get_text_embedding(lbl).to(device) for lbl in entity_labels]
    positive = torch.stack(pos_embeddings).mean(dim=0)  # [768]

    # Negative entity from a random different sample
    for _ in range(3):  # 최대 3번 시도
        neg_sample = random.choice(dataset)
        neg_graph = neg_sample["graph"]
        neg_entities = [d["label"] for _, d in neg_graph.nodes(data=True) if d.get("type") == "entity"]
        if neg_entities:
            neg_text = random.choice(neg_entities)
            negative = get_text_embedding(neg_text).to(device)
            break
    else:
        return torch.tensor(0.0, device=device)

    # Compute triplet loss
    triplet_loss = aligner(
            anchor,
            positive.unsqueeze(0),
            negative.unsqueeze(0)
        )
    return align_loss_weight * triplet_loss
