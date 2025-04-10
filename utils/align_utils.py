import torch
import random
from models.cross_modal_align import CrossModalAlignLoss
from utils.visual_feature import get_text_embedding


def compute_image_entity_alignment_loss(
    graph,
    image_embedding,
    dataset,
    device,
    aligner=None,
    align_loss_weight=0.1
):
    """
    Compute triplet alignment loss between image and entity text embeddings.

    Args:
        graph (nx.MultiDiGraph): current MMKG sample graph
        image_embedding (Tensor): [512]
        dataset (Dataset): full MMKG dataset (for negative sampling)
        device (torch.device): device for computation
        aligner (CrossModalAlignLoss or None): optional triplet loss module
        align_loss_weight (float): weight for alignment loss

    Returns:
        Tensor: weighted alignment loss (scalar)
    """
    if image_embedding is None:
        return torch.tensor(0.0, device=device)

    if aligner is None:
        aligner = CrossModalAlignLoss(margin=0.2).to(device)

    anchor = image_embedding.to(device)  

    # Positive: mean of entity embeddings from current graph
    entity_labels = [d["label"] for _, d in graph.nodes(data=True) if d.get("type") == "entity"]
    if not entity_labels:
        return torch.tensor(0.0, device=device)

    positive_embeddings = [get_text_embedding(label).to(device) for label in entity_labels]
    positive = torch.stack(positive_embeddings).mean(dim=0)  # [768]

    # Negative: randomly sampled entity from another graph
    for _ in range(3):  # try up to 3 times to get valid negative
        negative_sample = random.choice(dataset)
        neg_graph = negative_sample["graph"]
        neg_entities = [d["label"] for _, d in neg_graph.nodes(data=True) if d.get("type") == "entity"]
        if neg_entities:
            neg_label = random.choice(neg_entities)
            negative = get_text_embedding(neg_label).to(device)
            break
    else:
        return torch.tensor(0.0, device=device)

    triplet_loss = aligner(
        anchor.unsqueeze(0),
        positive.unsqueeze(0),
        negative.unsqueeze(0)
    )

    return align_loss_weight * triplet_loss
