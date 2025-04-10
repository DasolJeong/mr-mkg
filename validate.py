import torch
from torch.utils.data import DataLoader
from data.mrmkg_dataset import MMKGDataset
from utils.graph_utils import convert_nx_to_dgl
from utils.visual_feature import get_node_initial_embeddings
from utils.align_utils import compute_image_entity_alignment_loss

def validation(model, val_dataloader, tokenizer, device):
    model.eval()
    total_loss = 0
    total_samples = 0

    with torch.no_grad():
        for batch in val_dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            graph = batch["graph"][0]  # batch size = 1 가정
            image_embedding = batch["image_embedding"]
            if image_embedding is not None:
                image_embedding = image_embedding.to(device)

            labels = input_ids  # self-supervised

            # DGL 변환용 입력 준비
            dgl_graph, node2id, rel2id, rel_types = convert_nx_to_dgl(graph)
            if dgl_graph is None:
                continue
            node_feat = get_node_initial_embeddings(graph, node2id).to(device)
            rel_types = rel_types.to(device)
            dgl_graph = dgl_graph.to(device)

            output = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                graph=graph,
                image_embedding=image_embedding,
                labels=labels,
            )

            total_loss += output.loss.item()
            total_samples += 1

    avg_loss = total_loss / max(total_samples, 1)
    print(f"[Validation] Loss: {avg_loss:.4f}")
    return avg_loss