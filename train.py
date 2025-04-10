import os
import torch
from torch.utils.data import DataLoader
from transformers import T5ForConditionalGeneration, AutoTokenizer

from validate import validation
from data.mrmkg_dataset import MMKGDataset
from models.mr_mkg import MR_MKG_Model
from models.encoders import LanguageEncoder, KGEncoderRGAT
from models.adapters import VisualAdapter, KnowledgeAdapter
from models.cross_modal_align import CrossModalAlignLoss
from utils.graph_utils import convert_nx_to_dgl
from utils.visual_feature import get_node_initial_embeddings
from utils.align_utils import compute_image_entity_alignment_loss


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def custom_collate(batch):
    collated = {}
    for key in batch[0]:
        if key in ["graph", "prompt", "answer"]:  
            collated[key] = [item[key] for item in batch]

        elif key == "image_embedding":
            if any(item[key] is not None for item in batch):
                collated[key] = torch.stack([
                    item[key] if item[key] is not None
                    else torch.zeros_like(batch[0]["input_ids"][0], dtype=torch.float)
                    for item in batch
                ])
            else:
                collated[key] = None

        else:
            collated[key] = torch.stack([item[key] for item in batch])

    return collated



def main():
    # 1. Load dataset
    train_dataset = MMKGDataset("data/scienceqa/mmkg_graphs/train")
    val_dataset = MMKGDataset("data/scienceqa/mmkg_graphs/val")
    tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-base")

    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True, collate_fn=custom_collate)
    val_loader = DataLoader(val_dataset, batch_size=1, collate_fn=custom_collate)

    # 2. Build model
    llm = T5ForConditionalGeneration.from_pretrained("google/flan-t5-base").to(DEVICE)
    llm.eval()
    for param in llm.parameters():
        param.requires_grad = False

    model = MR_MKG_Model(
        llm=llm,
        language_encoder=LanguageEncoder("google/flan-t5-base").to(DEVICE),
        visual_adapter=VisualAdapter(512, 768).to(DEVICE),
        knowledge_adapter=KnowledgeAdapter(768, 768).to(DEVICE),
        rgat_encoder=KGEncoderRGAT(512, 512, 768, num_relations=10).to(DEVICE)
    ).to(DEVICE)

    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=5e-5
    )
    aligner = CrossModalAlignLoss(margin=0.2).to(DEVICE)
    align_loss_weight = 0.1

    # 3. Training loop
    model.train()
    for epoch in range(3):
        for batch in train_loader:
            input_ids = batch["input_ids"].to(DEVICE)
            attention_mask = batch["attention_mask"].to(DEVICE)
            graphs = batch["graph"]
            image_embeddings = batch["image_embedding"]

            for i, graph in enumerate(graphs):
                input_ids_i = input_ids[i].unsqueeze(0)
                attention_mask_i = attention_mask[i].unsqueeze(0)
                image_i = image_embeddings[i].unsqueeze(0) if image_embeddings is not None else None

                dgl_graph, node2id, _, rel_types = convert_nx_to_dgl(graph)
                if dgl_graph is None:
                    continue

                node_features = get_node_initial_embeddings(graph, node2id).to(DEVICE)
                rel_types = rel_types.to(DEVICE)
                dgl_graph = dgl_graph.to(DEVICE)

                output = model(
                    input_ids=input_ids_i,
                    attention_mask=attention_mask_i,
                    graph=graph,
                    image_embedding=image_i,
                    labels=input_ids_i
                )
                loss = output.loss

                align_loss = compute_image_entity_alignment_loss(
                    graph=graph,
                    image_embedding=image_i.squeeze(0) if image_i is not None else None,
                    dataset=train_dataset,
                    device=DEVICE,
                    aligner=aligner,
                    align_loss_weight=align_loss_weight
                )

                total_loss = loss + align_loss

                total_loss.backward()
                optimizer.step()
                optimizer.zero_grad()

                print(f"Loss: {loss.item():.4f} | Align Loss: {align_loss.item():.4f}")

        # 4. Validation & Checkpoint
        validation(model, val_loader, tokenizer, DEVICE)
        os.makedirs("checkpoints", exist_ok=True)
        save_path = f"checkpoints/mrmkg_epoch{epoch + 1}.pt"
        torch.save(model.state_dict(), save_path)
        print(f"[Checkpoint saved] {save_path}")


if __name__ == "__main__":
    main()
