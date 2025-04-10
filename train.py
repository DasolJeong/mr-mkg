# train.py

import torch
from torch.utils.data import DataLoader
from transformers import T5ForConditionalGeneration
from transformers import AutoTokenizer
import random
from validate import validation
from data.mrmkg_dataset import MMKGDataset
from models.mr_mkg import MR_MKG_Model
from models.encoders import LanguageEncoder, KGEncoderRGAT
from models.adapters import VisualAdapter, KnowledgeAdapter
from models.cross_modal_align import CrossModalAlignLoss
from utils.visual_feature import get_node_initial_embeddings, get_image_embedding, get_text_embedding
from utils.graph_utils import convert_nx_to_dgl
from utils.align_utils import compute_image_entity_alignment_loss

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def custom_collate(batch):
    collated = {}

    for key in batch[0]:
        if key == "graph":
            # NetworkX 그래프는 리스트로 유지
            collated[key] = [item[key] for item in batch]
        elif key == "prompt":
            # 문자열도 리스트로 유지
            collated[key] = [item[key] for item in batch]
        elif key == "image_embedding":
            # image_embedding은 None이 있을 수 있으므로 처리 필요
            if any(item[key] is not None for item in batch):
                collated[key] = torch.stack([
                    item[key] if item[key] is not None else torch.zeros_like(batch[0]["input_ids"][0], dtype=torch.float)
                    for item in batch
                ])
            else:
                collated[key] = None
        else:
            # 기본적으로 텐서는 stack
            collated[key] = torch.stack([item[key] for item in batch])

    return collated



aligner = CrossModalAlignLoss(margin=0.2).to(DEVICE)
align_loss_weight = 0.1 

# 1. 데이터셋 불러오기
dataset = MMKGDataset("data/scienceqa/mmkg_graphs/train")
dataloader = DataLoader(dataset, batch_size=1, shuffle=True, collate_fn=custom_collate)
tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-base")
val_dataset = MMKGDataset("data/scienceqa/mmkg_graphs/val")
val_loader = DataLoader(val_dataset, batch_size=1, collate_fn=custom_collate)


# 2. 모델 구성
llm = T5ForConditionalGeneration.from_pretrained("google/flan-t5-base").to(DEVICE)
llm.eval()
for param in llm.parameters():
    param.requires_grad = False
language_encoder = LanguageEncoder("google/flan-t5-base").to(DEVICE)
visual_adapter = VisualAdapter(input_dim=512, output_dim=768).to(DEVICE)
knowledge_adapter = KnowledgeAdapter(input_dim=768, output_dim=768).to(DEVICE)
rgat_encoder = KGEncoderRGAT(in_dim=512, hidden_dim=512, out_dim=768, num_rels=10).to(DEVICE)

model = MR_MKG_Model(
    llm=llm,
    language_encoder=language_encoder,
    visual_adapter=visual_adapter,
    knowledge_adapter=knowledge_adapter,
    rgat_encoder=rgat_encoder
).to(DEVICE)

# 3. 옵티마이저 및 손실 설정
optimizer = torch.optim.Adam(
    filter(lambda p: p.requires_grad, model.parameters()),
    lr=5e-5
)

# 4. 학습 루프
model.train()

for epoch in range(3):
    for batch in dataloader:
        input_ids = batch["input_ids"].to(DEVICE)
        attention_mask = batch["attention_mask"].to(DEVICE)
        graph_list  = batch["graph"]
        image_embedding = batch["image_embedding"]
        if image_embedding is not None:
            image_embedding = image_embedding.to(DEVICE)
            
        for i in range(len(graph_list)):
            graph = graph_list[i]
            input_ids_i = input_ids[i].unsqueeze(0)
            attention_mask_i = attention_mask[i].unsqueeze(0)
            image_i = image_embedding[i].unsqueeze(0) if image_embedding is not None else None
            

            # Subgraph → DGL 변환 + 노드 임베딩
            dgl_graph, node2id, rel2id, rel_types = convert_nx_to_dgl(graph)
            if dgl_graph is None:
                continue
            node_feat = get_node_initial_embeddings(graph, node2id).to(DEVICE)
            rel_types = rel_types.to(DEVICE)

            # Forward + Loss
            output = model(
                input_ids=input_ids_i,
                attention_mask=attention_mask_i,
                graph=graph,
                image_embedding=image_i,
                labels=input_ids_i  
            )
        
            loss = output.loss
            
            # for name, param in model.named_parameters():
            #     if param.requires_grad:
            #         print(f"Trainable: {name} - shape: {param.shape}")
            
            
            align_loss = compute_image_entity_alignment_loss(
                graph=graph,
                image_embedding=image_embedding,
                dataset=dataset,
                device=DEVICE,
                aligner=aligner,
                align_loss_weight=0.1
            )

            total_loss = loss + align_loss

            
            total_loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            print(f"Loss: {loss.item():.4f} | Align Loss: {align_loss.item():.4f}")
            
    
    validation(model, val_loader, tokenizer, DEVICE)      
    save_path = f"checkpoints/mrmkg_epoch{epoch+1}.pt"
    torch.save(model.state_dict(), save_path)
    print(f"[Checkpoint saved] {save_path}")