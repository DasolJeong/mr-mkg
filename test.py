# test.py

import torch
from transformers import T5ForConditionalGeneration, AutoTokenizer
from data.mrmkg_dataset import MMKGDataset
from models.mr_mkg import MR_MKG_Model
from models.encoders import LanguageEncoder, KGEncoderRGAT
from models.adapters import VisualAdapter, KnowledgeAdapter
from utils.graph_utils import convert_nx_to_dgl
from utils.visual_feature import get_node_initial_embeddings


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def run_inference(model, sample, tokenizer, device, max_new_tokens=32):
    model.eval()

    input_ids = sample["input_ids"].unsqueeze(0).to(device)
    attention_mask = sample["attention_mask"].unsqueeze(0).to(device)
    graph = sample["graph"]
    image_embedding = sample["image_embedding"]
    if image_embedding is not None:
        image_embedding = image_embedding.unsqueeze(0).to(device)

    dgl_graph, node2id, rel2id, rel_types = convert_nx_to_dgl(graph)
    if dgl_graph is None:
        print("Empty graph. Skipping inference.")
        return None

    node_feat = get_node_initial_embeddings(graph, node2id).to(device)
    rel_types = rel_types.to(device)
    dgl_graph = dgl_graph.to(device)

    with torch.no_grad():
        text_feat = model.language_encoder(input_ids)  # [1, L, 768]

        kg_embed = model.rgat_encoder(dgl_graph, node_feat, rel_types)
        kg_embed = kg_embed.mean(dim=0, keepdim=True).expand(text_feat.size())  # [1, L, 768]
        kg_aligned = model.knowledge_adapter(kg_embed, text_feat)

        if image_embedding is not None:
            img_aligned = model.visual_adapter(image_embedding, text_feat)
        else:
            img_aligned = torch.zeros_like(text_feat)

        prompt_embed = text_feat + kg_aligned + img_aligned

        output_ids = model.llm.generate(
            inputs_embeds=prompt_embed,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            num_beams=4,
            early_stopping=True
        )

        output_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
        return output_text


if __name__ == "__main__":
    print("[✓] Loading tokenizer and dataset...")
    tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-base")
    test_dataset = MMKGDataset("data/scienceqa/mmkg_graphs/test")

    print("[✓] Loading model...")
    llm = T5ForConditionalGeneration.from_pretrained("google/flan-t5-base").to(DEVICE)
    for param in llm.parameters():
        param.requires_grad = False

    model = MR_MKG_Model(
        llm=llm,
        language_encoder=LanguageEncoder("google/flan-t5-base").to(DEVICE),
        visual_adapter=VisualAdapter(input_dim=512, output_dim=768).to(DEVICE),
        knowledge_adapter=KnowledgeAdapter(input_dim=768, output_dim=768).to(DEVICE),
        rgat_encoder=KGEncoderRGAT(in_dim=512, hidden_dim=512, out_dim=768, num_rels=10).to(DEVICE)
    ).to(DEVICE)

    # Optional: load trained checkpoint
    # model.load_state_dict(torch.load("checkpoints/mrmkg_best.pt"))

    print("[✓] Running inference on 5 test samples...\n")
    for i in range(5):
        sample = test_dataset[i+100]
        output = run_inference(model, sample, tokenizer, DEVICE)
        
        gt_answer = sample.get("label", "").strip()
        
        print(f"[Sample {i+1}]")
        print("Prompt:\n", sample["prompt"])
        print("Generated Answer:\n", output)
        print("Ground Truth Answer:\n", gt_answer)
        print("="*60)
    

