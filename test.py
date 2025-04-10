import torch
from transformers import T5ForConditionalGeneration, AutoTokenizer

from data.mrmkg_dataset import MMKGDataset
from models.mr_mkg import MR_MKG_Model
from models.encoders import LanguageEncoder, KGEncoderRGAT
from models.adapters import VisualAdapter, KnowledgeAdapter
from utils.graph_utils import convert_nx_to_dgl
from utils.visual_feature import get_node_initial_embeddings


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


@torch.no_grad()
def run_inference(model, sample, tokenizer, device, max_new_tokens=32):
    model.eval()

    input_ids = sample["input_ids"].unsqueeze(0).to(device)
    attention_mask = sample["attention_mask"].unsqueeze(0).to(device)
    graph = sample["graph"]
    image_embedding = sample["image_embedding"]
    if image_embedding is not None:
        image_embedding = image_embedding.unsqueeze(0).to(device)

    dgl_graph, node2id, _, rel_types = convert_nx_to_dgl(graph)
    if dgl_graph is None:
        print("[Warning] Skipping empty graph.")
        return None

    node_features = get_node_initial_embeddings(graph, node2id).to(device)
    rel_types = rel_types.to(device)
    dgl_graph = dgl_graph.to(device)

    text_embed = model.language_encoder(input_ids)
    kg_embed = model.rgat_encoder(dgl_graph, node_features, rel_types).mean(dim=0, keepdim=True)
    kg_embed = kg_embed.expand(text_embed.size())
    kg_aligned = model.knowledge_adapter(kg_embed, text_embed)

    img_aligned = model.visual_adapter(image_embedding, text_embed) if image_embedding is not None else torch.zeros_like(text_embed)
    prompt_embed = text_embed + kg_aligned + img_aligned

    output_ids = model.llm.generate(
        inputs_embeds=prompt_embed,
        attention_mask=attention_mask,
        max_new_tokens=max_new_tokens,
        num_beams=4,
        early_stopping=True
    )
    return tokenizer.decode(output_ids[0], skip_special_tokens=True)


if __name__ == "__main__":
    print("[✓] Loading tokenizer and test dataset...")
    tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-base")
    test_dataset = MMKGDataset("data/scienceqa/mmkg_graphs/test")

    print("[✓] Initializing model...")
    llm = T5ForConditionalGeneration.from_pretrained("google/flan-t5-base").to(DEVICE)
    for param in llm.parameters():
        param.requires_grad = False

    model = MR_MKG_Model(
        llm=llm,
        language_encoder=LanguageEncoder("google/flan-t5-base").to(DEVICE),
        visual_adapter=VisualAdapter(512, 768).to(DEVICE),
        knowledge_adapter=KnowledgeAdapter(768, 768).to(DEVICE),
        rgat_encoder=KGEncoderRGAT(512, 512, 768, num_relations=10).to(DEVICE)
    ).to(DEVICE)

    model.load_state_dict(torch.load("checkpoints/mrmkg_epoch3.pt", map_location=DEVICE))

    print("[✓] Running test inference...\n")

    start_idx = 100
    num_samples = 5

    for i in range(num_samples):
        sample = test_dataset[start_idx + i]
        output = run_inference(model, sample, tokenizer, DEVICE)

        ground_truth = sample.get("answer", "").strip()
        
        print(f"[Sample {i+1}]")
        print("Prompt:\n", sample["prompt"])
        print("Generated Answer:\n", output)
        print("="*60)
    

