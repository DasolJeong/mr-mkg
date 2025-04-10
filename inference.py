import os
import torch
import networkx as nx
from PIL import Image
from transformers import T5ForConditionalGeneration, AutoTokenizer

from models.mr_mkg import MR_MKG_Model
from models.encoders import LanguageEncoder, KGEncoderRGAT
from models.adapters import VisualAdapter, KnowledgeAdapter
from utils.graph_utils import convert_nx_to_dgl
from utils.visual_feature import get_node_initial_embeddings, get_image_embedding

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def encode_prompt(prompt: str, answer: str, tokenizer, max_length=512):
    full_input = prompt + " " + answer
    encoded = tokenizer(
        full_input,
        truncation=True,
        max_length=max_length,
        padding="max_length",
        return_tensors="pt"
    )
    return encoded["input_ids"], encoded["attention_mask"]


@torch.no_grad()
def generate_answer_from_question(model, question: str, tokenizer, device, image_path: str = None, max_new_tokens=32):
    """
    Generate an answer from a question and optional image using MR-MKG model.
    """
    model.eval()

    # 1. Prepare prompt
    prompt = f"Question: {question}\nImage: [IMAGE]\nAnswer:"
    input_ids, attention_mask = encode_prompt(prompt, "[ANSWER]", tokenizer)
    input_ids, attention_mask = input_ids.to(device), attention_mask.to(device)

    # 2. Create a dummy MMKG graph
    graph = nx.MultiDiGraph()
    graph.add_node(0, type="text", text=question)
    graph.add_node(1, type="entity", label="unknown")  # dummy entity
    graph.add_edge(0, 1, relation="mentions")

    # 3. Convert to DGL
    dgl_graph, node2id, _, rel_types = convert_nx_to_dgl(graph)
    if dgl_graph is None:
        print("[Error] Graph conversion failed.")
        return "Unable to answer"

    node_feat = get_node_initial_embeddings(graph, node2id).to(device)
    rel_types = rel_types.to(device)
    dgl_graph = dgl_graph.to(device)

    # 4. Embedding: text + KG + image
    text_embed = model.language_encoder(input_ids)  # [1, L, 768]
    kg_embed = model.rgat_encoder(dgl_graph, node_feat, rel_types).mean(dim=0, keepdim=True)
    kg_embed = kg_embed.expand(text_embed.size())
    kg_aligned = model.knowledge_adapter(kg_embed, text_embed)

    # 5. Optional image embedding
    if image_path and os.path.exists(image_path):
        img_feature = get_image_embedding(image_path).unsqueeze(0).to(device)  # [1, 1, 512]
        img_aligned = model.visual_adapter(img_feature, text_embed)
    else:
        img_aligned = torch.zeros_like(text_embed)

    # 6. Combine embeddings
    prompt_embed = text_embed + kg_aligned + img_aligned

    # 7. Generate answer
    output_ids = model.llm.generate(
        inputs_embeds=prompt_embed,
        attention_mask=attention_mask,
        max_new_tokens=max_new_tokens,
        num_beams=4,
        early_stopping=True
    )
    return tokenizer.decode(output_ids[0], skip_special_tokens=True)


if __name__ == "__main__":
    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-base")
    llm = T5ForConditionalGeneration.from_pretrained("google/flan-t5-base").to(DEVICE)
    for param in llm.parameters():
        param.requires_grad = False

    model = MR_MKG_Model(
        llm=llm,
        language_encoder=LanguageEncoder("google/flan-t5-base").to(DEVICE),
        visual_adapter=VisualAdapter(512, 768).to(DEVICE),
        knowledge_adapter=KnowledgeAdapter(768, 768).to(DEVICE),
        rgat_encoder=KGEncoderRGAT(512, 512, 768, num_rels=10).to(DEVICE)
    ).to(DEVICE)

    # Run interactive QA
    question = input("Enter a question: ")
    image_path = input("Enter image path (or leave blank): ")
    image_path = image_path.strip() or None

    answer = generate_answer_from_question(model, question, tokenizer, DEVICE, image_path)

    print(f"Question: {question}")
    print(f"Answer: {answer}")
