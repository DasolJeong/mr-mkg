import torch
from transformers import T5ForConditionalGeneration, AutoTokenizer
from PIL import Image
from models.mr_mkg import MR_MKG_Model
from models.encoders import LanguageEncoder, KGEncoderRGAT
from models.adapters import VisualAdapter, KnowledgeAdapter
from utils.graph_utils import convert_nx_to_dgl
from utils.visual_feature import get_node_initial_embeddings, get_image_embedding
import networkx as nx

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def generate_answer_from_question(model, question, tokenizer, device, image_path=None, max_new_tokens=32):
    model.eval()

    # Step 1: Prepare prompt for model
    prompt = f"Question: {question}\nImage: [IMAGE]\nAnswer:"
    input_ids, attention_mask = encode_prompt(prompt, "[ANSWER]", tokenizer)

    input_ids = input_ids.to(device)
    attention_mask = attention_mask.to(device)

    # Step 2: Generate answer using model
    with torch.no_grad():
        # Step 2.1: 임의로 사용할 dummy 그래프 생성 (MultiDiGraph 형식이 아니므로, 추후 확장 필요)
        graph = {}  # 빈 딕셔너리 대신 실제 그래프가 필요할 경우 dummy 그래프 생성

        # ** 실제 빈 그래프 대신 샘플로 사용할 작은 그래프 생성 **
        G_dummy = nx.MultiDiGraph()
        G_dummy.add_node(0, type="text", text="What is the capital of France?")
        G_dummy.add_node(1, type="entity", label="France")
        G_dummy.add_edge(0, 1, relation="mentions")
        graph = G_dummy  # 생성된 dummy 그래프 사용

        # DGL 그래프 변환
        dgl_graph, node2id, rel2id, rel_types = convert_nx_to_dgl(graph)
        if dgl_graph is None:
            print("Empty graph. Skipping inference.")
            return None
        node_feat = get_node_initial_embeddings(graph, node2id).to(device)
        rel_types = rel_types.to(device)
        dgl_graph = dgl_graph.to(device)

        # Step 2.2: Generate embeddings using the model
        text_feat = model.language_encoder(input_ids)  # [1, L, 768]
        kg_embed = model.rgat_encoder(dgl_graph, node_feat, rel_types)
        kg_embed = kg_embed.mean(dim=0, keepdim=True).expand(text_feat.size())  # [1, L, 768]
        kg_aligned = model.knowledge_adapter(kg_embed, text_feat)

        img_aligned = torch.zeros_like(text_feat)  # Dummy image embedding (if necessary)

        if image_path is not None:
            # Step 2.3: 이미지 임베딩을 추가 (이미지 경로가 있을 경우)
            # image = Image.open(image_path).convert("RGB")
            with torch.no_grad():
                img_feature = get_image_embedding(image_path)  # [1, 768]
                img_feature = img_feature.unsqueeze(0).to(device)
            img_embedding = model.visual_adapter(img_feature, text_feat).to(device)
            img_aligned = img_embedding

        # Combine all embeddings
        prompt_embed = text_feat + kg_aligned + img_aligned  # [1, L, 768]

        # Step 3: LLM forward
        output_ids = model.llm.generate(
            inputs_embeds=prompt_embed,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            num_beams=4,
            early_stopping=True
        )

        output_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
        return output_text


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


if __name__ == "__main__":
    # Load model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-base")
    llm = T5ForConditionalGeneration.from_pretrained("google/flan-t5-base").to(DEVICE)
    for param in llm.parameters():
        param.requires_grad = False  # freeze LLM

    model = MR_MKG_Model(
        llm=llm,
        language_encoder=LanguageEncoder("google/flan-t5-base").to(DEVICE),
        visual_adapter=VisualAdapter(input_dim=512, output_dim=768).to(DEVICE),
        knowledge_adapter=KnowledgeAdapter(input_dim=768, output_dim=768).to(DEVICE),
        rgat_encoder=KGEncoderRGAT(in_dim=512, hidden_dim=512, out_dim=768, num_rels=10).to(DEVICE)
    ).to(DEVICE)

    # Optional: load trained checkpoint
    # model.load_state_dict(torch.load("checkpoints/mrmkg_best.pt"))

    # Allow user to input a question
    question = input("Enter a question: ")
    image_path = input("Enter an image path (or leave empty for no image): ")

    if image_path == "":
        image_path = None  # If no image, set to None

    answer = generate_answer_from_question(model, question, tokenizer, DEVICE, image_path)

    print(f"Question: {question}")
    print(f"Generated Answer: {answer}")
