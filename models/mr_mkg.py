import torch
import torch.nn as nn
from utils.graph_utils import convert_nx_to_dgl
from utils.visual_feature import get_node_initial_embeddings

class MR_MKG_Model(nn.Module):
    def __init__(self, 
                 llm, 
                 language_encoder, 
                 visual_adapter, 
                 knowledge_adapter, 
                 rgat_encoder):
        super().__init__()
        self.llm = llm  # frozen
        self.language_encoder = language_encoder
        self.visual_adapter = visual_adapter
        self.knowledge_adapter = knowledge_adapter
        self.rgat_encoder = rgat_encoder

    def forward(self, 
                input_ids, 
                attention_mask, 
                graph, 
                image_embedding=None, 
                labels=None):
        """
        Returns:
            Seq2SeqLMOutput with loss and logits
        """
        device = input_ids.device

        # 1. Text Embedding
        text_embed = self.language_encoder(input_ids)  # [B, L, 768]

        # 2. KG Embedding
        dgl_graph, node2id, rel2id, rel_types = convert_nx_to_dgl(graph)
        node_feat = get_node_initial_embeddings(graph, node2id).to(device)
        rel_types = rel_types.to(device)
        dgl_graph = dgl_graph.to(device)

        kg_embed = self.rgat_encoder(dgl_graph, node_feat, rel_types)  # [N, 768]
        kg_embed = kg_embed.mean(dim=0, keepdim=True).expand(text_embed.size())  # [B, L, 768]
        kg_aligned = self.knowledge_adapter(kg_embed, text_embed)  # [B, L, 768]

        # 3. Image Embedding
        if image_embedding is not None:
            image_embedding = image_embedding.to(device).unsqueeze(1)  # [B, 1, 768]
            img_aligned = self.visual_adapter(image_embedding, text_embed)  # [B, L, 768]
        else:
            img_aligned = torch.zeros_like(text_embed)

        # 4. Combine All
        prompt_embed = text_embed + kg_aligned + img_aligned  # [B, L, 768]

        # 5. LLM Forward using inputs_embeds (loss computed via logits)
        output = self.llm(
            inputs_embeds=prompt_embed,
            attention_mask=attention_mask,
            labels=labels
        )

        return output


def generate(self, input_ids, attention_mask, image_embedding=None, graph=None, dgl_graph=None, rel_types=None, node_feat=None, **kwargs):
    return self.llm.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,
        max_new_tokens=kwargs.get("max_new_tokens", 20)
    )