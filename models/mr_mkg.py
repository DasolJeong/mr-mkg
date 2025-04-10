import torch
import torch.nn as nn
from utils.graph_utils import convert_nx_to_dgl
from utils.visual_feature import get_node_initial_embeddings


class MR_MKG_Model(nn.Module):
    def __init__(self, llm, language_encoder, visual_adapter, knowledge_adapter, rgat_encoder):
        super().__init__()
        self.llm = llm  # frozen LLM (e.g., Flan-T5)
        self.language_encoder = language_encoder
        self.visual_adapter = visual_adapter
        self.knowledge_adapter = knowledge_adapter
        self.rgat_encoder = rgat_encoder

    def forward(self, input_ids, attention_mask, graph, image_embedding=None, labels=None):
        """
        Args:
            input_ids: Tensor of shape [B, L]
            attention_mask: Tensor of shape [B, L]
            graph: NetworkX MultiDiGraph
            image_embedding: Tensor of shape [B, 768] or None
            labels: Tensor of shape [B, L] or None

        Returns:
            Seq2SeqLMOutput from HuggingFace LLM
        """
        device = input_ids.device

        # 1. Text Encoding
        text_embed = self.language_encoder(input_ids)  # [B, L, 768]

        # 2. KG Encoding
        dgl_graph, node2id, _, rel_types = convert_nx_to_dgl(graph)
        node_features = get_node_initial_embeddings(graph, node2id).to(device)
        dgl_graph = dgl_graph.to(device)
        rel_types = rel_types.to(device)

        kg_embed = self.rgat_encoder(dgl_graph, node_features, rel_types)  # [N, 768]
        kg_embed = kg_embed.mean(dim=0, keepdim=True).expand(text_embed.size())  # [B, L, 768]
        kg_aligned = self.knowledge_adapter(kg_embed, text_embed)

        # 3. Image Encoding
        if image_embedding is not None:
            image_embedding = image_embedding.to(device).unsqueeze(1)  # [B, 1, 768]
            img_aligned = self.visual_adapter(image_embedding, text_embed)
        else:
            img_aligned = torch.zeros_like(text_embed)

        # 4. Combine modal embeddings
        prompt_embed = text_embed + kg_aligned + img_aligned  # [B, L, 768]

        # 5. LLM forward
        output = self.llm(
            inputs_embeds=prompt_embed,
            attention_mask=attention_mask,
            labels=labels
        )
        return output

