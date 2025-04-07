"""
Main MR-MKG model integrating visual adapter, KG encoder, and LLM.
"""

"""
Main MR-MKG model integrating visual adapter, KG encoder, and LLM.
"""

import torch
import torch.nn as nn
from transformers import T5ForConditionalGeneration
from models.visual_adapter import VisualAdapter
from models.knowledge_encoder import KnowledgeEncoder

class MrMKGModel(nn.Module):
    def __init__(self, visual_dim=512, llm_name="google/flan-t5-base"):
        super().__init__()
        self.llm = T5ForConditionalGeneration.from_pretrained(llm_name)
        self.llm_dim = self.llm.config.d_model

        self.visual_adapter = VisualAdapter(visual_dim, self.llm_dim)
        self.knowledge_encoder = KnowledgeEncoder(in_dim=self.llm_dim, hidden_dim=self.llm_dim)

    def forward(self, input_ids, attention_mask, image_embedding=None, labels=None, graph=None):
        # Step 1: Embed text input
        inputs_embeds = self.llm.encoder.embed_tokens(input_ids)

        # Step 2: Inject image embedding into [IMAGE] token position if available
        if image_embedding is not None:
            image_token_id = self.llm.tokenizer.convert_tokens_to_ids("[IMAGE]")
            image_positions = (input_ids == image_token_id).nonzero(as_tuple=True)
            for batch_idx, token_idx in zip(*image_positions):
                inputs_embeds[batch_idx, token_idx] = image_embedding[batch_idx]

        # Step 3: Pass through LLM
        outputs = self.llm(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            labels=labels
        )
        return outputs