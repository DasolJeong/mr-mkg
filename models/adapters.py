import torch
import torch.nn as nn
import math

class VisualAdapter(nn.Module):
    def __init__(self, input_dim=512, output_dim=768):
        super().__init__()
        self.proj = nn.Linear(input_dim, output_dim)
        self.attn = nn.MultiheadAttention(embed_dim=output_dim, num_heads=1, batch_first=True)

    def forward(self, image_feat, text_feat):
        """
        Args:
            image_feat: [1, 768] or [B, 768] – extracted from CLIP
            text_feat: [B, L, 768] – token-level LLM input embedding
        Returns:
            aligned: [B, L, 768]
        """
        if image_feat.dim() == 2:
            image_feat = image_feat.unsqueeze(1)  # [B, 1, 768]
        x = self.proj(image_feat)                # [B, 1, 768]
        out, _ = self.attn(query=text_feat, key=x, value=x)  # Cross-attention
        return out
    
    
class KnowledgeAdapter(nn.Module):
    def __init__(self, input_dim=512, output_dim=768):
        super().__init__()
        self.proj = nn.Linear(input_dim, output_dim)
        self.attn = nn.MultiheadAttention(embed_dim=output_dim, num_heads=1, batch_first=True)

    def forward(self, kg_feat, text_feat):
        """
        Args:
            kg_feat: [B, K, 768] or [1, 768]
            text_feat: [B, L, 768]
        Returns:
            aligned: [B, L, 768]
        """
        if kg_feat.dim() == 2:
            kg_feat = kg_feat.unsqueeze(1)  # [B, 1, 768]
        x = self.proj(kg_feat)
        out, _ = self.attn(query=text_feat, key=x, value=x)
        return out