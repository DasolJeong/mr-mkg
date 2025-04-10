import torch
import torch.nn as nn

class VisualAdapter(nn.Module):
    def __init__(self, input_dim=512, output_dim=768):
        super().__init__()
        self.projection = nn.Linear(input_dim, output_dim)
        self.attention = nn.MultiheadAttention(embed_dim=output_dim, num_heads=1, batch_first=True)

    def forward(self, image_features, text_embeddings):
        """
        Args:
            image_features: Tensor of shape [B, 512] or [1, 512] – from CLIP
            text_embeddings: Tensor of shape [B, L, 768] – token-level embeddings from LLM

        Returns:
            Tensor of shape [B, L, 768] – cross-modally aligned output
        """
        if image_features.dim() == 2:
            image_features = image_features.unsqueeze(1)  # [B, 1, 512]

        projected_features = self.projection(image_features)  # [B, 1, 768]
        aligned_output, _ = self.attention(
            query=text_embeddings,
            key=projected_features,
            value=projected_features
        )
        return aligned_output


class KnowledgeAdapter(nn.Module):
    def __init__(self, input_dim=512, output_dim=768):
        super().__init__()
        self.projection = nn.Linear(input_dim, output_dim)
        self.attention = nn.MultiheadAttention(embed_dim=output_dim, num_heads=1, batch_first=True)

    def forward(self, kg_features, text_embeddings):
        """
        Args:
            kg_features: Tensor of shape [B, K, 768] or [1, 768] – from KG encoder
            text_embeddings: Tensor of shape [B, L, 768] – token-level embeddings from LLM

        Returns:
            Tensor of shape [B, L, 768] – cross-modally aligned output
        """
        if kg_features.dim() == 2:
            kg_features = kg_features.unsqueeze(1)  # [B, 1, 768]

        projected_features = self.projection(kg_features)
        aligned_output, _ = self.attention(
            query=text_embeddings,
            key=projected_features,
            value=projected_features
        )
        return aligned_output
