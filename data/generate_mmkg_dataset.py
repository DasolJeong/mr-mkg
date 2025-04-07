
"""
Torch Dataset for loading MMKG graphs from ScienceQA splits.
Includes:
- Graph-to-prompt conversion
- Tokenization for FLAN-T5
- Optional CLIP embedding + VisualAdapter projection
"""

import torch
from torch.utils.data import Dataset
from pathlib import Path
import networkx as nx
import os
import glob
from transformers import AutoTokenizer
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
from models.visual_adapter import VisualAdapter

class MMKGDataset(Dataset):
    def __init__(self, graph_dir: str, tokenizer_name="google/flan-t5-base", max_length=512, use_clip=True):
        self.graph_files = sorted(glob.glob(os.path.join(graph_dir, '*.pt')))
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.max_length = max_length
        self.use_clip = use_clip

        if self.use_clip:
            self.clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
            self.clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
            self.visual_adapter = VisualAdapter(input_dim=512, output_dim=self.tokenizer.model_max_length)

    def __len__(self):
        return len(self.graph_files)

    def __getitem__(self, idx):
        graph_path = self.graph_files[idx]
        G = torch.load(graph_path)

        prompt = graph_to_prompt(G)
        answer = "[ANSWER]"

        input_ids, attention_mask = encode_prompt(prompt, answer, self.tokenizer, self.max_length)

        image_tensor = None
        if self.use_clip:
            image_tensor = self.extract_image_embedding(G)
            if image_tensor is not None:
                image_tensor = self.visual_adapter(image_tensor)

        return {
            "input_ids": input_ids.squeeze(0),
            "attention_mask": attention_mask.squeeze(0),
            "graph": G,
            "prompt": prompt,
            "image_embedding": image_tensor
        }

    def extract_image_embedding(self, G):
        for _, data in G.nodes(data=True):
            if data.get("type") == "image" and data.get("path") and os.path.exists(data["path"]):
                image = Image.open(data["path"]).convert("RGB")
                inputs = self.clip_processor(images=image, return_tensors="pt")
                with torch.no_grad():
                    features = self.clip_model.get_image_features(**inputs)
                return features.squeeze(0)
        return None


def graph_to_prompt(G: nx.MultiDiGraph, use_image_token=True) -> str:
    q_node = next((n for n, d in G.nodes(data=True) if d.get("type") == "text"), None)
    if not q_node:
        raise ValueError("No question node found in graph")
    question_text = G.nodes[q_node].get("text", "")

    entity_labels = [d["label"] for _, d in G.nodes(data=True) if d.get("type") == "entity"]

    prompt = f"Question: {question_text}\n"
    if entity_labels:
        prompt += f"Entities: {', '.join(entity_labels)}\n"
    if use_image_token and any(d.get("type") == "image" for _, d in G.nodes(data=True)):
        prompt += "Image: [IMAGE]\n"
    prompt += "Answer:"
    return prompt


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
    dataset = MMKGDataset("data/scienceqa/mmkg_graphs/train")
    print(f"Loaded {len(dataset)} MMKG graphs")
    sample = dataset[0]
    print("\nGenerated Prompt:\n", sample["prompt"])
    print("Token IDs:", sample["input_ids"][:10])
    print("Image Embedding shape:", sample["image_embedding"].shape if sample["image_embedding"] is not None else None)
