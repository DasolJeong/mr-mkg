import os
import glob
import torch
import networkx as nx
from PIL import Image
from pathlib import Path
from torch.utils.data import Dataset
from transformers import AutoTokenizer, CLIPProcessor, CLIPModel

from models.adapters import VisualAdapter
from utils.graph_utils import extract_mrmkg_subgraph


SPECIAL_TOKENS = ["[IMAGE]", "[KNOWLEDGE]"]


class MMKGDataset(Dataset):
    def __init__(self, graph_dir: str, tokenizer_name="google/flan-t5-base", max_length=512, use_clip=True):
        self.graph_files = sorted(glob.glob(os.path.join(graph_dir, "*.pt")))
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.tokenizer.add_special_tokens({'additional_special_tokens': SPECIAL_TOKENS})
        self.max_length = max_length
        self.use_clip = use_clip

        if self.use_clip:
            self.visual_adapter = VisualAdapter(input_dim=512, output_dim=768)
            self.clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
            self.clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

    def __len__(self):
        return len(self.graph_files)

    def __getitem__(self, idx):
        graph_path = self.graph_files[idx]
        full_graph = torch.load(graph_path, weights_only=False)

        graph = extract_mrmkg_subgraph(full_graph)

        prompt = graph_to_prompt(graph)
        answer = self.get_answer_from_graph(graph)

        input_ids, attention_mask = encode_prompt(prompt, answer, self.tokenizer, self.max_length)

        image_embedding = None
        if self.use_clip:
            image_embedding = self.extract_image_embedding(graph)

        return {
            "input_ids": input_ids.squeeze(0),
            "attention_mask": attention_mask.squeeze(0),
            "graph": graph,
            "prompt": prompt,
            "image_embedding": image_embedding,
            "answer": answer
        }

    def get_answer_from_graph(self, graph):
        for node, data in graph.nodes(data=True):
            if data.get("type") == "answer":
                return data.get("text", "")
        return "[ANSWER]"

    def extract_image_embedding(self, graph):
        for node, data in graph.nodes(data=True):
            if data.get("type") == "image" and data.get("path") and os.path.exists(data["path"]):
                image = Image.open(data["path"]).convert("RGB")
                inputs = self.clip_processor(images=image, return_tensors="pt")
                with torch.no_grad():
                    outputs = self.clip_model.get_image_features(**inputs)
                return outputs.squeeze(0)
        return None


def graph_to_prompt(graph: nx.MultiDiGraph, use_image_token=True, include_knowledge_token=True) -> str:
    """
    Generate prompt string from graph.

    Returns:
        A string prompt including question, entities, and special tokens.
    """
    question_node = next((n for n, d in graph.nodes(data=True) if d.get("type") == "text"), None)
    if not question_node:
        raise ValueError("No question node found in graph")

    question_text = graph.nodes[question_node].get("text", "")
    entity_labels = [d["label"] for _, d in graph.nodes(data=True) if d.get("type") == "entity"]

    prompt = f"Question: {question_text}\n"
    if entity_labels:
        prompt += f"Entities: {', '.join(entity_labels)}\n"
    if include_knowledge_token:
        prompt += "Knowledge: [KNOWLEDGE]\n"
    if use_image_token and any(d.get("type") == "image" for _, d in graph.nodes(data=True)):
        prompt += "Image: [IMAGE]\n"
    prompt += "Answer:"
    return prompt


def encode_prompt(prompt: str, answer: str, tokenizer, max_length=512):
    """
    Tokenize prompt + answer using HuggingFace tokenizer.
    """
    full_input = prompt + " " + answer
    encoded = tokenizer(
        full_input,
        truncation=True,
        max_length=max_length,
        padding="max_length",
        return_tensors="pt"
    )
    return encoded["input_ids"], encoded["attention_mask"]
