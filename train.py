"""
Training script for MR-MKG model on ScienceQA with visual input and knowledge encoding.
"""

import torch
from torch.utils.data import DataLoader
from transformers import AdamW
from models.mrmkg_model import MrMKGModel
from data.mrmkg_dataset import MMKGDataset
from utils.graph_utils import nx_to_pyg


def train():
    # Hyperparameters
    batch_size = 4
    epochs = 3
    lr = 5e-5
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load dataset
    dataset = MMKGDataset("data/scienceqa/mmkg_graphs/train")
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Initialize model
    model = MrMKGModel()
    model.to(device)

    # Optimizer
    optimizer = AdamW(model.parameters(), lr=lr)

    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            image_embedding = batch["image_embedding"]
            if image_embedding is not None:
                image_embedding = image_embedding.to(device)

            # Convert MMKG graph to PyG graph for KG encoder
            graph = batch["graph"]
            pyg_data = nx_to_pyg(graph)
            pyg_data = pyg_data.to(device)

            # Extract answer from prompt (after 'Answer:' line)
            prompt = batch["prompt"]
            answers = [p.split("Answer:")[-1].strip() if "Answer:" in p else "" for p in prompt]

            # Tokenize labels (answers)
            label_tokens = dataset.tokenizer(
                answers,
                truncation=True,
                padding="max_length",
                max_length=dataset.max_length,
                return_tensors="pt"
            )
            labels = label_tokens["input_ids"].to(device)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                image_embedding=image_embedding,
                labels=labels,
                graph=pyg_data
            )

            loss = outputs.loss
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            total_loss += loss.item()

        print(f"Epoch {epoch+1} - Loss: {total_loss / len(dataloader):.4f}")


if __name__ == "__main__":
    train()