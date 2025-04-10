"""
Generate MMKG graphs from the entire ScienceQA dataset and save them as individual .pt files.
"""

from load_scienceqa import load_scienceqa
from entity_extract import extract_entities
from build_mmkg import build_mmkg
import torch
from pathlib import Path
from tqdm import tqdm


def generate_mmkg_dataset(problems_path: str, splits_path: str, image_root: str, output_dir: str, split: str):
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    questions = load_scienceqa(problems_path, splits_path, image_root, split)

    for q in tqdm(questions):
        qid = q["question_id"]
        qtext = q["question"]
        qimage = q.get("image_path")
        qentities = extract_entities(qtext)

        G = build_mmkg(qid, qtext, qimage, qentities)
        torch.save(G, Path(output_dir) / f"mmkg_{split}_{qid}.pt")

    print(f"Saved {len(questions)} MMKG graphs to {output_dir} for split: {split}")


if __name__ == "__main__":
    for split in ["train", "val", "test"]:
        generate_mmkg_dataset(
            problems_path="data/ScienceQA/data/scienceqa/problems.json",
            splits_path="data/ScienceQA/data/scienceqa/pid_splits.json",
            image_root="data/ScienceQA/",
            output_dir=f"data/scienceqa/mmkg_graphs/{split}",
            split=split
        )