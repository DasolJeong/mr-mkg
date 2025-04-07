
"""
Load ScienceQA problems with image paths, based on official repo structure.
"""

import json
from pathlib import Path
from typing import List, Dict


def load_scienceqa(problems_path: str, splits_path: str, image_root: str, split: str = "train") -> List[Dict]:
    """
    Load ScienceQA questions with resolved image paths based on split.

    Args:
        problems_path (str): Path to problems.json
        splits_path (str): Path to pid_splits.json
        image_root (str): Root folder of images (contains train/val/test)
        split (str): One of ['train', 'val', 'test']

    Returns:
        List[Dict]: List of questions with image_path and metadata
    """
    problems = json.load(open(problems_path, 'r'))
    pid_splits = json.load(open(splits_path, 'r'))

    selected_pids = set(pid_splits[split])  # keep as string for matching
    questions = []

    for pid_str, sample in problems.items():
        if pid_str not in selected_pids:
            continue

        q = {
            "question_id": pid_str,
            "question": sample.get("question"),
            "choices": sample.get("choices"),
            "answer": sample.get("answer"),
        }

        # Resolve image path if available
        if sample.get("image"):
            image_path = Path(image_root) / split / pid_str / "image.png"
            if image_path.exists():
                q["image_path"] = str(image_path)
            else:
                q["image_path"] = None
        else:
            q["image_path"] = None

        questions.append(q)

    return questions



if __name__ == "__main__":
    qs = load_scienceqa(
        problems_path="data/ScienceQA/data/scienceqa/problems.json",
        splits_path="data/ScienceQAdata/scienceqa/pid_splits.json",
        image_root="data/ScienceQAdata/",
        split="train"
    )
    print(f"Loaded {len(qs)} questions from train split")
    print(qs[0])