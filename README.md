# MR-MKG: Multimodal Reasoning with Multimodal Knowledge Graphs

This repository is an implementation of the ICLR 2024 paper:
**"Multimodal Reasoning with Multimodal Knowledge Graphs" (MR-MKG)**.
It supports visual-textual reasoning using knowledge-enhanced LLMs with no LLM fine-tuning.

---

## Project Structure

```
mr-mkg/
├── train.py                  # Main training pipeline for MR-MKG
├── inference.py              # Inference script for free-form questions + image input
├── test.py                   # Evaluation script on test set with answer comparison
├── validate.py               # Validation-only evaluation script (loss calculation)
│
├── models/                   # Core model components
│   ├── mr_mkg.py             # Full MR-MKG model: integrates LLM, KG encoder, adapters
│   ├── encoders.py           # LanguageEncoder (LLM-based), KGEncoder (RGAT-based)
│   ├── adapters.py           # VisualAdapter and KnowledgeAdapter (cross-modal fusion)
│   ├── rgat.py               # Relational GAT layers for knowledge graph encoding
│   └── cross_modal_align.py  # Triplet loss module for image-text alignment
│
├── data/                     # Dataset loading and preprocessing
│   ├── build_mmkg.py         # Convert ScienceQA to MMKG format (graph-based)
│   ├── generate_mmkg_dataset.py # Build final MMKG .pt datasets (train/val/test)
│   ├── load_scienceqa.py     # Load and parse original ScienceQA format
│   ├── mrmkg_dataset.py      # PyTorch Dataset class for MMKG graphs
│   └── scienceqa/            # Directory for raw ScienceQA JSON files
│
├── utils/                    # Utility modules
│   ├── graph_utils.py        # Graph conversion: NetworkX ↔ DGL, subgraph extraction
│   ├── visual_feature.py     # CLIP-based image embedding and node feature init
│   └── align_utils.py        # Cross-modal alignment loss (image ↔ entity embedding)
│
└── README.md                 # Project overview, instructions, and usage guide

```

---

## Method Overview

MR-MKG introduces a pipeline to integrate multimodal knowledge graphs into a frozen LLM using:

- **CLIP**: for image feature extraction
- **Visual Adapter**: linear projection to LLM space
- **RGAT**: for knowledge graph (MMKG) encoding
- **Knowledge Adapter**: attention-based summarization of KG embeddings
- **Prompt Injection**: `[IMAGE]`, `[KNOWLEDGE]` tokens are replaced with respective embeddings
- **Triplet Loss**: alignment between image & KG representations

---

## Setup

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

If using PyTorch Geometric:
```bash
pip install torch-scatter torch-sparse torch-cluster torch-spline-conv torch-geometric \
  -f https://data.pyg.org/whl/torch-<torch_version>.html
```

### 2. Download and prepare ScienceQA
Place the `problems.json`, `pid_splits.json`, `captions.json`, and image folders into:
```
data/scienceqa/
├── train/<pid>/image.png
├── val/<pid>/image.png
├── test/<pid>/image.png
├── problems.json
├── pid_splits.json
└── captions.json
```

### 3. Generate MMKG graphs
```bash
python data/generate_mmkg_dataset.py
```

---

### Model Architecture 
```pgsql
Input: [text prompt] + [image] + [entity KG]

          ┌───────────────────────────┐
          │     Language Encoder      │ ← input_ids
          └────────────┬──────────────┘
                       │
         ┌─────────────▼──────────────┐
         │       Prompt Embedding     │ = HT + HK + HI
         └────────┬────────┬──────────┘
                  │        │
       ┌──────────▼┐      ┌▼─────────────┐
       │ KGEncoder │      │ VisualAdapter│
       └──────────▲┘      └▲─────────────┘
                  │        │
                  └────┬───┘
                       ▼
                  T5 (frozen)
                       │
                    Answer

```

## Training
```bash
python train.py
```
Logs training & validation loss across epochs with both generative and alignment loss.

---

## Evaluation
```bash
python validate.py    # validation set
python test.py        # test set

```


## Inference
```bash
python inference.py
```
Enter a question: What is this?
Enter an image path (or leave empty for no image): ./images/eiffel.jpg

---

## Citation
If you use this code, please consider citing the original paper:

```bibtex
@inproceedings{diao2024mrmkg,
  title={Multimodal Reasoning with Multimodal Knowledge Graphs},
  author={Diao, Haotian and Zhang, Junnan and Zhang, Xiyang and Lin, Dahua and Wang, Yichen},
  booktitle={ICLR},
  year={2024}
}
```

---

## 💬 Contact
For any questions, feel free to reach out via [issues](https://github.com/DasolJeong/mr-mkg/issues) or email.

---
