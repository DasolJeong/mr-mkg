# MR-MKG: Multimodal Reasoning with Multimodal Knowledge Graphs

This repository is an implementation of the ICLR 2024 paper:
**"Multimodal Reasoning with Multimodal Knowledge Graphs" (MR-MKG)**.
It supports visual-textual reasoning using knowledge-enhanced LLMs with no LLM fine-tuning.

---

## Project Structure

```
mr-mkg/
├── train.py                  # Training script
├── inference.py              # Free-form question + image inference
├── test.py                   # Test set evaluation
├── validate.py               # Validation loss evaluation
│
├── models/                   # Model components
│   ├── mr_mkg.py             # Full MR-MKG model
│   ├── encoders.py           # Text and KG encoders
│   ├── adapters.py           # Visual and knowledge adapters
│   ├── rgat.py               # RGAT layers
│   └── cross_modal_align.py  # Image-text alignment loss
│
├── data/                     # Dataset processing
│   ├── build_mmkg.py         # Build MMKG graphs
│   ├── generate_mmkg_dataset.py # Save MMKG datasets
│   ├── load_scienceqa.py     # Load ScienceQA JSON
│   ├── mrmkg_dataset.py      # MMKG PyTorch Dataset
│   └── scienceqa/            # Raw ScienceQA data
│
├── utils/                    # Helper functions
│   ├── graph_utils.py        # Graph processing
│   ├── visual_feature.py     # Image embeddings
│   └── align_utils.py        # Alignment loss
│
└── README.md             


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
