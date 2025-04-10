# MR-MKG: Multimodal Reasoning with Multimodal Knowledge Graphs

This repository is an implementation of the ICLR 2024 paper:
**"Multimodal Reasoning with Multimodal Knowledge Graphs" (MR-MKG)**.
It supports visual-textual reasoning using knowledge-enhanced LLMs with no LLM fine-tuning.

---

## Project Structure

```
mr-mkg/
├── train.py                  # 학습 전체 파이프라인
├── inference.py              # 임의 질문 + 이미지 추론 스크립트
├── test.py                   # test set 기반 평가
├── validate.py               # validation loss 평가용 스크립트
│
├── models/                   # 모델 구성 모듈
│   ├── mr_mkg.py             # 전체 MR-MKG 모델 정의
│   ├── encoders.py           # LanguageEncoder, KGEncoder(RGAT)
│   ├── adapters.py           # VisualAdapter, KnowledgeAdapter
│   ├── rgat.py               # RGAT layer 정의
│   └── cross_modal_align.py  # 이미지-엔티티 정렬 loss (Triplet)
│
├── data/
│   ├── build_mmkg.py         # ScienceQA → MMKG 변환
│   ├── generate_mmkg_dataset.py # 전체 데이터셋 구축 스크립트
│   ├── load_scienceqa.py     # ScienceQA 포맷 로딩
│   ├── mrmkg_dataset.py      # 학습용 MMKG Dataset 클래스
│   └── scienceqa/            # 원본 데이터 저장 위치
│
├── utils/
│   ├── graph_utils.py        # NetworkX → DGL 변환, subgraph 추출
│   ├── visual_feature.py     # CLIP 임베딩 및 초기화
│   └── align_utils.py        # cross-modal triplet loss 계산
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
