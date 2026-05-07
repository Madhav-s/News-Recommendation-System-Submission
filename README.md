# News Recommendation System — MIND Dataset

## Introduction

This project implements an end-to-end neural news recommendation system using the MIND dataset and an NRMS-style architecture. The pipeline covers the full machine learning workflow: exploratory analysis, preprocessing, feature engineering, model training, hyperparameter comparison, and ranking evaluation.

News recommendation is uniquely challenging compared with product or movie recommendation. News articles have very short lifespans, user interests shift rapidly, and cold-start behavior is common because many users have limited history. These properties make ranking quality sensitive to both temporal dynamics and sparse user context.

The project uses **MIND-small** as the primary benchmark-scale dataset (about 50,000 users, about 65,000 articles, and about 230,000 impressions). This size is large enough to reflect real recommendation complexity while remaining practical for coursework experimentation.

## Learning Objectives

- Working with large-scale behavioral datasets (impression logs, click histories)
- Applying NLP techniques (tokenization, GloVe embeddings) to news content
- Designing and implementing NRMS neural architecture in PyTorch
- Evaluating with standard ranking metrics (AUC, MRR, nDCG)
- Analyzing model behavior and identifying failure modes

## Methodology

### Dataset

MIND-small is collected from anonymized Microsoft News logs (Oct-Nov 2019). The core files used here are:
- `news.tsv` for article metadata
- `behaviors.tsv` for user impression/click logs

### Model Architecture — NRMS

The model uses two encoders and a dot-product ranker:
- **News Encoder:** GloVe word embeddings -> dropout -> linear projection -> Multi-Head Self-Attention -> Additive Attention -> news vector
- **User Encoder:** clicked news vectors -> Multi-Head Self-Attention -> Additive Attention -> user vector
- **Prediction:** dot product between user vector and each candidate news vector

```
+---------------------------+     +---------------------------+
|      NEWS ENCODER         |     |      USER ENCODER         |
|  Title Words              |     |  Clicked News [n1..nH]    |
|  Word Embedding (GloVe)   |     |  News Encoder (shared)    |
|  Multi-Head Self-Attention|     |  Multi-Head Self-Attention|
|  Additive Attention       |     |  Additive Attention       |
|  News Vector (d-dim)      |     |  User Vector (d-dim)      |
+---------------------------+     +---------------------------+
            |                                 |
            +----------> DOT PRODUCT <--------+
                              |
                         Click Score
```

### Training Strategy

- Loss: CrossEntropyLoss (1 positive + K negative candidates per sample)
- Optimizer: Adam
- Gradient clipping: max_norm=1.0
- Negative sampling: K=4 non-clicked articles sampled per positive click

## Project Structure

```text
mind-recommender/
├── data/
│   ├── MINDsmall_train/
│   │   ├── behaviors.tsv
│   │   ├── news.tsv
│   │   ├── entity_embedding.vec
│   │   └── relation_embedding.vec
│   ├── MINDsmall_dev/
│   │   ├── behaviors.tsv
│   │   ├── news.tsv
│   │   ├── entity_embedding.vec
│   │   └── relation_embedding.vec
│   └── glove/
│       └── glove.6B.300d.txt
├── src/
│   ├── data_loader.py
│   ├── news_encoder.py
│   ├── user_encoder.py
│   ├── model.py
│   ├── train.py
│   └── evaluate.py
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_preprocessing.ipynb
│   └── 03_training.ipynb
├── models/
├── results/
├── README.md
└── requirements.txt
```

## Setup Instructions

### 1. Clone the repository
```bash
git clone <your-repo-url>
cd mind-recommender
```

### 2. Create environment and install dependencies
```bash
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 3. Download the MIND-small dataset
Download from Kaggle: [https://www.kaggle.com/datasets/arashnic/mind-news-dataset](https://www.kaggle.com/datasets/arashnic/mind-news-dataset)  
Place files as follows:
```text
data/MINDsmall_train/behaviors.tsv
data/MINDsmall_train/news.tsv
data/MINDsmall_dev/behaviors.tsv
data/MINDsmall_dev/news.tsv
```

### 4. Download GloVe embeddings
Download `glove.6B.zip` from [https://nlp.stanford.edu/projects/glove/](https://nlp.stanford.edu/projects/glove/)  
Extract and place:
```text
data/glove/glove.6B.300d.txt
```

**Note:** Data files and model checkpoints are excluded from Git due to size. Download datasets manually and regenerate checkpoints by running `notebooks/03_training.ipynb`.

## How to Run

### Option 1: Run Notebooks in Order (Recommended)
```text
1. notebooks/01_eda.ipynb          — Exploratory Data Analysis
2. notebooks/02_preprocessing.ipynb — Data Preprocessing
3. notebooks/03_training.ipynb      — Training & Evaluation
```

### Option 2: Run Training Script Directly
```bash
python src/train.py \
  --train-news data/MINDsmall_train/news.tsv \
  --train-behaviors data/MINDsmall_train/behaviors.tsv \
  --dev-news data/MINDsmall_dev/news.tsv \
  --dev-behaviors data/MINDsmall_dev/behaviors.tsv \
  --glove data/glove/glove.6B.300d.txt \
  --epochs 5 \
  --batch-size 64 \
  --lr 1e-4 \
  --device cuda
```

## Results

### Final Evaluation Metrics
| Metric   | Random Baseline | Basic NRMS | Tuned NRMS | Our Result |
|----------|----------------|------------|------------|------------|
| AUC      | ~0.500         | 0.62–0.66  | 0.67–0.70  | 0.590875 |
| MRR      | ~0.200         | 0.28–0.31  | 0.31–0.34  | 0.384113 |
| nDCG@5   | ~0.200         | 0.30–0.34  | 0.34–0.38  | 0.419472 |
| nDCG@10  | ~0.300         | 0.36–0.40  | 0.40–0.44  | 0.528586 |

### Hyperparameter Experiments
| Experiment | LR | Batch Size | Neg K | Dropout | AUC | MRR | nDCG@5 | nDCG@10 |
|---|---|---|---|---|---|---|---|---|
| Baseline | 1e-4 | 32 | 4 | 0.2 | 0.584000 | 0.516817 | 0.636197 | 0.636197 |
| Higher LR | 5e-4 | 64 | 4 | 0.2 | 0.573250 | 0.514083 | 0.633868 | 0.633868 |
| More Negatives | 1e-4 | 32 | 8 | 0.3 | 0.595625 | 0.389353 | 0.422445 | 0.532559 |

The highest AUC came from the higher-difficulty negative sampling setup (K=8 with dropout 0.3). The baseline kept stronger MRR/nDCG@5 on the sampled run, indicating a trade-off between global discrimination and top-rank precision. Additional full-data training epochs should improve both.

## Error Analysis

### Cold-Start Problem
Users with very short click histories (fewer than 5 articles) are the hardest cases for the model. When history is empty or near-empty, the User Encoder has almost no signal to aggregate, and the user vector defaults to a near-zero or noise representation. This makes personalized ranking nearly impossible for new users.

### Category Imbalance
If certain categories dominate the dataset (e.g. news, sports), the model learns strong priors toward those categories and may underperform on underrepresented categories like finance or health. Users whose interests lie in minority categories are likely to get lower-quality recommendations.

### Short Training Limitation
The first-pass training run used only 1 epoch on a sample of the data. AUC near 0.58 (below even a Basic NRMS baseline of 0.62–0.66) is expected in this scenario and does not reflect the model's full capacity. Training for 5+ epochs on the full dataset is expected to bring metrics into the baseline range.

### nDCG@5 vs nDCG@10 Gap
When nDCG@5 equals nDCG@10, it indicates the model is not successfully placing relevant articles in positions 6–10, meaning nearly all ranking benefit comes from the very top results. A well-tuned model would show nDCG@10 meaningfully higher than nDCG@5.

## Conclusions

This project successfully implemented an end-to-end neural news recommendation system based on the NRMS architecture, covering the complete machine learning pipeline from raw data ingestion through evaluation. The model correctly implements the two-encoder design (News Encoder and User Encoder) with multi-head self-attention and GloVe word embeddings.

The best-performing hyperparameter configuration was the **More Negatives** experiment (`lr=1e-4`, `batch_size=32`, `neg_k=8`, `dropout=0.3`, `num_heads=16`) based on AUC. Increasing negative samples (K=8) improved discrimination on AUC, while a higher learning rate converged quickly but produced lower AUC in this setup.

Key limitations of the current approach include reliance on static GloVe embeddings (which do not capture context), the inability to handle cold-start users well, and the lack of category-aware features.

Future improvements that would meaningfully boost performance:
1. Replace GloVe with DistilBERT or a fine-tuned language model for contextualized title representations
2. Add category and subcategory embeddings concatenated to the news vector (Category-Aware Encoding)
3. Implement a cold-start fallback using global article popularity for users with fewer than 5 history clicks
4. Train for 10+ epochs on the full MIND-small dataset with a learning rate scheduler

## Reproducibility
- Python version: 3.9+
- PyTorch version: 2.0+
- Random seed: 42 (set in train.py for torch, numpy, and random)
- Hardware: NVIDIA GPU (CUDA) recommended; CPU training also supported
- All dependencies listed in requirements.txt
