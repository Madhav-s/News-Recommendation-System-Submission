import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import nltk
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

nltk.download("punkt", quiet=True)
nltk.download("punkt_tab", quiet=True)


NEWS_COLS = [
    "news_id",
    "category",
    "subcategory",
    "title",
    "abstract",
    "url",
    "title_entities",
    "abstract_entities",
]

BEHAVIOR_COLS = ["impression_id", "user_id", "time", "history", "impressions"]


def load_news(news_path: str) -> pd.DataFrame:
    """Load MIND news metadata TSV into a DataFrame."""
    return pd.read_csv(news_path, sep="\t", names=NEWS_COLS, usecols=range(8))


def load_behaviors(behaviors_path: str) -> pd.DataFrame:
    """Load MIND user behavior TSV into a DataFrame."""
    return pd.read_csv(behaviors_path, sep="\t", names=BEHAVIOR_COLS)


class NewsTokenizer:
    """Tokenizer/vocabulary helper for title text encoding."""

    def __init__(self, max_title_len: int = 30, min_word_freq: int = 2):
        self.max_title_len = max_title_len
        self.min_word_freq = min_word_freq
        self.word2idx = {"<PAD>": 0, "<UNK>": 1}
        self.idx2word = {0: "<PAD>", 1: "<UNK>"}

    def build_vocab(self, titles: List[str]) -> None:
        """Build word vocabulary from training titles."""
        nltk.download("punkt", quiet=True)
        nltk.download("punkt_tab", quiet=True)
        counts = Counter()
        for title in titles:
            if pd.isna(title):
                continue
            counts.update(nltk.word_tokenize(str(title).lower()))
        for token, freq in counts.items():
            if freq >= self.min_word_freq and token not in self.word2idx:
                idx = len(self.word2idx)
                self.word2idx[token] = idx
                self.idx2word[idx] = token

    def encode_title(self, title: str) -> List[int]:
        """Convert one title to fixed-length token index sequence."""
        nltk.download("punkt", quiet=True)
        nltk.download("punkt_tab", quiet=True)
        tokens = nltk.word_tokenize(str(title).lower()) if pd.notna(title) else []
        ids = [self.word2idx.get(tok, self.word2idx["<UNK>"]) for tok in tokens]
        if len(ids) < self.max_title_len:
            ids += [self.word2idx["<PAD>"]] * (self.max_title_len - len(ids))
        else:
            ids = ids[: self.max_title_len]
        return ids

    def dump(self, path: str) -> None:
        """Persist tokenizer configuration and vocabulary to JSON."""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "max_title_len": self.max_title_len,
                    "min_word_freq": self.min_word_freq,
                    "word2idx": self.word2idx,
                },
                f,
                indent=2,
            )

    @staticmethod
    def load(path: str) -> "NewsTokenizer":
        """Load tokenizer configuration and vocabulary from JSON."""
        with open(path, "r", encoding="utf-8") as f:
            payload = json.load(f)
        tok = NewsTokenizer(
            max_title_len=payload["max_title_len"],
            min_word_freq=payload["min_word_freq"],
        )
        tok.word2idx = payload["word2idx"]
        tok.idx2word = {idx: w for w, idx in tok.word2idx.items()}
        return tok


def load_glove(glove_path: str, word2idx: Dict[str, int], embed_dim: int = 300) -> np.ndarray:
    """Create embedding matrix from a GloVe text file and project vocabulary."""
    matrix = np.random.normal(0, 0.1, (len(word2idx), embed_dim)).astype("float32")
    matrix[word2idx["<PAD>"]] = 0.0
    try:
        with open(glove_path, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.rstrip().split(" ")
                word, values = parts[0], parts[1:]
                if word in word2idx and len(values) == embed_dim:
                    matrix[word2idx[word]] = np.asarray(values, dtype="float32")
    except FileNotFoundError as exc:
        raise FileNotFoundError(
            f"GloVe file not found at '{glove_path}'. "
            "Please verify the path, for example: data/glove/glove.6B.300d.txt"
        ) from exc
    return matrix


def build_news_encoding(news_df: pd.DataFrame, tokenizer: NewsTokenizer) -> Dict[str, np.ndarray]:
    """Encode every news title into token-id arrays keyed by news_id."""
    mapping: Dict[str, np.ndarray] = {}
    for _, row in news_df.iterrows():
        mapping[row["news_id"]] = np.asarray(tokenizer.encode_title(row["title"]), dtype=np.int64)
    return mapping


@dataclass
class TrainSample:
    """Container for one training sample in NRMS candidate ranking format."""

    history: np.ndarray
    candidates: np.ndarray
    labels: np.ndarray
    hist_mask: np.ndarray


def parse_behaviors_train(
    behaviors_df: pd.DataFrame,
    news_encoded: Dict[str, np.ndarray],
    max_history: int = 50,
    neg_k: int = 4,
    rng_seed: int = 42,
) -> List[TrainSample]:
    """Parse behaviors into NRMS train samples with negative sampling."""
    rng = np.random.default_rng(rng_seed)
    samples: List[TrainSample] = []
    pad_news = np.zeros_like(next(iter(news_encoded.values())))

    for _, row in behaviors_df.iterrows():
        history_ids = row["history"].split() if pd.notna(row["history"]) else []
        history_vecs = [news_encoded[nid] for nid in history_ids if nid in news_encoded][-max_history:]
        hist_mask = np.ones(len(history_vecs), dtype=np.int64)
        if len(history_vecs) < max_history:
            n_pad = max_history - len(history_vecs)
            history_vecs += [pad_news] * n_pad
            hist_mask = np.concatenate([hist_mask, np.zeros(n_pad, dtype=np.int64)])

        impressions = row["impressions"].split() if pd.notna(row["impressions"]) else []
        positives = [item.split("-")[0] for item in impressions if item.endswith("-1")]
        negatives = [item.split("-")[0] for item in impressions if item.endswith("-0")]
        if len(negatives) == 0:
            continue

        for pos in positives:
            if pos not in news_encoded:
                continue
            take = min(neg_k, len(negatives))
            sampled_neg = rng.choice(negatives, size=take, replace=False).tolist()
            cand_ids = [pos] + sampled_neg
            cand_vecs = [news_encoded[cid] for cid in cand_ids if cid in news_encoded]
            if len(cand_vecs) < 2:
                continue
            labels = np.zeros(len(cand_vecs), dtype=np.int64)
            labels[0] = 1
            samples.append(
                TrainSample(
                    history=np.stack(history_vecs, axis=0),
                    candidates=np.stack(cand_vecs, axis=0),
                    labels=labels,
                    hist_mask=hist_mask,
                )
            )

    return samples


class NRMSDataset(Dataset):
    """PyTorch Dataset wrapper for parsed NRMS training samples."""

    def __init__(self, samples: List[TrainSample]):
        self.samples = samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        item = self.samples[idx]
        return {
            "history": torch.tensor(item.history, dtype=torch.long),
            "candidates": torch.tensor(item.candidates, dtype=torch.long),
            "labels": torch.tensor(item.labels, dtype=torch.long),
            "hist_mask": torch.tensor(item.hist_mask, dtype=torch.long),
        }


def collate_train(batch):
    """Batch variable-size candidate sets with zero-padding for DataLoader."""
    max_cands = max(x["candidates"].shape[0] for x in batch)
    title_len = batch[0]["history"].shape[-1]
    hist_len = batch[0]["history"].shape[0]
    bsz = len(batch)

    histories = torch.zeros((bsz, hist_len, title_len), dtype=torch.long)
    hist_masks = torch.zeros((bsz, hist_len), dtype=torch.long)
    candidates = torch.zeros((bsz, max_cands, title_len), dtype=torch.long)
    labels = torch.zeros((bsz, max_cands), dtype=torch.long)

    for i, x in enumerate(batch):
        nc = x["candidates"].shape[0]
        histories[i] = x["history"]
        hist_masks[i] = x["hist_mask"]
        candidates[i, :nc] = x["candidates"]
        labels[i, :nc] = x["labels"]

    return {"history": histories, "candidates": candidates, "labels": labels, "hist_mask": hist_masks}
