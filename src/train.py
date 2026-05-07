import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from data_loader import (
    NRMSDataset,
    NewsTokenizer,
    build_news_encoding,
    collate_train,
    load_behaviors,
    load_glove,
    load_news,
    parse_behaviors_train,
)
from evaluate import evaluate
from model import NRMSModel


def set_seed(seed: int = 42):
    """Set random seeds across Python, NumPy, and PyTorch for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def parse_args():
    """Parse command-line arguments for NRMS training."""
    p = argparse.ArgumentParser(description="Train NRMS on MIND-small")
    p.add_argument("--train-news", default="data/MINDsmall_train/news.tsv")
    p.add_argument("--train-behaviors", default="data/MINDsmall_train/behaviors.tsv")
    p.add_argument("--val-news", default="data/MINDsmall_dev/news.tsv")
    p.add_argument("--val-behaviors", default="data/MINDsmall_dev/behaviors.tsv")
    p.add_argument("--glove-path", default="data/glove/glove.6B.300d.txt")
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--neg-k", type=int, default=4)
    p.add_argument("--max-history", type=int, default=50)
    p.add_argument("--max-title-len", type=int, default=30)
    p.add_argument("--num-heads", type=int, default=16)
    p.add_argument("--head-dim", type=int, default=16)
    p.add_argument("--dropout", type=float, default=0.2)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    p.add_argument("--require-cuda", action="store_true")
    p.add_argument("--max-train-samples", type=int, default=0)
    p.add_argument("--max-val-samples", type=int, default=0)
    return p.parse_args()


def main():
    """Run end-to-end training and validation evaluation."""
    args = parse_args()
    set_seed(args.seed)
    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device
    if args.require_cuda and not torch.cuda.is_available():
        raise RuntimeError("CUDA was required but is not available in this Python environment.")
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("Requested CUDA device but torch.cuda.is_available() is False.")

    Path("models").mkdir(parents=True, exist_ok=True)
    Path("results").mkdir(parents=True, exist_ok=True)

    train_news = load_news(args.train_news)
    train_beh = load_behaviors(args.train_behaviors)
    val_news = load_news(args.val_news)
    val_beh = load_behaviors(args.val_behaviors)

    tokenizer = NewsTokenizer(max_title_len=args.max_title_len, min_word_freq=2)
    tokenizer.build_vocab(train_news["title"].fillna("").tolist())
    tokenizer.dump("results/tokenizer.json")

    embedding_matrix = load_glove(args.glove_path, tokenizer.word2idx, embed_dim=300)

    train_encoded = build_news_encoding(train_news, tokenizer)
    val_encoded = build_news_encoding(val_news, tokenizer)

    train_samples = parse_behaviors_train(
        train_beh, train_encoded, max_history=args.max_history, neg_k=args.neg_k, rng_seed=args.seed
    )
    val_samples = parse_behaviors_train(
        val_beh, val_encoded, max_history=args.max_history, neg_k=args.neg_k, rng_seed=args.seed
    )
    if args.max_train_samples > 0:
        train_samples = train_samples[: args.max_train_samples]
    if args.max_val_samples > 0:
        val_samples = val_samples[: args.max_val_samples]

    train_loader = DataLoader(
        NRMSDataset(train_samples), batch_size=args.batch_size, shuffle=True, collate_fn=collate_train
    )
    val_loader = DataLoader(NRMSDataset(val_samples), batch_size=args.batch_size, shuffle=False, collate_fn=collate_train)

    model = NRMSModel(embedding_matrix, args.num_heads, args.head_dim, args.dropout).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss()

    losses = []
    for epoch in range(args.epochs):
        model.train()
        total_loss = 0.0
        loop = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{args.epochs}")
        for batch in loop:
            history = batch["history"].to(device)
            candidates = batch["candidates"].to(device)
            hist_mask = batch["hist_mask"].to(device)

            optimizer.zero_grad()
            scores = model(history, candidates, hist_mask)
            target = torch.zeros(scores.size(0), dtype=torch.long, device=device)
            loss = criterion(scores, target)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            total_loss += loss.item()
            loop.set_postfix(loss=loss.item())

        avg_loss = total_loss / max(1, len(train_loader))
        losses.append(avg_loss)
        ckpt = f"models/nrms_epoch{epoch + 1}.pt"
        torch.save(model.state_dict(), ckpt)
        print(f"Epoch {epoch + 1}: loss={avg_loss:.4f} checkpoint={ckpt}")

    metrics = evaluate(model, val_loader, device=device)
    print(json.dumps(metrics, indent=2))

    with open("results/metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    with open("results/losses.json", "w", encoding="utf-8") as f:
        json.dump(losses, f, indent=2)


if __name__ == "__main__":
    main()
