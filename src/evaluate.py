import numpy as np
import torch
from sklearn.metrics import roc_auc_score


def dcg_score(y_true, y_score, k=10):
    """Compute DCG@k for binary relevance labels and predicted scores."""
    order = np.argsort(y_score)[::-1][:k]
    gains = np.array(y_true)[order]
    discounts = np.log2(np.arange(len(gains)) + 2)
    return np.sum(gains / discounts)


def ndcg_score(y_true, y_score, k=10):
    """Compute normalized DCG@k."""
    best = dcg_score(y_true, y_true, k)
    if best == 0:
        return 0.0
    return dcg_score(y_true, y_score, k) / best


def mrr_score(y_true, y_score):
    """Compute reciprocal rank of the first relevant item."""
    order = np.argsort(y_score)[::-1]
    ranked = np.array(y_true)[order]
    for i, val in enumerate(ranked):
        if val == 1:
            return 1.0 / (i + 1)
    return 0.0


def evaluate(model, val_loader, device="cpu"):
    """Evaluate ranking metrics on a validation DataLoader."""
    model.eval()
    aucs, mrrs, ndcg5s, ndcg10s = [], [], [], []
    with torch.no_grad():
        for batch in val_loader:
            history = batch["history"].to(device)
            candidates = batch["candidates"].to(device)
            labels = batch["labels"].cpu().numpy()
            hist_mask = batch["hist_mask"].to(device)
            scores = model(history, candidates, hist_mask).cpu().numpy()

            for i in range(len(labels)):
                y_true = labels[i]
                y_score = scores[i][: len(y_true)]
                if y_true.sum() == 0 or y_true.sum() == len(y_true):
                    continue
                aucs.append(roc_auc_score(y_true, y_score))
                mrrs.append(mrr_score(y_true, y_score))
                ndcg5s.append(ndcg_score(y_true, y_score, 5))
                ndcg10s.append(ndcg_score(y_true, y_score, 10))

    return {
        "AUC": float(np.mean(aucs)) if aucs else 0.0,
        "MRR": float(np.mean(mrrs)) if mrrs else 0.0,
        "nDCG@5": float(np.mean(ndcg5s)) if ndcg5s else 0.0,
        "nDCG@10": float(np.mean(ndcg10s)) if ndcg10s else 0.0,
    }
