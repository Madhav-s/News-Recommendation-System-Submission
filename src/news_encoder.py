import torch
import torch.nn as nn
import torch.nn.functional as F


class AdditiveAttention(nn.Module):
    """Aggregate sequence features into one vector via additive attention."""

    def __init__(self, dim: int, hidden_dim: int = 200):
        super().__init__()
        self.proj = nn.Linear(dim, hidden_dim)
        self.query = nn.Linear(hidden_dim, 1, bias=False)

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        """Compute attention-weighted pooled representation with optional mask."""
        e = torch.tanh(self.proj(x))
        scores = self.query(e).squeeze(-1)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        weights = F.softmax(scores, dim=-1)
        return torch.bmm(weights.unsqueeze(1), x).squeeze(1)


class NewsEncoder(nn.Module):
    """Encode tokenized news titles into dense news vectors."""

    def __init__(self, embedding_matrix, num_heads: int = 16, head_dim: int = 16, dropout: float = 0.2):
        super().__init__()
        embed_dim = embedding_matrix.shape[1]
        attn_dim = num_heads * head_dim
        self.word_embed = nn.Embedding.from_pretrained(torch.FloatTensor(embedding_matrix), freeze=False)
        self.dropout = nn.Dropout(dropout)
        self.proj = nn.Linear(embed_dim, attn_dim)
        self.multihead_attn = nn.MultiheadAttention(embed_dim=attn_dim, num_heads=num_heads, batch_first=True)
        self.additive_attn = AdditiveAttention(attn_dim)

    def forward(self, title_ids: torch.Tensor) -> torch.Tensor:
        """Encode title token ids to news embeddings."""
        x = self.dropout(self.word_embed(title_ids))
        x = self.proj(x)
        x, _ = self.multihead_attn(x, x, x)
        x = self.dropout(x)
        return self.additive_attn(x)
