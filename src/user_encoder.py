import torch
import torch.nn as nn

from news_encoder import AdditiveAttention


class UserEncoder(nn.Module):
    """Encode clicked news sequence into a user preference vector."""

    def __init__(self, news_dim: int, num_heads: int = 16, dropout: float = 0.2):
        super().__init__()
        self.multihead_attn = nn.MultiheadAttention(embed_dim=news_dim, num_heads=num_heads, batch_first=True)
        self.additive_attn = AdditiveAttention(news_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, clicked_news_vecs: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        """Compute user vector from clicked-news vectors with optional mask."""
        x, _ = self.multihead_attn(clicked_news_vecs, clicked_news_vecs, clicked_news_vecs)
        x = self.dropout(x)
        return self.additive_attn(x, mask)
