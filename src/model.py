import torch
import torch.nn as nn

from news_encoder import NewsEncoder
from user_encoder import UserEncoder


class NRMSModel(nn.Module):
    """NRMS ranking model: shared news encoder + user encoder + dot-product scorer."""

    def __init__(self, embedding_matrix, num_heads: int = 16, head_dim: int = 16, dropout: float = 0.2):
        super().__init__()
        news_dim = num_heads * head_dim
        self.news_encoder = NewsEncoder(embedding_matrix, num_heads=num_heads, head_dim=head_dim, dropout=dropout)
        self.user_encoder = UserEncoder(news_dim, num_heads=num_heads, dropout=dropout)

    def forward(self, history_ids: torch.Tensor, candidate_ids: torch.Tensor, hist_mask: torch.Tensor = None):
        """Return click scores for candidates given user history and optional mask."""
        batch, hist_len, title_len = history_ids.shape
        _, num_cands, _ = candidate_ids.shape

        hist_flat = history_ids.view(-1, title_len)
        hist_vecs = self.news_encoder(hist_flat).view(batch, hist_len, -1)

        cand_flat = candidate_ids.view(-1, title_len)
        cand_vecs = self.news_encoder(cand_flat).view(batch, num_cands, -1)

        user_vec = self.user_encoder(hist_vecs, hist_mask)
        scores = torch.bmm(cand_vecs, user_vec.unsqueeze(-1)).squeeze(-1)
        return scores
