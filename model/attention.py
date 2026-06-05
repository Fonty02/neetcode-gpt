import torch
import torch.nn as nn
from torchtyping import TensorType

class SingleHeadAttention(nn.Module):

    def __init__(self, embedding_dim: int, attention_dim: int):
        super().__init__()
        torch.manual_seed(0)
        # Create three linear projections (Key, Query, Value) with bias=False
        # Instantiation order matters for reproducible weights: key, query, value
        self.embedding_dim=embedding_dim
        self.attention_dim=attention_dim
        self.Key=nn.Linear(embedding_dim,attention_dim,False)
        self.Query=nn.Linear(embedding_dim,attention_dim,False)
        self.Value=nn.Linear(embedding_dim,attention_dim,False)

    def forward(self, embedded: TensorType[float]) -> TensorType[float]:
        # 1. Project input through K, Q, V linear layers
        # 2. Compute attention scores: (Q @ K^T) / sqrt(attention_dim)
        # 3. Apply causal mask: use torch.tril(torch.ones(...)) to build lower-triangular matrix,
        #    then masked_fill positions where mask == 0 with float('-inf')
        # 4. Apply softmax(dim=2) to masked scores
        # 5. Return (scores @ V) rounded to 4 decimal places
        K=self.Key(embedded)
        Q=self.Query(embedded)
        V=self.Value(embedded)
        attn_scores=(Q@K.transpose(-1,-2))/torch.sqrt(torch.tensor(self.attention_dim))
        seq_len=embedded.shape[1]
        mask = torch.tril(torch.ones(seq_len, seq_len))
        attn_scores = attn_scores.masked_fill(mask == 0, float('-inf'))
        attn_weights = torch.softmax(attn_scores, dim=-1)
        out = attn_weights @ V
        return torch.round(out, decimals=4)
