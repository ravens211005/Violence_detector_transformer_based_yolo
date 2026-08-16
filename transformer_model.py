"""
transformer_model.py  —  Violence Detection Transformer
========================================================
Architecture preserved as-is per requirements.
No modifications required by REQ-1 through REQ-4.
"""

import math
import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):
    """Standard sinusoidal positional encoding."""

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 512):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq_len, d_model)
        x = x + self.pe[:, : x.size(1)]
        return self.dropout(x)


class ViolenceTransformer(nn.Module):
    """
    Transformer Encoder for binary violence classification.

    Input:  (batch, seq_len, feature_dim)
    Output: (batch, 2)  — logits for [non-violence, violence]
    """

    def __init__(
        self,
        feature_dim:   int,
        embedding_dim: int   = 256,
        num_heads:     int   = 8,
        num_layers:    int   = 4,
        dropout:       float = 0.3,
        num_classes:   int   = 2,
    ):
        super().__init__()

        # Project raw pose features into the model dimension
        self.input_projection = nn.Linear(feature_dim, embedding_dim)

        self.pos_encoding = PositionalEncoding(embedding_dim, dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embedding_dim,
            nhead=num_heads,
            dim_feedforward=embedding_dim * 4,
            dropout=dropout,
            batch_first=True,
            norm_first=True,         # Pre-norm (more stable)
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
            enable_nested_tensor=False,
        )

        self.norm       = nn.LayerNorm(embedding_dim)
        self.classifier = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(embedding_dim // 2, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq_len, feature_dim)
        x = self.input_projection(x)        # → (batch, seq_len, embedding_dim)
        x = self.pos_encoding(x)
        x = self.transformer_encoder(x)     # → (batch, seq_len, embedding_dim)
        x = self.norm(x)
        # Global average pooling across the sequence dimension
        x = x.mean(dim=1)                   # → (batch, embedding_dim)
        logits = self.classifier(x)         # → (batch, num_classes)
        return logits
