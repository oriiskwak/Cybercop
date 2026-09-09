"""Packaged S3~S6 shared-encoder multi-head model definition."""

from __future__ import annotations

import torch
import torch.nn as nn
from transformers import AutoConfig, AutoModel


def mean_pooling(last_hidden_state: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    mask = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
    summed = torch.sum(last_hidden_state * mask, dim=1)
    counts = torch.clamp(mask.sum(dim=1), min=1e-9)
    return summed / counts


def load_encoder(model_name: str):
    try:
        return AutoModel.from_pretrained(model_name)
    except OSError:
        return AutoModel.from_config(AutoConfig.from_pretrained(model_name))


class GuidelineMultiTaskClassifier(nn.Module):
    def __init__(self, model_name: str, status_columns: list[str], dropout: float = 0.2):
        super().__init__()
        self.encoder = load_encoder(model_name)
        hidden_size = self.encoder.config.hidden_size
        self.status_columns = list(status_columns)
        self.heads = nn.ModuleDict({
            column: nn.Sequential(
                nn.Linear(hidden_size, 256),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(256, 4),
            )
            for column in self.status_columns
        })

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        pooled = mean_pooling(outputs.last_hidden_state, attention_mask)
        return torch.stack([self.heads[c](pooled) for c in self.status_columns], dim=1)
