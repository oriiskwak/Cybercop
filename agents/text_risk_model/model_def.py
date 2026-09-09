"""
model/model_def.py
==================
노트북에 분산된 모델 클래스 3종을 한 파일에 통합합니다.

- KoreanRiskClassifier      : MLP risk_score 회귀 모델 (Cell 8)
- SemiSupMLPClassifier      : MLP risk_score 준지도 회귀 모델 (Cell 11)
- KoreanRiskTextClassifier  : text-only MLP risk_score 회귀 모델
- SemiSupTextClassifier     : text-only MLP risk_score 준지도 회귀 모델
- KoreanTextAutoEncoder     : AE 이상탐지 모델 (Cell 14)
- SeverityMLP               : AE 이후 심각도 분류 모델 (Cell 14)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoConfig, AutoModel


# =========================================
# 공통 유틸
# =========================================

def mean_pooling(last_hidden_state: torch.Tensor,
                 attention_mask: torch.Tensor) -> torch.Tensor:
    """attention mask 기반 mean pooling."""
    mask   = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
    summed = torch.sum(last_hidden_state * mask, dim=1)
    counts = torch.clamp(mask.sum(dim=1), min=1e-9)
    return summed / counts


def load_encoder(model_name: str):
    """Load encoder from weights if available, otherwise from config.

    The packaged risk_agent stores the fine-tuned checkpoint separately, so the
    local HuggingFace folder only needs config/tokenizer files. The checkpoint
    state_dict is loaded immediately after model construction.
    """
    try:
        return AutoModel.from_pretrained(model_name)
    except OSError:
        config = AutoConfig.from_pretrained(model_name)
        return AutoModel.from_config(config)


# =========================================
# Model 1: MLP 지도학습
# =========================================

class KoreanRiskClassifier(nn.Module):
    """
    KoSimCSE-roberta 인코더 + (hidden + amount) → risk_score 회귀.
    학습 시 인코더 전체를 fine-tune합니다.
    """

    def __init__(self, model_name: str, num_classes: int = 1, dropout: float = 0.2):
        super().__init__()
        self.encoder = load_encoder(model_name)
        hidden_size  = self.encoder.config.hidden_size

        self.classifier = nn.Sequential(
            nn.Linear(hidden_size + 1, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes),
        )

    def forward(self,
                input_ids: torch.Tensor,
                attention_mask: torch.Tensor,
                amount_feat: torch.Tensor) -> torch.Tensor:
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        pooled  = mean_pooling(outputs.last_hidden_state, attention_mask)
        fused   = torch.cat([pooled, amount_feat], dim=1)
        return self.classifier(fused)


# =========================================
# Model 1-B: Text-only MLP 지도학습
# =========================================

class KoreanRiskTextClassifier(nn.Module):
    """
    KoSimCSE-roberta 인코더 + hidden → risk_score 회귀.
    amount feature를 사용하지 않는 text-only 지도학습 모델입니다.
    """

    def __init__(self, model_name: str, num_classes: int = 1, dropout: float = 0.2):
        super().__init__()
        self.encoder = load_encoder(model_name)
        hidden_size = self.encoder.config.hidden_size

        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes),
        )

    def forward(self,
                input_ids: torch.Tensor,
                attention_mask: torch.Tensor) -> torch.Tensor:
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        pooled = mean_pooling(outputs.last_hidden_state, attention_mask)
        return self.classifier(pooled)


# =========================================
# Model 2: MLP 준지도학습
# =========================================

class SemiSupMLPClassifier(nn.Module):
    """
    KoreanRiskClassifier와 구조는 동일하나,
    인코더의 마지막 N개 레이어만 unfreeze하여 학습 효율을 높입니다.
    pseudo labeling 라운드마다 재초기화해서 사용합니다.
    """

    def __init__(self,
                 model_name: str,
                 num_classes: int = 1,
                 dropout: float = 0.2,
                 unfreeze_last_n: int = 2):
        super().__init__()
        self.encoder = load_encoder(model_name)

        # 인코더 전체 freeze
        for param in self.encoder.parameters():
            param.requires_grad = False

        # 마지막 N개 레이어만 unfreeze
        for layer in self.encoder.encoder.layer[-unfreeze_last_n:]:
            for param in layer.parameters():
                param.requires_grad = True

        # pooler unfreeze (있는 경우)
        if hasattr(self.encoder, "pooler") and self.encoder.pooler is not None:
            for param in self.encoder.pooler.parameters():
                param.requires_grad = True

        hidden_size = self.encoder.config.hidden_size
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size + 1, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 1),
        )

    def forward(self,
                input_ids: torch.Tensor,
                attention_mask: torch.Tensor,
                amount_feat: torch.Tensor) -> torch.Tensor:
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        pooled  = mean_pooling(outputs.last_hidden_state, attention_mask)
        fused   = torch.cat([pooled, amount_feat], dim=1)
        return self.classifier(fused)


# =========================================
# Model 2-B: Text-only MLP 준지도학습
# =========================================

class SemiSupTextClassifier(nn.Module):
    """
    SemiSupMLPClassifier의 text-only 버전.
    encoder 마지막 N개 레이어만 unfreeze하고 amount feature 없이 risk_score를 회귀합니다.
    """

    def __init__(self,
                 model_name: str,
                 num_classes: int = 1,
                 dropout: float = 0.2,
                 unfreeze_last_n: int = 2):
        super().__init__()
        self.encoder = load_encoder(model_name)

        for param in self.encoder.parameters():
            param.requires_grad = False

        for layer in self.encoder.encoder.layer[-unfreeze_last_n:]:
            for param in layer.parameters():
                param.requires_grad = True

        if hasattr(self.encoder, "pooler") and self.encoder.pooler is not None:
            for param in self.encoder.pooler.parameters():
                param.requires_grad = True

        hidden_size = self.encoder.config.hidden_size
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 1),
        )

    def forward(self,
                input_ids: torch.Tensor,
                attention_mask: torch.Tensor) -> torch.Tensor:
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        pooled = mean_pooling(outputs.last_hidden_state, attention_mask)
        return self.classifier(pooled)


# =========================================
# Model 4: FraudAutoEncoder (사기 전용 비지도)
# =========================================

class FraudAutoEncoder(nn.Module):
    """
    사기 데이터 전용 AutoEncoder. 768 → 256 → 64 → 256 → 768.
    roberta 인코더는 freeze하고 projector + decoder만 학습합니다.
    latent z (64차원)를 클러스터링의 representation으로 사용합니다.
    """

    def __init__(self, model_name: str, dropout: float = 0.2):
        super().__init__()
        self.encoder = load_encoder(model_name)

        self.projector = nn.Sequential(
            nn.Linear(768, 256), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(256, 64),
        )
        self.decoder = nn.Sequential(
            nn.Linear(64, 256), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(256, 768),
        )

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor):
        with torch.no_grad():
            outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        pooled = mean_pooling(outputs.last_hidden_state, attention_mask)
        z      = self.projector(pooled)
        recon  = self.decoder(z)
        return pooled, z, recon

    def encode(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        pooled = mean_pooling(outputs.last_hidden_state, attention_mask)
        return self.projector(pooled)


# =========================================
# Model 3-A: AutoEncoder (이상 탐지)
# =========================================

class KoreanTextAutoEncoder(nn.Module):
    """
    정상 데이터만으로 학습하는 재구성 기반 이상 탐지 모델.
    인코더 → latent → 재구성 벡터를 출력합니다.
    MSE + Cosine 손실로 정상 분포를 학습합니다.
    """

    def __init__(self,
                 model_name: str,
                 latent_dim: int = 256,
                 dropout: float = 0.2):
        super().__init__()
        self.encoder    = load_encoder(model_name)
        hidden_size     = self.encoder.config.hidden_size

        self.projector  = nn.Sequential(
            nn.Linear(hidden_size, latent_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.decoder    = nn.Sequential(
            nn.Linear(latent_dim, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size),
        )

    def forward(self,
                input_ids: torch.Tensor,
                attention_mask: torch.Tensor):
        """
        Returns:
            pooled : (B, H)  원본 CLS 풀링 벡터
            z      : (B, latent_dim) 잠재 표현
            recon  : (B, H)  재구성 벡터
        """
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        pooled  = mean_pooling(outputs.last_hidden_state, attention_mask)
        z       = self.projector(pooled)
        recon   = self.decoder(z)
        return pooled, z, recon


# =========================================
# Model 3-B: Severity MLP (fraud-only 심각도)
# =========================================

class SeverityMLP(nn.Module):
    """
    AE에서 추출한 (anomaly_score, amount_norm) 2-feature 벡터로
    사기 심각도(하/중/상)를 3-class 분류합니다.
    """

    def __init__(self,
                 input_dim: int = 2,
                 hidden_dim: int = 32,
                 dropout: float = 0.1,
                 num_classes: int = 3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
