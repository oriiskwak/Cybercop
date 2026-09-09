"""
Text risk predictor for risk_agent.

The packaged checkpoint predicts the new ``final_risk_level`` labels
(하/중/상) from petition text only. Loading is intentionally lazy because
torch/transformers are heavy optional dependencies for the standalone package.
"""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer

from model_def import KoreanRiskTextClassifier


CHECKPOINT_PATH = (
    Path(__file__).resolve().parent
    / "checkpoints"
    / "rule14b1_final_risk_level_text_classifier_best.pt"
)
HF_MODEL_DIR = Path(__file__).resolve().parent / "hf_model"


def clean_text(text: str) -> str:
    if text is None:
        return ""
    text = str(text)
    text = text.replace("*", " ")
    text = text.replace("amp;", " ")
    text = text.replace("&nbsp;", " ")
    text = text.replace("\n", " ")
    text = text.replace("\t", " ")
    text = re.sub(r"http[s]?://\S+|www\.\S+", " ", text)
    text = re.sub(r"[^가-힣a-zA-Z0-9\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def compute_expected_risk_score(prob_vec: np.ndarray) -> float:
    weights = np.array([0.0, 1.0, 2.0], dtype=np.float32)
    return float((np.asarray(prob_vec, dtype=np.float32) * weights).sum())


class TextRiskPredictor:
    def __init__(
        self,
        model: torch.nn.Module,
        tokenizer: Any,
        max_length: int,
        id2label: dict[int, str],
        target_column: str = "final_risk_level",
        device: str = "cpu",
    ):
        self.model = model.eval()
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.id2label = id2label
        self.target_column = target_column
        self.device = device

    @classmethod
    def from_checkpoint(cls, checkpoint_path: str | Path = CHECKPOINT_PATH, device: str | None = None):
        # This packaged demo is intentionally CPU-first. It avoids CUDA/NCCL
        # runtime mismatches on servers and Windows demo machines.
        device = device or "cpu"
        checkpoint_path = Path(checkpoint_path)
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"텍스트 위험도 모델 파일을 찾을 수 없습니다: {checkpoint_path}")

        ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
        model_name = str(HF_MODEL_DIR) if HF_MODEL_DIR.exists() else ckpt["tokenizer_name"]
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        id2label = ckpt.get("id2label")
        if not id2label:
            class_names = ckpt.get("class_names")
            if class_names:
                id2label = {index: name for index, name in enumerate(class_names)}
            else:
                label_map = ckpt.get("label_map") or {"하": 0, "중": 1, "상": 2}
                id2label = {int(index): name for name, index in label_map.items()}
        id2label = {int(k): v for k, v in id2label.items()}
        model = KoreanRiskTextClassifier(model_name, num_classes=len(id2label))
        model.load_state_dict(ckpt["model_state_dict"])
        model.to(device).eval()

        return cls(
            model=model,
            tokenizer=tokenizer,
            max_length=ckpt.get("max_length", 128),
            id2label=id2label,
            target_column=ckpt.get("target_column", "final_risk_level"),
            device=device,
        )

    @torch.no_grad()
    def predict(self, text: str, amount: float = 0.0) -> dict[str, Any]:
        cleaned = clean_text(text)
        if not cleaned:
            raise ValueError("텍스트 위험도 모델 입력 텍스트가 비어 있습니다.")

        enc = self.tokenizer(
            cleaned,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        input_ids = enc["input_ids"].to(self.device)
        attention_mask = enc["attention_mask"].to(self.device)

        logits = self.model(input_ids, attention_mask)
        prob = F.softmax(logits, dim=-1).cpu().numpy()[0]
        pred_id = int(prob.argmax())
        label = self.id2label.get(pred_id, str(pred_id))
        score = compute_expected_risk_score(prob)

        return {
            "available": True,
            "model": "rule14b1_final_risk_level_text_classifier",
            "target_column": self.target_column,
            "pred_label": label,
            "pred_label_id": pred_id,
            "risk_score": round(score, 6),
            "risk_score_scale": "0~2",
            "predicted_risk_percentile": None,
            "risk_percentile_top": None,
            "input_amount": float(amount or 0.0),
            "amount_used_by_model": False,
            "probabilities": {
                self.id2label.get(i, str(i)): round(float(p), 6)
                for i, p in enumerate(prob)
            },
            "input_text": cleaned,
        }


_PREDICTOR: TextRiskPredictor | None = None
_LOAD_ERROR: str | None = None


def predict_text_risk(text: str, amount: float = 0.0) -> dict[str, Any]:
    global _PREDICTOR, _LOAD_ERROR
    if _LOAD_ERROR:
        return {"available": False, "error": _LOAD_ERROR, "input_text": clean_text(text)}
    try:
        if _PREDICTOR is None:
            _PREDICTOR = TextRiskPredictor.from_checkpoint()
        return _PREDICTOR.predict(text, amount)
    except Exception as exc:
        _LOAD_ERROR = str(exc)
        return {"available": False, "error": _LOAD_ERROR, "input_text": clean_text(text)}
