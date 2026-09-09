"""S1/S2 raw inputs + S3~S6 text predictions + guideline aggregation."""

from __future__ import annotations

import itertools
import os
import re
from pathlib import Path
from typing import Any

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer

from guideline_model_def import GuidelineMultiTaskClassifier


MODEL_DIR = Path(__file__).resolve().parent
CHECKPOINT_PATH = MODEL_DIR / "checkpoints" / "guideline_multitask_best.pt"
HF_MODEL_DIR = MODEL_DIR.parent / "text_risk_model" / "hf_model"
DEFAULT_STATUS_COLUMNS = ["S3_status", "S4_status", "S5_status", "S6_status"]
DEFAULT_STATUS_NAMES = ["not_observed", "explicitly_negative", "suspected", "confirmed"]
STATUS_KO = {
    "not_observed": "미관측",
    "explicitly_negative": "명시적 부정",
    "suspected": "의심",
    "confirmed": "확인",
}
RISK_NAMES = ["하", "중", "상"]
RISK_EN = ["low", "medium", "high"]
INDICATOR_WEIGHTS = {"S1": 0.25, "S2": 0.35, "S3": 0.11, "S4": 0.08, "S5": 0.17, "S6": 0.04}


def clean_text(text: str) -> str:
    if text is None:
        return ""
    value = str(text).replace("*", " ").replace("amp;", " ").replace("&nbsp;", " ")
    value = value.replace("\n", " ").replace("\t", " ")
    value = re.sub(r"http[s]?://\S+|www\.\S+", " ", value)
    value = re.sub(r"[^가-힣a-zA-Z0-9\s]", " ", value)
    return re.sub(r"\s+", " ", value).strip()


def amount_to_s1(value: int | float | None) -> int:
    if value is None:
        return 0
    amount = max(float(value), 0.0)
    if amount < 1_000_000:
        return 10
    if amount < 5_000_000:
        return 30
    if amount < 20_000_000:
        return 60
    if amount < 100_000_000:
        return 80
    return 100


def victims_to_s2(value: int | float | None) -> int:
    count = max(int(value or 0), 0)
    if count <= 0:
        return 0
    if count == 1:
        return 10
    if count <= 4:
        return 40
    if count <= 9:
        return 70
    return 100


def score_to_band(score: int) -> str:
    if score >= 70:
        return "high"
    if score >= 40:
        return "medium"
    return "low"


def apply_guideline(s1: int, s2: int, qualitative_scores: list[int]) -> tuple[float, int, list[str]]:
    s3, s4, s5, s6 = qualitative_scores
    score = round(0.25 * s1 + 0.35 * s2 + 0.11 * s3 + 0.08 * s4 + 0.17 * s5 + 0.04 * s6, 2)
    level = 2 if score >= 70 else (1 if score >= 40 else 0)
    overrides: list[str] = []
    if s5 == 100 or s6 == 100:
        level = 2
        overrides.append("S5 또는 S6 확인으로 상향")
    if any(v == 100 for v in qualitative_scores) and level < 1:
        level = 1
        overrides.append("S3~S6 확인 지표로 최소 중위험")
    if (s3 >= 50 or s4 >= 50) and level < 1:
        level = 1
        overrides.append("S3 또는 S4 의심 이상으로 최소 중위험")
    return score, level, overrides


def aggregate_level_probabilities(
    s1: int,
    s2: int,
    head_probabilities: np.ndarray,
    status_scores: list[int],
) -> list[float]:
    """네 head를 조건부 독립으로 보고 모든 4^4 조합의 등급 확률을 합산."""
    output = np.zeros(3, dtype=np.float64)
    for combo in itertools.product(range(4), repeat=4):
        probability = float(np.prod([head_probabilities[h, c] for h, c in enumerate(combo)]))
        scores = [int(status_scores[c]) for c in combo]
        _, level, _ = apply_guideline(s1, s2, scores)
        output[level] += probability
    total = float(output.sum())
    if total > 0:
        output /= total
    return [round(float(x), 6) for x in output]


class GuidelineMultiTaskPredictor:
    def __init__(self, model, tokenizer, checkpoint: dict[str, Any], device: str):
        self.model = model.eval()
        self.tokenizer = tokenizer
        self.max_length = int(checkpoint.get("max_length", 128))
        self.status_columns = checkpoint.get("status_columns", DEFAULT_STATUS_COLUMNS)
        self.status_names = checkpoint.get("status_names", DEFAULT_STATUS_NAMES)
        score_map = (checkpoint.get("guideline") or {}).get("status_scores") or {
            "not_observed": 0, "explicitly_negative": 0, "suspected": 50, "confirmed": 100,
        }
        self.status_scores = [int(score_map[name]) for name in self.status_names]
        self.device = device

    @classmethod
    def from_checkpoint(cls, checkpoint_path: str | Path = CHECKPOINT_PATH, device: str = "cpu"):
        checkpoint_path = Path(checkpoint_path)
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"S1~S6 결합 모델 파일을 찾을 수 없습니다: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        model_name = str(HF_MODEL_DIR) if HF_MODEL_DIR.exists() else checkpoint["tokenizer_name"]
        status_columns = checkpoint.get("status_columns", DEFAULT_STATUS_COLUMNS)
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = GuidelineMultiTaskClassifier(model_name, status_columns=status_columns)
        model.load_state_dict(checkpoint["model_state_dict"])
        model.to(device).eval()
        return cls(model, tokenizer, checkpoint, device)

    @torch.no_grad()
    def predict(self, text: str, victim_count: int, total_loss_won: int) -> dict[str, Any]:
        cleaned = clean_text(text)
        if not cleaned:
            raise ValueError("S1~S6 결합 모델 입력 텍스트가 비어 있습니다.")
        encoded = self.tokenizer(
            cleaned, padding=True, truncation=True,
            max_length=self.max_length, return_tensors="pt",
        )
        logits = self.model(
            encoded["input_ids"].to(self.device), encoded["attention_mask"].to(self.device)
        )
        probabilities = F.softmax(logits, dim=-1).cpu().numpy()[0]
        predicted_ids = probabilities.argmax(axis=1)
        s1, s2 = amount_to_s1(total_loss_won), victims_to_s2(victim_count)
        qualitative_scores = [self.status_scores[int(index)] for index in predicted_ids]
        weighted_score, level_id, overrides = apply_guideline(s1, s2, qualitative_scores)
        level_probabilities = aggregate_level_probabilities(
            s1, s2, probabilities, self.status_scores
        )

        indicators: dict[str, Any] = {
            "S1": {
                "name": "피해 금액", "score": s1, "source": "raw_input",
                "weight": INDICATOR_WEIGHTS["S1"],
                "weighted_score": round(s1 * INDICATOR_WEIGHTS["S1"], 2),
                "basis": f"총 피해금액 {int(total_loss_won or 0):,}원",
                "risk_band": score_to_band(s1),
            },
            "S2": {
                "name": "피해자 수", "score": s2, "source": "raw_input",
                "weight": INDICATOR_WEIGHTS["S2"],
                "weighted_score": round(s2 * INDICATOR_WEIGHTS["S2"], 2),
                "basis": f"피해자 {int(victim_count or 0)}명",
                "risk_band": score_to_band(s2),
            },
        }
        names = {
            "S3": "대포계정·대포통장 사용 징후",
            "S4": "추가 유인 징후",
            "S5": "조직적 범행 징후",
            "S6": "가상자산 거래 징후",
        }
        for head_index, column in enumerate(self.status_columns):
            code = column.replace("_status", "")
            predicted_status = self.status_names[int(predicted_ids[head_index])]
            indicators[code] = {
                "name": names[code],
                "score": qualitative_scores[head_index],
                "source": "text_model",
                "weight": INDICATOR_WEIGHTS[code],
                "weighted_score": round(
                    qualitative_scores[head_index] * INDICATOR_WEIGHTS[code], 2
                ),
                "predicted_status": predicted_status,
                "predicted_status_ko": STATUS_KO[predicted_status],
                "basis": f"텍스트 모델 예측 상태: {STATUS_KO[predicted_status]}",
                "probabilities": {
                    status: round(float(probabilities[head_index, class_id]), 6)
                    for class_id, status in enumerate(self.status_names)
                },
            }

        return {
            "available": True,
            "model": "rule14b1_guideline_multitask",
            "model_input": "victim_count + total_loss_won + statement_text",
            "victim_count": int(victim_count or 0),
            "total_loss_won": int(total_loss_won or 0),
            "score": weighted_score,
            "final_score": weighted_score,
            "max_score": 100,
            "weighted_score": weighted_score,
            "signal_based_level": RISK_NAMES[level_id],
            "final_risk_label": RISK_NAMES[level_id],
            "final_risk": RISK_EN[level_id],
            "risk_by_victim_count": score_to_band(s2),
            "risk_by_total_loss": score_to_band(s1),
            "quantitative_score": round(0.25 * s1 + 0.35 * s2, 2),
            "qualitative_score": round(
                0.11 * qualitative_scores[0] + 0.08 * qualitative_scores[1]
                + 0.17 * qualitative_scores[2] + 0.04 * qualitative_scores[3], 2
            ),
            "quantitative_detectable_max": 60,
            "qualitative_detectable_max": 40,
            "level_probabilities": dict(zip(RISK_NAMES, level_probabilities)),
            "applied_overrides": overrides,
            "indicator_scores": indicators,
            "scenario_upgrades": [],
            "input_text": cleaned,
            "scope_note": "S1~S6 신호 기반 등급이며 범죄수법 기준은 포함하지 않습니다.",
        }


_PREDICTOR: GuidelineMultiTaskPredictor | None = None
_LOAD_ERROR: str | None = None


def predict_guideline_multitask_risk(text: str, victim_count: int, total_loss_won: int) -> dict[str, Any]:
    global _PREDICTOR, _LOAD_ERROR
    if _LOAD_ERROR:
        return {"available": False, "error": _LOAD_ERROR, "input_text": clean_text(text)}
    try:
        if _PREDICTOR is None:
            _PREDICTOR = GuidelineMultiTaskPredictor.from_checkpoint()
        return _PREDICTOR.predict(text, victim_count, total_loss_won)
    except Exception as exc:
        _LOAD_ERROR = str(exc)
        return {"available": False, "error": _LOAD_ERROR, "input_text": clean_text(text)}
