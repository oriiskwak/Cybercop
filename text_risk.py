"""
text_risk.py — Cybercop 분석 결과를 risk_agent 위험도/사기유형 분류 모델에 연결.

risk_agent는 원래 "진정서 metadata"를 입력받아 텍스트 기반 위험도(하/중/상, S1~S6 가이드라인 점수)와
사기유형을 계산하도록 만들어진 코드다. Cybercop은 이미 완성된 이 파이프라인을 그대로 재사용한다 —
자체 RAG(범죄유형 매칭)를 다시 만들지 않고, Cybercop 결과(classify_summary, objects, scam_evidence,
ocr_after)에서 "진술 텍스트"를 재구성해 risk_agent의 모델에 넘긴다.

코드는 저장소의 `agents/` 아래에 포함되어 있고, 체크포인트(.pt) 두 개만 크기 제한 때문에 저장소
바깥에 있다 — `python download_models.py`로 받으면 된다. 체크포인트가 없어도 사기유형 분류(키워드
기반, 모델 불필요)는 동작하고 위험도만 available: false 로 반환된다.

risk_agent 쪽 예측 함수(predict_text_risk, predict_guideline_multitask_risk,
classify_crime_from_metagraph)는 전부 내부에서 CUDA_VISIBLE_DEVICES=""를 강제하는 CPU 전용 함수라
GPU 점유 여부와 무관하게 동작한다. 모델은 각 함수 내부에서 lazy singleton으로 캐싱되므로, 배치
처리 중에는 최초 1건에서만 로드된다.

의존성: risk_api.py를 import하려면 fastapi/uvicorn/python-multipart/pydantic 외에 sse-starlette가
필요하다(requirements.txt에 포함됨). risk_api.py는 import만 하고 uvicorn.run()은 실행하지 않는다
(그 호출은 `if __name__ == "__main__":` 가드 안에 있음).
"""
from __future__ import annotations

import json
import os
import re
import sys
from pathlib import Path
from typing import Any

BASE_DIR = Path(__file__).resolve().parent

# 기본값은 저장소 자신 — `agents/` 가 여기 포함되어 있다. 외부 risk_agent_package를 쓰고 싶으면
# RISK_AGENT_DIR 환경변수로 그 경로를 지정하면 된다(해당 경로 아래 agents/ 를 찾는다).
RISK_AGENT_DIR = Path(os.getenv("RISK_AGENT_DIR", str(BASE_DIR)))

# ──────────────────────────────────────────────
# risk_agent 함수 지연 로딩 (모델 로드는 최초 assess_risk() 호출 시점에만)
# ──────────────────────────────────────────────
_predict_text_risk = None
_predict_guideline_multitask_risk = None
_classify_crime_from_metagraph = None
_parse_money_to_won = None
_LOAD_ERROR: str | None = None


def _ensure_loaded() -> None:
    global _predict_text_risk, _predict_guideline_multitask_risk
    global _classify_crime_from_metagraph, _parse_money_to_won, _LOAD_ERROR

    if _predict_text_risk is not None or _LOAD_ERROR is not None:
        return

    try:
        text_risk_dir = RISK_AGENT_DIR / "agents" / "text_risk_model"
        if str(text_risk_dir) not in sys.path:
            sys.path.insert(0, str(text_risk_dir))
        from predictor import predict_text_risk

        guideline_dir = RISK_AGENT_DIR / "agents" / "guideline_multitask_model"
        if str(guideline_dir) not in sys.path:
            sys.path.insert(0, str(guideline_dir))
        from guideline_predictor import predict_guideline_multitask_risk

        risk_agent_dir = RISK_AGENT_DIR / "agents" / "risk_agent"
        if str(risk_agent_dir) not in sys.path:
            sys.path.insert(0, str(risk_agent_dir))
        from risk_api import classify_crime_from_metagraph
        from risk_utils import parse_money_to_won

        _predict_text_risk = predict_text_risk
        _predict_guideline_multitask_risk = predict_guideline_multitask_risk
        _classify_crime_from_metagraph = classify_crime_from_metagraph
        _parse_money_to_won = parse_money_to_won
    except Exception as exc:
        _LOAD_ERROR = (
            f"risk_agent 모델 로드 실패 ({RISK_AGENT_DIR}): {exc}. "
            "RISK_AGENT_DIR 환경변수 또는 risk_agent_package 설치 상태를 확인하세요."
        )


# ──────────────────────────────────────────────
# 진술 텍스트 빌더
# ──────────────────────────────────────────────
def _general_objects(payload: dict) -> list[str]:
    """payload["objects"]는 Cybercop 안에서 이미 scam_evidence ∪ 일반객체로 합쳐져 있다
    (cybercop_pipeline_AdotX_v0_4.py의 objects_out). scam_evidence는 별도 필드로 이미 넘기므로,
    여기서는 겹치는 항목을 빼고 순수 일반객체만 돌려준다 — 안 그러면 같은 단어가 진술 텍스트/hints에
    두 번(탐지된 객체 + 사기 증거) 들어가 그만큼 가중치가 실질적으로 두 배가 된다."""
    objects = payload.get("objects") or []
    scam_evidence = set(payload.get("scam_evidence") or [])
    return [o for o in objects if o not in scam_evidence]


def build_statement_text(payload: dict) -> str:
    """Cybercop 결과 dict에서 risk_agent 모델 입력용 진술 텍스트를 구성한다.

    risk_api.py의 _build_statement_text()와 같은 접근(여러 필드를 한국어 문장 조각으로 이어붙임)을
    Cybercop 스키마(classify_summary/objects/scam_evidence/ocr_after)에 맞게 재작성.

    cls_label/final_label(우리 자체 판정)은 의도적으로 제외한다 — assess_risk()는 final_label이
    "사기"/"검토필요"일 때만 호출되므로 이 값은 항상 같은 결과("의심됨")만 반복해 정보량이 없고,
    S3~S6가 실제 증거 대신 판정 문구 자체에 끌려 편향될 위험이 있다. 순수 증거만 넘긴다.
    """
    parts: list[str] = []

    classify_summary = payload.get("classify_summary")
    if classify_summary:
        parts.append(str(classify_summary).strip())

    general_objects = _general_objects(payload)
    if general_objects:
        parts.append("탐지된 객체 " + ", ".join(general_objects))

    scam_evidence = payload.get("scam_evidence") or []
    if scam_evidence:
        parts.append("사기 증거 " + ", ".join(str(e) for e in scam_evidence))

    ocr_after = payload.get("ocr_after")
    if ocr_after:
        parts.append(str(ocr_after).strip())

    return "\n".join(p for p in parts if p)


# ──────────────────────────────────────────────
# 금액 보조 추출 (best-effort)
# ──────────────────────────────────────────────
_AMOUNT_CANDIDATE_RE = re.compile(
    r"\d[\d,]*\s*억(?:\s*\d[\d,]*\s*천만)?(?:\s*\d[\d,]*\s*만)?\s*원?"
    r"|\d[\d,]*\s*천만\s*원?"
    r"|\d[\d,]*\s*만\s*원?"
    r"|\d[\d,]{5,}\s*원"
)


def extract_amount_won(text: str) -> int:
    """OCR 텍스트 등에서 금액 패턴 후보를 찾아 parse_money_to_won()으로 파싱, 최댓값을 반환.
    후보가 없거나 전부 파싱 실패하면 0."""
    _ensure_loaded()
    if _LOAD_ERROR or not text:
        return 0
    best = 0
    for match in _AMOUNT_CANDIDATE_RE.finditer(text):
        won = _parse_money_to_won(match.group())
        if won and won > best:
            best = won
    return best


# ──────────────────────────────────────────────
# 공개 API
# ──────────────────────────────────────────────
def assess_risk(
    payload: dict,
    victim_count: int | None = None,
    total_loss_won: int | None = None,
) -> dict[str, Any]:
    """Cybercop 결과 dict(payload)에서 진술 텍스트를 구성해 risk_agent의 위험도/사기유형
    분류를 실행하고, risk_assessment 서브 dict를 반환한다.

    victim_count/total_loss_won은 Cybercop 결과에 없는 정보라 억지로 채우지 않는다 — None이면
    각각 0(피해자 수) / ocr_after에서 추출 시도(금액)로 처리하고, 모델은 0을 "정보 없음" 최소
    베이스라인으로 이미 처리한다. 명시적으로 넘기면(수동 override) 그 값을 그대로 쓴다.
    """
    _ensure_loaded()
    if _LOAD_ERROR:
        return {"available": False, "error": _LOAD_ERROR}

    statement_text = build_statement_text(payload)

    if total_loss_won is not None:
        amount = total_loss_won
        amount_source = "manual"
    else:
        amount = extract_amount_won(payload.get("ocr_after") or "")
        amount_source = "ocr_extracted" if amount else "none"

    victims = victim_count if victim_count is not None else 0

    text_risk = _predict_text_risk(statement_text, amount)
    guideline_risk = _predict_guideline_multitask_risk(statement_text, victims, amount)

    hints: list[str] = []
    if payload.get("classify_summary"):
        hints.append(str(payload["classify_summary"]))
    hints.extend(_general_objects(payload))
    hints.extend(str(e) for e in (payload.get("scam_evidence") or []))
    if payload.get("ocr_after"):
        hints.append(str(payload["ocr_after"]))
    # entities/relations는 그래프 데이터가 없으므로 빈 리스트 — hints만 키워드/별칭 매칭에 쓰임.
    crime_classification = _classify_crime_from_metagraph(entities=[], relations=[], hints=hints)

    return {
        "available": True,
        "statement_text": statement_text,
        "victim_count_used": victims,
        "total_loss_won_used": amount,
        "amount_source": amount_source,
        "text_risk": text_risk,
        "guideline_risk": guideline_risk,
        "crime_classification": crime_classification,
    }


# ──────────────────────────────────────────────
# 독립 실행 (디버깅용)
# ──────────────────────────────────────────────
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Cybercop 결과 JSON 하나를 risk_agent 모델에 넣어본다 (디버깅용).")
    parser.add_argument("--result", required=True, help="Cybercop 결과 JSON 파일 경로")
    parser.add_argument("--victim_count", type=int, default=None)
    parser.add_argument("--total_loss_won", type=int, default=None)
    args = parser.parse_args()

    with open(args.result, encoding="utf-8") as f:
        result = json.load(f)

    out = assess_risk(result, victim_count=args.victim_count, total_loss_won=args.total_loss_won)
    print(json.dumps(out, ensure_ascii=False, indent=2))
