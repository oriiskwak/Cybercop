"""
risk_utils.py
- 피해금액 문자열 파싱 및 위험도 산출 유틸리티.
- risk_api.py에서 import하여 사용한다.
"""
import re


RISK_WEIGHTS = {
    "S1": 0.25,  # 피해 금액
    "S2": 0.35,  # 피해자 수
    "S3": 0.11,  # 대포계정/대포통장 사용 징후
    "S4": 0.08,  # 추가 유인 징후
    "S5": 0.17,  # 조직적 범행 징후
    "S6": 0.04,  # 가상자산 거래 징후
}


QUALITATIVE_SIGNAL_DEFS = {
    "S3": {
        "name": "대포계정/대포통장 사용 징후",
        "short_name": "대포계정/통장",
        "weight": RISK_WEIGHTS["S3"],
        "scenario": "계좌 명의인과 판매자명이 다르거나, 접수 직후 계정 폐쇄/외국인 명의 계좌 등 대포계정 정황이 확인되면 위험도가 상승합니다.",
    },
    "S4": {
        "name": "추가 유인 징후",
        "short_name": "추가 입금 유도",
        "weight": RISK_WEIGHTS["S4"],
        "scenario": "수수료, 세금, 보증금, 환전 조건 등 명목으로 추가 입금을 요구했거나 실제 추가 입금이 발생하면 위험도가 상승합니다.",
    },
    "S5": {
        "name": "조직적 범행 징후",
        "short_name": "집단/조직적 사기",
        "weight": RISK_WEIGHTS["S5"],
        "scenario": "스크립트형 멘트, 상담원/관리자 역할 분담, 외부 메신저 유도 등이 함께 확인되면 고위험으로 재평가될 수 있습니다.",
    },
    "S6": {
        "name": "가상자산 거래 징후",
        "short_name": "가상자산 거래",
        "weight": RISK_WEIGHTS["S6"],
        "scenario": "지갑주소, USDT/BTC 등 코인명, 거래소 이체 또는 입금 후 코인화 정황이 확인되면 위험도가 상승합니다.",
    },
}


def parse_money_to_won(text: str) -> int | None:
    """'3억 2천만', '1,500,000원' 등 다양한 표기의 피해금액을 원(KRW) 정수로 변환."""
    if not text:
        return None

    s = str(text).strip()
    if not s or s in ("-", "미상", "없음", "unknown", "Unknown"):
        return None

    s = s.replace(",", "").replace(" ", "")
    s = s.replace("원", "")

    if re.fullmatch(r"\d+", s):
        return int(s)

    total = 0

    def _grab(unit_char: str, unit_value: int) -> int:
        nonlocal s
        if unit_char not in s:
            return 0
        parts = s.split(unit_char, 1)
        left = parts[0]
        right = parts[1]
        if not re.fullmatch(r"\d+", left):
            return 0
        s = right
        return int(left) * unit_value

    total += _grab("억", 100_000_000)

    if "천만" in s:
        parts = s.split("천만", 1)
        if re.fullmatch(r"\d+", parts[0]):
            total += int(parts[0]) * 10_000_000
            s = parts[1]

    if "만" in s:
        parts = s.split("만", 1)
        if re.fullmatch(r"\d+", parts[0]):
            total += int(parts[0]) * 10_000
            s = parts[1]

    if re.fullmatch(r"\d+", s):
        total += int(s)

    return total if total > 0 else None


def _score_loss(total_loss_won: int) -> int:
    """가이드라인 S1 피해금액 구간점수(0~100)."""
    amount = max(int(total_loss_won or 0), 0)
    if amount < 1_000_000:
        return 10
    if amount < 5_000_000:
        return 30
    if amount < 20_000_000:
        return 60
    if amount < 100_000_000:
        return 80
    return 100


def _score_victim_count(victim_count: int) -> int:
    """가이드라인 S2 피해자 수 구간점수(0~100)."""
    count = max(int(victim_count or 0), 0)
    if count <= 0:
        return 0
    if count == 1:
        return 10
    if count <= 4:
        return 40
    if count <= 9:
        return 70
    return 100


def _risk_label(score: float, qualitative_scores: dict[str, int]) -> str:
    """가이드라인 threshold + 보수적 분류 규칙."""
    if score >= 70:
        return "high"
    if qualitative_scores.get("S5") == 100 or qualitative_scores.get("S6") == 100:
        return "high"
    if score >= 40:
        return "medium"
    if any(v >= 100 for v in qualitative_scores.values()):
        return "medium"
    if qualitative_scores.get("S3", 0) >= 50 or qualitative_scores.get("S4", 0) >= 50:
        return "medium"
    return "low"


def _score_to_band(score: int) -> str:
    if score >= 70:
        return "high"
    if score >= 40:
        return "medium"
    return "low"


def compute_risk(
    victim_count: int,
    total_loss_won: int,
    qualitative_scores: dict[str, int] | None = None,
    qualitative_evidence: dict[str, list[str]] | None = None,
) -> dict:
    """가이드라인 v0.4 기반 1차 위험도 점수와 등급을 계산."""
    q_scores = {k: 0 for k in QUALITATIVE_SIGNAL_DEFS}
    for key, val in (qualitative_scores or {}).items():
        if key in q_scores:
            q_scores[key] = max(0, min(int(val or 0), 100))
    q_evidence = {k: list((qualitative_evidence or {}).get(k, [])) for k in QUALITATIVE_SIGNAL_DEFS}

    s1_score = _score_loss(total_loss_won)
    s2_score = _score_victim_count(victim_count)

    indicator_scores = {
        "S1": {
            "name": "피해 금액",
            "score": s1_score,
            "weight": RISK_WEIGHTS["S1"],
            "weighted_score": round(s1_score * RISK_WEIGHTS["S1"], 2),
            "basis": f"총 피해금액 {int(total_loss_won or 0):,}원",
            "risk_band": _score_to_band(s1_score),
        },
        "S2": {
            "name": "피해자 수",
            "score": s2_score,
            "weight": RISK_WEIGHTS["S2"],
            "weighted_score": round(s2_score * RISK_WEIGHTS["S2"], 2),
            "basis": f"피해자 {int(victim_count or 0)}명",
            "risk_band": _score_to_band(s2_score),
        },
    }
    for key, spec in QUALITATIVE_SIGNAL_DEFS.items():
        score = q_scores[key]
        indicator_scores[key] = {
            "name": spec["name"],
            "short_name": spec["short_name"],
            "score": score,
            "weight": spec["weight"],
            "weighted_score": round(score * spec["weight"], 2),
            "evidence": q_evidence[key],
            "status": "confirmed" if score == 100 else ("suspected" if score == 50 else "not_detected"),
            "scenario": spec["scenario"],
            "potential_additional_score": round((100 - score) * spec["weight"], 2),
        }

    final_score = round(sum(v["weighted_score"] for v in indicator_scores.values()), 2)
    final_risk = _risk_label(final_score, q_scores)
    quantitative_score = round(
        indicator_scores["S1"]["weighted_score"] + indicator_scores["S2"]["weighted_score"], 2
    )
    qualitative_score = round(final_score - quantitative_score, 2)

    scenarios = []
    for key in ("S3", "S4", "S5", "S6"):
        cur = q_scores[key]
        if cur >= 100:
            continue
        scenario_score = round(final_score + (100 - cur) * RISK_WEIGHTS[key], 2)
        scenario_q = dict(q_scores)
        scenario_q[key] = 100
        scenario_risk = _risk_label(scenario_score, scenario_q)
        scenarios.append({
            "code": key,
            "name": QUALITATIVE_SIGNAL_DEFS[key]["name"],
            "short_name": QUALITATIVE_SIGNAL_DEFS[key]["short_name"],
            "current_score": cur,
            "scenario_score": scenario_score,
            "scenario_risk": scenario_risk,
            "delta": round(scenario_score - final_score, 2),
            "description": QUALITATIVE_SIGNAL_DEFS[key]["scenario"],
        })

    return {
        "victim_count": victim_count,
        "total_loss_won": total_loss_won,
        "score": final_score,
        "final_score": final_score,
        "max_score": 100,
        "quantitative_score": quantitative_score,
        "qualitative_score": qualitative_score,
        "quantitative_detectable_max": 60,
        "qualitative_detectable_max": 40,
        "indicator_scores": indicator_scores,
        "scenario_upgrades": scenarios,
        # 기존 프론트/호출부 호환 필드
        "risk_by_victim_count": indicator_scores["S2"]["risk_band"],
        "risk_by_total_loss": indicator_scores["S1"]["risk_band"],
        "final_risk": final_risk,
    }
