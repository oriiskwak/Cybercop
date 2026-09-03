"""
프레임별 Grounding DINO 결과를 영상 단위로 집계.

출력 스키마는 FeedbackMemory / evidence-guided tool augmentation에 넘길 수 있도록 설계.
"""
from collections import defaultdict

# top_labels(=objects/RAG 쿼리로 넘어가는 값)에 포함시킬 최소 신뢰도.
# GroundingDetector의 box_threshold(0.40)보다 높게 잡아서, 탐지 경계선 근처의 애매한
# 라벨("screen" 0.41, "명품 로고" 0.50 같은)이 RAG 쿼리를 오염시키는 걸 막는다.
# 0.5로는 부족했음 — 액체질소 시연 영상(무관한 내용)에서 "오픈채팅방" 0.5011, "휴대폰 화면"
# 0.5373 같은 지속적 오탐(각각 13프레임 중 10, 1개에서 반복 탐지)이 그대로 통과해버림.
# 0.55로 올려서 이 경계선 오탐들을 걸러내고, 대신 VLM 폴백이 대신 작동하도록 함.
# evidence_score(에스컬레이션 판단) 계산에도 동일하게 적용 — 처음엔 "에스컬레이션은
# 놓치는 것보다 과탐지가 낫다"는 원칙으로 여기만 안 걸렀는데, 오히려 그 경계선 오탐들이
# evidence_score를 끌어올려 정상 영상을 검토필요로 잘못 넘기는 문제가 있어서 통일함.
MIN_LABEL_SCORE = 0.55


def aggregate(frame_detections: list[dict], video_duration: float = 0.0) -> dict:
    """
    Args:
        frame_detections: detect_batch() 반환값
            [{"sec": float, "detections": [{"box": ..., "label": str, "score": float}]}]
        video_duration:   영상 전체 길이 (초), 0이면 Early/Late 계산 생략

    Returns:
        {
            "labels": {
                "bank transfer screen": {
                    "count":        int,      # 탐지된 총 인스턴스 수
                    "frame_count":  int,      # 등장 프레임 수
                    "frame_ratio":  float,    # 등장 프레임 / 전체 프레임
                    "first_sec":    float,
                    "last_sec":     float,
                    "segment":      "early" | "late" | "both" | "unknown",
                    "avg_score":    float,
                },
                ...
            },
            "evidence_score":    float,   # 영상 전체 사기 증거 강도 (0~1 정규화)
            "n_detected_frames": int,     # 1개 이상 탐지된 프레임 수
            "top_labels":        list[str],  # avg_score >= MIN_LABEL_SCORE 인 것만, avg_score × frame_ratio 내림차순 상위 5개
        }
    """
    n_frames = len(frame_detections)
    if n_frames == 0:
        return _empty()

    half = video_duration / 2.0

    label_data: dict[str, list] = defaultdict(list)
    for fd in frame_detections:
        for det in fd["detections"]:
            label_data[det["label"]].append({"sec": fd["sec"], "score": det["score"]})

    labels_summary = {}
    for label, instances in label_data.items():
        secs   = [i["sec"]   for i in instances]
        scores = [i["score"] for i in instances]
        unique_frames = len(set(secs))

        if video_duration > 0:
            in_early = any(s <= half for s in secs)
            in_late  = any(s >  half for s in secs)
            segment  = "both" if (in_early and in_late) else ("early" if in_early else "late")
        else:
            segment = "unknown"

        avg_score = sum(scores) / len(scores)
        labels_summary[label] = {
            "count":       len(instances),
            "frame_count": unique_frames,
            "frame_ratio": round(unique_frames / n_frames, 4),
            "first_sec":   round(min(secs), 2),
            "last_sec":    round(max(secs), 2),
            "segment":     segment,
            "avg_score":   round(avg_score, 4),
        }

    # 전체 증거 강도: (가중 탐지 수) / (전체 프레임 수)
    # 한 프레임에 여러 탐지가 있으면 더 강한 신호
    # MIN_LABEL_SCORE 미만(애매한 탐지)은 top_labels와 마찬가지로 여기서도 제외 —
    # 안 그러면 경계선 오탐(예: 액체질소 영상의 "오픈채팅방" 0.50)이 evidence_score를
    # 끌어올려 정상 영상을 검토필요로 잘못 에스컬레이션시킴.
    total_weighted = sum(
        info["count"] * info["avg_score"]
        for info in labels_summary.values()
        if info["avg_score"] >= MIN_LABEL_SCORE
    )
    evidence_score = min(total_weighted / max(n_frames, 1), 1.0)

    n_detected = sum(1 for fd in frame_detections if fd["detections"])

    # 상위 레이블 (avg_score × frame_ratio 기준) — MIN_LABEL_SCORE 미만은 애매한 탐지로 보고 제외
    top_labels = sorted(
        (lb for lb in labels_summary if labels_summary[lb]["avg_score"] >= MIN_LABEL_SCORE),
        key=lambda lb: labels_summary[lb]["avg_score"] * labels_summary[lb]["frame_ratio"],
        reverse=True,
    )[:5]

    return {
        "labels":            labels_summary,
        "evidence_score":    round(evidence_score, 4),
        "n_detected_frames": n_detected,
        "top_labels":        top_labels,
    }


def _empty() -> dict:
    return {
        "labels": {},
        "evidence_score": 0.0,
        "n_detected_frames": 0,
        "top_labels": [],
    }


def should_escalate(agg: dict, threshold: float = 0.15) -> bool:
    """evidence_score가 threshold 이상이면 CoT가 정상으로 판단해도 재검토 대상으로 올림."""
    return agg["evidence_score"] >= threshold
