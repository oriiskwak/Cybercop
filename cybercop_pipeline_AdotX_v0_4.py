"""
cybercop_pipeline_AdotX_v0_4.py — CoT 분류 + VLM closed-set 증거/객체 탐지 + risk_agent 위험도/사기유형.

v0.3 대비 변경점: 자체 RAG(BGE-m3 임베딩 + rag/retrieval_docs_v2.json 기반 범죄유형 매칭)를 제거하고,
그 자리에 이미 완성되어 있는 별도 파이프라인(risk_agent_package)의 위험도(text_risk, S1~S6 가이드라인)와
사기유형 분류를 인라인으로 연결했다 (text_risk.assess_risk() — 자세한 내용은 text_risk.py 참고).
  - CrimeRAG 클래스/rag 필드 삭제. 다만 ObjectMapper(일반객체 427종 closed-set)는 BGE-m3 임베딩
    자체는 계속 써야 해서, CrimeRAG 없이 SentenceTransformer를 직접 로드해 재사용한다.
  - scam_evidence(사기증거 VLM 탐지) + should_escalate() 기반 final_label 에스컬레이션은 v0.3과
    동일하게 유지 (RAG용이 아니라 사기/정상/검토필요 판정 보정용이라 그대로 둠).
v0.2 대비 v0.3 변경점(그대로 유지됨): Grounding DINO(사기증거)와 VLM 자유서술+임베딩매핑(일반객체)을
전부 VLM closed-set 방식으로 교체.
  - 사기증거(36종, pipeline/vocab.py): 프레임마다 VLM한테 어휘 전체를 보여주고 closed-set으로 확인
    (36개는 짧아서 청크 불필요 — 평가 결과 DINO 대비 F1 0%→58.8%)
  - 일반객체(427종, rag/allowed_objects.json): "짧은 장면묘사 → BGE-m3(이미 로드된 ObjectMapper
    재사용)로 top-K(40개) 후보 retrieval → 그 후보만으로 closed-set 확인" 3단계.
    427개를 한 번에/청크로 다 주면 모델이 목록을 그대로 반복 생성하는 할루시네이션이 발생해서
    (평가 데이터 25개 중 다수에서 확인) retrieval로 후보를 줄이는 방식을 채택
    (평가 결과 F1 23.6%→36.5%, 호출 수는 오히려 자유서술 방식과 비슷).
  - Grounding DINO 로딩 자체를 제거해 GPU 메모리도 절약됨.
  - OCR 추출/VLM 교정 단계는 v0.2와 동일하게 유지 (실사용 리포트에 OCR 텍스트가 필요하므로).
이 파일 실행에는 같은 저장소 루트의 pipeline/ 패키지와 text_risk.py가 필요합니다.
"""
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# 새 배포 환경은 최초 실행 때 Hugging Face 모델을 내려받아야 한다. 완전히 캐시된
# 환경에서만 사용자가 명시적으로 HF_HUB_OFFLINE=1을 설정한다.

import sys
import re
import json
import time
import argparse
import csv
from pathlib import Path
from collections import defaultdict

import cv2
import yt_dlp
import numpy as np
import torch
from PIL import Image
from transformers import AutoModelForMultimodalLM, AutoProcessor
from huggingface_hub import snapshot_download
from sentence_transformers import SentenceTransformer

# PaddleOCR에서 RapidOCR로 교체
import onnxruntime as ort
from rapidocr import RapidOCR, EngineType, LangDet, LangRec, ModelType, OCRVersion

# risk_agent 위험도/사기유형 분류 연결 (RAG 대체)
from text_risk import assess_risk

# 새 파이프라인 모듈
from pipeline.frame_sampler import sample_uniform, sample_keyframe
from pipeline.vlm_classify import classify_scam
from pipeline.evidence_aggregator import should_escalate
from pipeline.vocab import SCAM_EVIDENCE_VOCAB  # {한국어: 영어} — v0.3에서는 한국어 키만 closed-set 목록으로 사용

# ──────────────────────────────────────────────
# 설정
# ──────────────────────────────────────────────
BASE_DIR             = Path(__file__).resolve().parent
MODEL_ID             = os.getenv("VLM_MODEL", "google/gemma-4-26B-A4B-it")
DEVICE              = "cuda" if torch.cuda.is_available() else "cpu"
TORCH_DTYPE         = torch.bfloat16 if torch.cuda.is_available() else torch.float32
DEFAULT_EMBED_MODEL = os.getenv("EMBED_MODEL", "BAAI/bge-m3")
ALLOWED_OBJECTS_PATH = Path(os.getenv("ALLOWED_OBJECTS_PATH", str(BASE_DIR / "rag" / "allowed_objects.json")))
MODEL_CACHE_DIR     = Path(os.getenv("MODEL_CACHE_DIR", str(BASE_DIR / "hf_models")))
TRUST_REMOTE_CODE   = os.getenv("HF_TRUST_REMOTE_CODE", "0").lower() in {"1", "true", "yes"}
VLM_GPU_RESERVE_GIB = max(float(os.getenv("VLM_GPU_RESERVE_GIB", "4")), 0.0)

SCAM_EVIDENCE_LABELS = list(SCAM_EVIDENCE_VOCAB.keys())  # 36종
OBJECT_RAG_TOP_K = 40  # 평가 결과 40과 60이 F1 거의 동일(36.5% vs 36.0%)해서 더 짧은 40 채택

# VLM 증거 점수 에스컬레이션 임계값. CoT="정상"이라도 evidence_score가
# 이 값 이상이면 "검토필요"로 표시한다.
ESCALATE_THR = 0.15

# ──────────────────────────────────────────────
# 유틸 함수들
# ──────────────────────────────────────────────
def ensure_dir(path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def format_ts(sec: float) -> str:
    m, s = divmod(int(sec), 60)
    return f"{m:02d}:{s:02d}"


def print_report(final_label: str, reason: str, evidence: dict, risk_assessment: dict | None) -> None:
    """사람이 바로 읽을 수 있는 요약 출력. 결과 JSON 파일 내용/필드는 안 바뀜 — 콘솔 출력만 이 포맷."""
    objs = ", ".join(evidence.get("labels", {}).keys()) or "없음"
    crime_type = "—"
    if risk_assessment and risk_assessment.get("available"):
        crime_type = risk_assessment.get("crime_classification", {}).get("top_crime_type") or "—"
    print(
        f"\n판정: {final_label}\n"
        f"근거 텍스트: {reason}\n"
        f"근거 객체(탐지): {objs}\n"
        f"범죄 유형(risk_agent): {crime_type}\n"
    )


# ──────────────────────────────────────────────
# 모델 로드
# ──────────────────────────────────────────────
def load_vlm(model_id: str):
    if os.path.isdir(model_id):
        local_dir = Path(model_id)
    else:
        safe_name = model_id.replace("/", "_").replace(".", "_")
        local_dir = MODEL_CACHE_DIR / safe_name
        if not (local_dir / "config.json").is_file():
            try:
                snapshot_download(repo_id=model_id, local_dir=str(local_dir))
            except Exception as exc:
                raise RuntimeError(
                    f"VLM 모델을 준비하지 못했습니다: {model_id}. "
                    "네트워크/Hugging Face 접근 권한을 확인하거나 MODEL_CACHE_DIR에 모델을 미리 받으세요."
                ) from exc
    processor = AutoProcessor.from_pretrained(
        str(local_dir), trust_remote_code=TRUST_REMOTE_CODE
    )
    model_kwargs = {
        "dtype": TORCH_DTYPE,
        "device_map": "auto" if DEVICE == "cuda" else "cpu",
        "trust_remote_code": TRUST_REMOTE_CODE,
    }
    if DEVICE == "cuda" and VLM_GPU_RESERVE_GIB:
        reserve_bytes = int(VLM_GPU_RESERVE_GIB * 1024**3)
        model_kwargs["max_memory"] = {
            index: max(torch.cuda.mem_get_info(index)[0] - reserve_bytes, 1024**3)
            for index in range(torch.cuda.device_count())
        }
    model = AutoModelForMultimodalLM.from_pretrained(str(local_dir), **model_kwargs)
    model.eval()
    return processor, model


def load_ocr() -> RapidOCR:
    """PaddleOCR -> RapidOCR"""

    return RapidOCR(params={
        "EngineConfig.onnxruntime.intra_op_num_threads": 4,
        "EngineConfig.onnxruntime.inter_op_num_threads": 1,
        "EngineConfig.onnxruntime.use_cuda": False,
        "Det.engine_type": EngineType.ONNXRUNTIME,
        "Det.lang_type": LangDet.CH,          # det는 언어 무관, ch 단일 모델
        "Det.model_type": ModelType.MOBILE,
        "Det.ocr_version": OCRVersion.PPOCRV5,
        "Rec.engine_type": EngineType.ONNXRUNTIME,
        "Rec.lang_type": LangRec.KOREAN,
        "Rec.model_type": ModelType.MOBILE,
        "Rec.ocr_version": OCRVersion.PPOCRV5,
    })


class ObjectMapper:
    """VLM 폴백(vlm_object_fallback)이 자유롭게 뱉은 객체명을 rag/allowed_objects.json의
    고정 어휘(427종)로 정규화. 나중에 정확도 검증/분석할 때 열린 어휘가 아니라 고정된 클래스
    기준으로 볼 수 있게 하기 위함 — DINO의 top_labels(SCAM_EVIDENCE_VOCAB)에는 적용하지 않음
    (성격이 다른 별개 어휘라 섞으면 오히려 의미가 흐려짐)."""

    def __init__(self, embed_model: SentenceTransformer, threshold: float = 0.7):
        # 0.6이면 "금고"→"망고"(0.63), "저장"→"컴퓨터"(0.62), "이모티콘"→"리모컨"(0.62) 처럼
        # 어휘에 없는 개념이 무관한 단어로 강제 매핑됨. 0.7로 올리면 확인된 오매핑은 다 걸러지고
        # "앱"→"텔레그램 앱"(0.73), "확인"→"인증"(0.77) 같은 정상 매핑은 유지됨.
        self.model = embed_model
        self.threshold = threshold
        with open(ALLOWED_OBJECTS_PATH, encoding="utf-8") as f:
            self.allowed = json.load(f)
        self.embeddings = self.model.encode(
            self.allowed, convert_to_numpy=True, normalize_embeddings=True
        ).astype("float32")

    def map(self, candidates: list) -> list:
        if not candidates:
            return []
        cand_embs = self.model.encode(
            candidates, convert_to_numpy=True, normalize_embeddings=True
        ).astype("float32")
        scores = cand_embs @ self.embeddings.T
        result, seen = [], set()
        for cand_scores in scores:
            best_idx = int(cand_scores.argmax())
            best_score = float(cand_scores[best_idx])
            if best_score >= self.threshold:
                mapped = self.allowed[best_idx]
                if mapped not in seen:
                    seen.add(mapped)
                    result.append(mapped)
        return result


# ──────────────────────────────────────────────
# OCR
# ──────────────────────────────────────────────
def _merge_same_line(lines, y_tol=10):
    """같은 y대의 라인들을 좌→우로 병합. 입력/출력 모두 {text, score, bbox} 딕셔너리."""
    if not lines:
        return []

    lines = sorted(lines, key=lambda d: d["bbox"][1])   # ymin 기준

    merged, cur = [], [lines[0]]
    for line in lines[1:]:
        if abs(line["bbox"][1] - cur[-1]["bbox"][1]) < y_tol:
            cur.append(line)
        else:
            merged.append(cur)
            cur = [line]
    merged.append(cur)

    result = []
    for grp in merged:
        grp.sort(key=lambda d: d["bbox"][0])            # xmin 기준 좌→우
        result.append({
            "text": " ".join(d["text"] for d in grp),
            "score": sum(d["score"] for d in grp) / len(grp),
            "bbox": [
                min(d["bbox"][0] for d in grp),
                min(d["bbox"][1] for d in grp),
                max(d["bbox"][2] for d in grp),
                max(d["bbox"][3] for d in grp),
            ],
        })
    return result


# ──────────────────────────────────────────────
# OCR 프레임간 클러스터링 (HS/cybercop_pipeline_shared/cybercop_pipeline_latest.py 포팅)
# 같은 UI 텍스트가 여러 프레임에서 OCR 오탈자로 조금씩 다르게 읽히는 걸
# IoU(위치)+CER(문자열 유사도)로 묶어 다수결 대표값을 뽑는다.
# 호가창 가격처럼 실제로 매 프레임 바뀌는 숫자값은 클러스터링 대상에서 제외한다.
# ──────────────────────────────────────────────
IOU_THRESHOLD = 0.7
CER_THRESHOLD = 0.5
MAX_FRAME_GAP = 30
TOP_N_CANDIDATES = 3
DOMINANT_SHARE_THRESHOLD = 0.5

NUMERIC_ONLY_RE = re.compile(r"^[\d.,:%+\-]+$")


def is_numeric_only(text: str) -> bool:
    return bool(NUMERIC_ONLY_RE.match(text))


def _levenshtein(a: str, b: str) -> int:
    if a == b:
        return 0
    m, n = len(a), len(b)
    if m == 0:
        return n
    if n == 0:
        return m
    dp = list(range(n + 1))
    for i in range(1, m + 1):
        prev, dp[0] = dp[0], i
        for j in range(1, n + 1):
            prev, dp[j] = dp[j], prev if a[i - 1] == b[j - 1] else 1 + min(prev, dp[j], dp[j - 1])
    return dp[n]


def _cer(a: str, b: str) -> float:
    if not a and not b:
        return 0.0
    return _levenshtein(a, b) / max(len(a), len(b), 1)


def _iou(box_a, box_b) -> float:
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


class _UnionFind:
    def __init__(self, n):
        self.parent = list(range(n))

    def find(self, x):
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[ra] = rb

def run_ocr_on_frame(engine: RapidOCR, image: Image.Image) -> list: # RapidOCR 버전
    """PIL(RGB) 프레임 하나에 대해 OCR을 실행, 라인 단위 검출 결과를 반환한다."""
    bgr = np.array(image.convert("RGB"))[:, :, ::-1]

    # bgr = correct_orientation(bgr)

    res = engine(bgr)
    if res is None or res.txts is None:
        return []

    lines = []
    for text, score, box in zip(res.txts, res.scores, res.boxes):
        if not text:
            continue
        pts = np.asarray(box, dtype=np.float32).reshape(-1, 2)
        bbox = [
            float(pts[:, 0].min()),   # x1
            float(pts[:, 1].min()),   # y1
            float(pts[:, 0].max()),   # x2
            float(pts[:, 1].max()),   # y2
        ]
        lines.append({"text": text, "score": float(score), "bbox": bbox})
    return lines

def cluster_ocr_detections(detections: list) -> list:
    """서로 다른 프레임의 검출 중 bbox가 겹치고(IoU) 텍스트도 비슷한(CER) 것들을
    동일 텍스트 인스턴스로 간주해 Union-Find로 묶는다. 숫자로만 이루어진 값은 클러스터링하지 않는다."""
    n = len(detections)
    uf = _UnionFind(n)
    for i in range(n):
        if is_numeric_only(detections[i]["text"]):
            continue
        for j in range(i + 1, n):
            if detections[j]["frame_idx"] - detections[i]["frame_idx"] > MAX_FRAME_GAP:
                break
            if detections[i]["frame_idx"] == detections[j]["frame_idx"]:
                continue
            if is_numeric_only(detections[j]["text"]):
                continue
            if _iou(detections[i]["bbox"], detections[j]["bbox"]) < IOU_THRESHOLD:
                continue
            if _cer(detections[i]["text"], detections[j]["text"]) > CER_THRESHOLD:
                continue
            uf.union(i, j)

    clusters = defaultdict(list)
    for i in range(n):
        clusters[uf.find(i)].append(detections[i])
    return list(clusters.values())


def merge_ocr_cluster(cluster: list) -> dict:
    """클러스터 내 텍스트 변형별 득표수(count) 기준 top-N 후보 랭킹."""
    variant_dets = defaultdict(list)
    for d in cluster:
        variant_dets[d["text"]].append(d)

    variant_stats = []
    for text, dets in variant_dets.items():
        scores = [d["score"] for d in dets]
        variant_stats.append({
            "text": text,
            "count": len(dets),
            "avg_score": round(sum(scores) / len(scores), 4),
        })
    variant_stats.sort(key=lambda v: (v["count"], v["avg_score"]), reverse=True)

    total = len(cluster)
    dominant_share = variant_stats[0]["count"] / total
    frames = sorted(d["frame_idx"] for d in cluster)
    secs = sorted(d["sec"] for d in cluster)
    avg_bbox = [sum(d["bbox"][k] for d in cluster) / len(cluster) for k in range(4)]
    return {
        "candidates": variant_stats[:TOP_N_CANDIDATES],
        "dominant_share": round(dominant_share, 4),
        "stable": dominant_share >= DOMINANT_SHARE_THRESHOLD,
        "num_frames": len(frames),
        "frame_range": [frames[0], frames[-1]],
        "start": format_ts(secs[0]),
        "end": format_ts(secs[-1]),
        "bbox": [round(v, 1) for v in avg_bbox],
    }


def _runs_by_identical_text(cluster: list) -> list:
    """다수결이 수렴하지 않는(stable=False) 클러스터에서, 실제로 값이 바뀌는 상황을
    하나의 대표값으로 덮어쓰지 않고 프레임 순서대로 동일 텍스트 구간만 묶어 보존한다."""
    members = sorted(cluster, key=lambda d: d["frame_idx"])
    runs = []
    for d in members:
        if runs and runs[-1]["text"] == d["text"]:
            runs[-1]["end_sec"] = d["sec"]
        else:
            runs.append({"text": d["text"], "start_sec": d["sec"], "end_sec": d["sec"]})
    return runs


def build_ocr_outputs(detections: list) -> tuple:
    """검출 리스트로부터 (1) 기존 호환 ocr 스팬 리스트 [{start,end,text}],
    (2) 클러스터별 상세 정보(candidates/dominant_share/stable 등) 리스트를 만든다."""
    if not detections:
        return [], []

    clusters = cluster_ocr_detections(detections)
    pairs = [(c, merge_ocr_cluster(c)) for c in clusters]
    pairs.sort(key=lambda p: p[1]["frame_range"][0])

    ocr_spans = []
    ocr_candidates = []
    for cluster, m in pairs:
        ocr_candidates.append(m)
        if m["stable"]:
            ocr_spans.append({"start": m["start"], "end": m["end"], "text": m["candidates"][0]["text"]})
        else:
            for run in _runs_by_identical_text(cluster):
                ocr_spans.append({
                    "start": format_ts(run["start_sec"]),
                    "end": format_ts(run["end_sec"]),
                    "text": run["text"],
                })
    ocr_spans.sort(key=lambda s: s["start"])
    return ocr_spans, ocr_candidates

def _ocr_worker(ocr: RapidOCR, frames: list, out: dict):
    detections, ocr_before = [], []
    for i, (sec, img) in enumerate(frames):
        lines = _merge_same_line(run_ocr_on_frame(ocr, img))
        ocr_before.append("\n".join(l["text"] for l in lines))
        for line in lines:
            detections.append({"frame_idx": i, "sec": sec,
                                "text": line["text"], "score": line["score"], "bbox": line["bbox"]})
    out["detections"] = detections
    out["ocr_before"] = ocr_before


def _vlm_generate(processor, model, image: Image.Image, prompt: str, device: str, max_new_tokens: int) -> str:
    messages = [{"role": "user", "content": [
        {"type": "image", "image": image.convert("RGB")},
        {"type": "text",  "text": prompt},
    ]}]
    text_input = processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
    inputs = processor(text=[text_input], images=[image.convert("RGB")], return_tensors="pt").to(device)
    with torch.inference_mode():
        out = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
    return processor.batch_decode(out[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True)[0].strip()


def _parse_bracket_list(raw: str, tag: str, allowed: set) -> list:
    m = re.search(rf"\[{tag}\]\s*(.+)", raw)
    text = m.group(1) if m else raw
    items = [o.strip() for o in text.split(",") if o.strip() and o.strip() != "없음"]
    return [it for it in items if it in allowed]  # closed-set 강제 — 목록에 없는 할루시네이션 제거


# ──────────────────────────────────────────────
# 사기증거(36종) — closed-set. 36개는 짧아서 청크/retrieval 없이 한 번에 프롬프트에 다 넣어도
# 할루시네이션이 안 남 (평가로 확인됨). Grounding DINO의 frame_dets/aggregate() 자리를 대체.
# ──────────────────────────────────────────────
def _scam_evidence_prompt() -> str:
    return (
        "다음은 사이버 사기 의심 영상/이미지에서 뽑은 프레임이다.\n"
        "아래 목록 중 이 프레임에 실제로 등장하는 화면/증거 유형만 골라라.\n"
        f"[목록]\n{', '.join(SCAM_EVIDENCE_LABELS)}\n\n"
        "규칙: 반드시 위 목록에 있는 단어 그대로만 쓴다. 목록에 없는 새 단어는 절대 만들지 않는다. "
        "확신 없으면 포함하지 않는다. 없으면 \"없음\".\n"
        "출력 형식 (이 형식만 사용, 설명 없이): [EVIDENCE] 항목1, 항목2"
    )


def vlm_scam_evidence_batch(frames, processor, model, device: str, video_duration: float = 0.0) -> dict:
    """프레임별 closed-set 사기증거 탐지 후 영상 단위로 집계.
    반환 스키마는 기존 evidence_aggregator.aggregate()와 최대한 호환되게 맞춤
    (labels/evidence_score/top_labels/n_detected_frames) — should_escalate()가 그대로 동작하도록.
    단, VLM closed-set은 연속적인 confidence score를 안 주기 때문에(있다/없다만 판단)
    avg_score 대신 frame_ratio 기반으로 evidence_score를 계산함."""
    prompt = _scam_evidence_prompt()
    n_frames = len(frames)
    label_secs: dict[str, list] = defaultdict(list)

    detected_frame_count = 0
    for sec, img in frames:
        raw = _vlm_generate(processor, model, img, prompt, device, max_new_tokens=150)
        detected_labels = _parse_bracket_list(raw, "EVIDENCE", set(SCAM_EVIDENCE_LABELS))
        if detected_labels:
            detected_frame_count += 1
        for lb in detected_labels:
            label_secs[lb].append(sec)

    half = video_duration / 2.0
    labels_summary = {}
    for label, secs in label_secs.items():
        unique_frames = len(set(secs))
        if video_duration > 0:
            in_early = any(s <= half for s in secs)
            in_late  = any(s > half for s in secs)
            segment  = "both" if (in_early and in_late) else ("early" if in_early else "late")
        else:
            segment = "unknown"
        labels_summary[label] = {
            "count": len(secs), "frame_count": unique_frames,
            "frame_ratio": round(unique_frames / n_frames, 4) if n_frames else 0.0,
            "first_sec": round(min(secs), 2), "last_sec": round(max(secs), 2),
            "segment": segment,
        }

    # 증거가 하나 이상 나온 프레임 수다. 레이블 종류 수와 혼동하지 않는다.
    n_detected = detected_frame_count
    # 서로 다른 증거 종류가 많이/자주 나올수록 evidence_score를 높임 (연속 score가 없어서 근사치)
    evidence_score = min(1.0, sum(lb["frame_ratio"] for lb in labels_summary.values()) / 2) if labels_summary else 0.0
    top_labels = sorted(labels_summary, key=lambda lb: labels_summary[lb]["frame_ratio"], reverse=True)[:5]

    return {
        "labels": labels_summary,
        "evidence_score": round(evidence_score, 4),
        "n_detected_frames": n_detected,
        "top_labels": top_labels,
    }


# ──────────────────────────────────────────────
# 일반객체(427종) — "짧은 장면묘사 → BGE-m3 retrieval(top-K) → closed-set 확인" 3단계.
# 427개를 프롬프트에 한 번에/청크로 다 넣으면 모델이 목록을 그대로 반복 생성하는 할루시네이션이
# 발생함(평가로 확인, F1 23.6%→29.5%에 그침). retrieval로 후보를 줄이니 F1 36.5%까지 개선되고
# 호출 수도 자유서술 방식과 비슷하게 유지됨. obj_mapper(ObjectMapper)가 이미 로드해둔 BGE-m3와
# 427종 임베딩을 그대로 재사용 — 새 모델 로딩 없음.
# ──────────────────────────────────────────────
DESCRIBE_PROMPT = "이 이미지를 한두 문장으로 자연스럽게 묘사해라 (객체 이름을 나열하지 말고, 장면을 서술하는 문장으로). 설명 없이 묘사 문장만 출력."


def _general_object_verify_prompt(candidates: list) -> str:
    return (
        "다음은 영상/이미지에서 뽑은 프레임이다.\n"
        "아래 후보 목록 중 이 프레임에 실제로 등장하는 객체만 골라라.\n"
        f"[후보]\n{', '.join(candidates)}\n\n"
        "규칙: 반드시 위 후보에 있는 단어 그대로만 쓴다. 후보에 없는 새 단어는 절대 만들지 않는다. "
        "확신 낮으면 제외. 없으면 \"없음\".\n"
        "출력 형식 (이 형식만 사용, 설명 없이): [OBJECTS] 항목1, 항목2, 항목3"
    )


def vlm_object_fallback(vlm_frames, processor, model, obj_mapper, device: str) -> list:
    """일반 객체(427종) closed-set 탐지. 함수명은 v0.2와의 호출부 호환을 위해 유지."""
    detected = set()
    for _, img in vlm_frames:
        desc = _vlm_generate(processor, model, img, DESCRIBE_PROMPT, device, max_new_tokens=60)

        desc_emb = obj_mapper.model.encode([desc], convert_to_numpy=True, normalize_embeddings=True)[0]
        sims = obj_mapper.embeddings @ desc_emb
        top_idx = np.argsort(-sims)[:OBJECT_RAG_TOP_K]
        candidates = [obj_mapper.allowed[i] for i in top_idx]

        verify_prompt = _general_object_verify_prompt(candidates)
        raw = _vlm_generate(processor, model, img, verify_prompt, device, max_new_tokens=100)
        detected.update(_parse_bracket_list(raw, "OBJECTS", set(candidates)))
    return list(detected)


def run_ocr(ocr_frames, ocr: RapidOCR) -> dict:
    ocr_out = {}
    _ocr_worker(ocr, ocr_frames, ocr_out)
    ocr_spans, ocr_candidates = build_ocr_outputs(ocr_out.get("detections", []))
    return {
        "ocr_before":     ocr_out.get("ocr_before", []),
        "ocr_spans":      ocr_spans,
        "ocr_candidates": ocr_candidates,
    }


def correct_ocr_text_with_vlm(processor, model, raw_text: str, device: str) -> str:
    """이미지 1장의 원시 OCR 텍스트를 VLM(텍스트 전용)에 넣어 문맥 기반으로 오타/오인식만 교정.
    이미지 시퀀스(각 이미지가 서로 다른 화면이라 프레임 간 클러스터링이 의미 없는 입력)에 사용."""
    if not raw_text.strip():
        return raw_text
    prompt = (
        "다음은 이미지 하나에서 추출한 원시 OCR 텍스트다.\n"
        "1. 문맥을 참고해 오타와 OCR 오인식을 자연스럽게 교정하라.\n"
        "2. 끊어진 문장은 자연스럽게 이어 붙여라.\n"
        "3. @아이디, URL, 해시태그, 전화번호, 계좌번호, 금액 등 식별자는 원문 그대로 보존하라.\n"
        "4. 교정된 텍스트만 출력하라 (설명, 따옴표, 마크다운 없이).\n\n"
        f"[원본 OCR]\n{raw_text}"
    )
    messages = [{"role": "user", "content": prompt}]
    text_input = processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
    inputs = processor(text=[text_input], return_tensors="pt").to(device)
    with torch.inference_mode():
        output_ids = model.generate(**inputs, max_new_tokens=1024, do_sample=False)
    corrected = processor.batch_decode(
        output_ids[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True
    )[0].strip()
    return corrected or raw_text

# ──────────────────────────────────────────────
# 영상 다운로드
# ──────────────────────────────────────────────
def download_video(url: str, out_dir: Path) -> tuple:
    outtmpl = str(out_dir / "%(title).80s_%(id)s.%(ext)s")
    ydl_opts = {"outtmpl": outtmpl, "format": "best[ext=mp4]/best",
                "noplaylist": True, "quiet": True, "no_warnings": True}
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=True)
    v_path = Path(ydl.prepare_filename(info))
    if not v_path.exists():
        v_path = list(out_dir.glob(f"*{info['id']}*"))[0]
    return v_path, info


# ──────────────────────────────────────────────
# 단일 영상/이미지 분석 (공통 로직)
# ──────────────────────────────────────────────
VIDEO_EXTS = {".mp4", ".avi", ".mkv", ".mov", ".webm", ".flv", ".ts"}
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".webp", ".tiff"}


def _get_video_frames(v_path: Path, args) -> tuple:
    """비디오 파일(로컬 경로, URL이든 다운로드 후든 동일)에서 1~3단계에 필요한
    프레임 세트와 재생시간을 뽑는다. URL/로컬 파일 공통 사용."""
    cap = cv2.VideoCapture(str(v_path))
    if not cap.isOpened():
        cap.release()
        raise ValueError(f"비디오를 열 수 없습니다: {v_path}")
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = float(cap.get(cv2.CAP_PROP_FPS))
    cap.release()
    if frame_count <= 0 or fps <= 0:
        raise ValueError(f"유효한 비디오 프레임/FPS가 없습니다: {v_path}")
    duration = frame_count / fps
    vlm_frames = sample_keyframe(v_path, args.scan_sec, args.max_vlm_frames)
    evidence_frames = sample_uniform(
        v_path, every_n=args.evidence_sec, max_frames=args.max_evidence_frames
    )
    ocr_frames = sample_uniform(v_path, args.sample_sec, args.max_frames)
    if not vlm_frames or not evidence_frames:
        raise ValueError(f"비디오에서 분석 프레임을 추출하지 못했습니다: {v_path}")
    return duration, vlm_frames, evidence_frames, ocr_frames


def _get_image_frames(image: Image.Image) -> tuple:
    """정지 이미지 1장 — 시간 축이 없어서 1~3단계 모두 같은 단일 프레임을 그대로 재사용."""
    frame = [(0.0, image)]
    return 0.0, frame, frame, frame


def _analyze_and_save(video_id, source, label, title, duration,
                       vlm_frames, evidence_frames, ocr_frames,
                       processor, model,
                       ocr: RapidOCR, args,
                       obj_mapper: ObjectMapper | None, t0: float,
                       is_sequence: bool = False) -> str | None:
    """1~3단계 분석(CoT 분류 → VLM closed-set 증거탐지 → 에스컬레이션 → OCR/객체 → risk_agent
    위험도/사기유형) + 결과 저장. URL 기반(process_single_video)과 로컬 파일 기반(process_local_file)이
    프레임만 각자 다르게 뽑아서 이 함수를 공통으로 호출한다."""
    res_dir   = ensure_dir(Path(args.out_dir) / "results")
    json_path = res_dir / f"ocr_results_{video_id}.json"
    try:
        # ── 1단계: CoT 분류 ──
        cls_label, cls_summary = classify_scam(processor, model, vlm_frames, title, device=DEVICE)
        reason_m = re.search(r"판단근거:\s*(.+)$", cls_summary, re.S)
        reason = reason_m.group(1).strip() if reason_m else cls_summary[:80]
        print(f"  [분류]  {cls_label}  | 판단근거: {reason}")

        # ── 2단계: VLM closed-set(36종) — 모든 영상에서 증거 탐지 (정상 판정도 에스컬레이션 체크 위해 실행) ──
        evidence = vlm_scam_evidence_batch(evidence_frames, processor, model, DEVICE, video_duration=duration)
        print(f"  [사기증거] score={evidence['evidence_score']:.3f}"
              f"  top={evidence['top_labels'][:3]}")

        # [v0.3] 양방향 에스컬레이션 — 단, 어느 방향이든 "검토필요"로만 이동하고 "정상"으로
        # 자동 확정되는 경우는 없음(1단계가 정상이라 해도 그대로 정상 유지일 뿐, 사기였던 걸
        # "정상"으로 지우는 경로는 없음). 사기→정상 자동 다운그레이드는 만들지 않았는데, 증거탐지
        # recall이 완벽하지 않은 상태에서 그걸로 진짜 사기 판정을 지워버리면 "사기를 놓치는 것보다
        # 과탐지가 낫다"는 원래 설계 원칙에 정면으로 위배되기 때문 — 대신 "검토필요"로만 내려서
        # 사람이 한 번 더 보게 함.
        final_label = cls_label
        if cls_label == "불명확":
            final_label = "검토필요"
            print("  [에스컬레이션] 분류 출력 파싱 실패 → 검토필요")
        elif cls_label == "정상" and should_escalate(evidence, threshold=args.escalate_thr):
            final_label = "검토필요"
            print(f"  [에스컬레이션] 정상 → 검토필요"
                  f"  (evidence_score={evidence['evidence_score']:.3f} >= {args.escalate_thr})")
        elif cls_label == "사기" and evidence["evidence_score"] < args.escalate_thr:
            final_label = "검토필요"
            print(f"  [디에스컬레이션] 사기 → 검토필요"
                  f"  (evidence_score={evidence['evidence_score']:.3f} < {args.escalate_thr}, 뒷받침 증거 없음)")

        base_payload = {
            "id":             video_id,
            "url":            source,
            "title":          title,
            "label":          label,           # GT (CSV/--dir 등에서 받은 것)
            "cls_label":      cls_label,        # 체크리스트 판단
            "final_label":    final_label,     # 최종 판단 (VLM 증거 에스컬레이션 포함)
            "classify_summary": cls_summary,
            "grounding":      evidence,
            "objects":        [],
            "scam_evidence":  evidence["top_labels"],
            "ocr_before":     [],
            "ocr":            [],
            "ocr_after":      "",
            "ocr_candidates": [],
            "total_inference_time": 0.0,
        }

        if final_label not in ("사기", "검토필요"):
            base_payload["skipped"] = True
            base_payload["total_inference_time"] = round(time.time() - t0, 2)
            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(base_payload, f, ensure_ascii=False, indent=2)
            print(f"  → {final_label} (skip detailed analysis)")
            print_report(final_label, reason, evidence, None)
            return final_label

        # ── 3단계: 사기/검토필요 → OCR + 객체 탐지. 객체는 2단계 사기증거(36종, top_labels)와
        # 일반객체(427종, RAG-retrieval closed-set)를 합침 — 사기로 확정된 영상은 증거를
        # 최대한 남기기 위해 둘 다 사용 (정상은 위에서 이미 스킵되므로 비용 문제 없음) ──
        if is_sequence:
            # 이미지 시퀀스: 각 이미지가 서로 다른 화면이라 프레임 간 IoU/CER 클러스터링(영상 전제)이
            # 안 맞음 — 화면 위치가 우연히 겹치는 별개 텍스트를 잘못 묶거나, 조각난 스팬 수십~수백 개를
            # 한 번에 VLM에 넣어 "병합"시키다 타임스탬프까지 망가짐. 대신 이미지별 원문(ocr_before)을
            # 그대로 유지한 채 VLM으로 오타/오인식만 교정 (문장 재배열/병합 없음).
            ocr_out = {}
            _ocr_worker(ocr, ocr_frames, ocr_out)
            ocr_before = ocr_out.get("ocr_before", [])
            corrected_ocr = [
                {"start": format_ts(float(i)), "end": format_ts(float(i)),
                 "text": correct_ocr_text_with_vlm(processor, model, text, DEVICE)}
                for i, text in enumerate(ocr_before) if text.strip()
            ]
            detail = {"ocr_before": ocr_before, "ocr_candidates": []}
        else:
            detail = run_ocr(ocr_frames, ocr)
            corrected_ocr = detail["ocr_spans"] # VLM 기반 후처리 제외
        ocr_all = " | ".join(s["text"] for s in corrected_ocr)

        general_objs = vlm_object_fallback(vlm_frames, processor, model, obj_mapper, DEVICE)
        objects_out = list(dict.fromkeys(evidence["top_labels"] + general_objs))

        full_payload = {
            **base_payload,
            "objects":        objects_out,        # 사기증거 top_labels ∪ 일반객체 (기존 스키마 호환)
            "scam_evidence":  evidence["top_labels"],  # 사기증거만 별도 필드 (평가/분석용)
            "ocr_before":     detail["ocr_before"],
            "ocr":            corrected_ocr,
            "ocr_after":      ocr_all,
            "ocr_candidates": detail["ocr_candidates"],
            "skipped":        False,
            "total_inference_time": round(time.time() - t0, 2),
        }

        # risk_agent 위험도/사기유형 분류 (예전 RAG 자리) — 사기/검토필요로 확정된 건에만 실행
        risk_assessment = assess_risk(
            full_payload, victim_count=args.victim_count, total_loss_won=args.total_loss_won
        )
        full_payload["risk_assessment"] = risk_assessment

        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(full_payload, f, ensure_ascii=False, indent=2)

        print(f"  저장: {json_path}  ({time.time()-t0:.1f}s)")
        print_report(final_label, reason, evidence, risk_assessment)
        return final_label

    except Exception as e:
        print(f"  [Error] {source}: {e}")
        import traceback; traceback.print_exc()
    return None


def process_single_video(url, label, processor, model,
                          ocr: RapidOCR, args,
                          obj_mapper: ObjectMapper) -> str | None:
    """YouTube/TikTok URL 처리 — 다운로드 후 _analyze_and_save() 공통 로직 호출."""
    vid_m = re.search(r"(?:v=|video/|shorts/)([a-zA-Z0-9_-]+)", url)
    video_id = vid_m.group(1) if vid_m else str(int(time.time()))
    res_dir   = ensure_dir(Path(args.out_dir) / "results")
    json_path = res_dir / f"ocr_results_{video_id}.json"

    if json_path.exists():
        print(f"  [Skip] {video_id}")
        with open(json_path, encoding="utf-8") as f:
            return json.load(f).get("final_label")

    print(f"  [Processing] {url}  (label={label})")
    t0 = time.time()
    try:
        v_path, info = download_video(url, Path(args.out_dir))
        title = info.get("title", "")
        duration, vlm_frames, evidence_frames, ocr_frames = _get_video_frames(v_path, args)
    except Exception as e:
        print(f"  [Error] {url}: {e}")
        import traceback; traceback.print_exc()
        return None

    return _analyze_and_save(video_id, url, label, title, duration,
                              vlm_frames, evidence_frames, ocr_frames,
                              processor, model, ocr, args, obj_mapper, t0)


def process_local_file(file_path: Path, label, processor, model,
                        ocr: RapidOCR, args,
                        obj_mapper: ObjectMapper) -> str | None:
    """로컬 영상/이미지 파일 처리 — 다운로드 없이 파일에서 바로 프레임을 뽑아
    _analyze_and_save() 공통 로직 호출. 이미지는 시간 축이 없어 프레임 1장으로 취급."""
    ext = file_path.suffix.lower()
    if ext not in VIDEO_EXTS | IMAGE_EXTS:
        print(f"  [Error] 지원하지 않는 파일 형식: {ext or '(확장자 없음)'}")
        return None
    video_id = file_path.stem
    res_dir   = ensure_dir(Path(args.out_dir) / "results")
    json_path = res_dir / f"ocr_results_{video_id}.json"

    if json_path.exists():
        print(f"  [Skip] {video_id}")
        with open(json_path, encoding="utf-8") as f:
            return json.load(f).get("final_label")

    print(f"  [Processing] {file_path.name}  (label={label})")
    t0 = time.time()
    try:
        if ext in IMAGE_EXTS:
            image = Image.open(file_path).convert("RGB")
            duration, vlm_frames, evidence_frames, ocr_frames = _get_image_frames(image)
        else:
            duration, vlm_frames, evidence_frames, ocr_frames = _get_video_frames(file_path, args)
    except Exception as e:
        print(f"  [Error] {file_path.name}: {e}")
        import traceback; traceback.print_exc()
        return None

    return _analyze_and_save(video_id, str(file_path), label, file_path.name, duration,
                              vlm_frames, evidence_frames, ocr_frames,
                              processor, model, ocr, args, obj_mapper, t0)


def process_image_sequence(dir_path: Path, label, processor, model,
                            ocr: RapidOCR, args,
                            obj_mapper: ObjectMapper) -> str | None:
    """이미지 시퀀스 폴더(예: 카톡 대화를 이어 찍은 스크린샷 여러 장) 처리.
    폴더 안 이미지들을 파일명 순으로 정렬해 하나의 프레임 시퀀스로 묶어 _analyze_and_save() 공통 로직 호출.
    --dir처럼 파일마다 따로 판정하지 않고, 폴더 전체를 한 건으로 판정한다."""
    video_id = dir_path.name
    res_dir   = ensure_dir(Path(args.out_dir) / "results")
    json_path = res_dir / f"ocr_results_{video_id}.json"

    if json_path.exists():
        print(f"  [Skip] {video_id}")
        with open(json_path, encoding="utf-8") as f:
            return json.load(f).get("final_label")

    image_files = sorted(f for f in dir_path.iterdir() if f.suffix.lower() in IMAGE_EXTS)
    if not image_files:
        print(f"  [Error] 이미지 없음: {dir_path}")
        return None

    print(f"  [Processing] {dir_path.name}/ ({len(image_files)}장)  (label={label})")
    t0 = time.time()
    try:
        images     = [Image.open(f).convert("RGB") for f in image_files]
        all_frames = [(float(i), img) for i, img in enumerate(images)]
        duration   = float(len(images))

        # 분류(CoT)용 대표 프레임: max_vlm_frames보다 많으면 전체 구간에서 고르게 추출.
        # (앞 4장만 보면 대화 뒷부분의 결정적 증거—계좌번호 요구 등—를 놓칠 수 있음)
        n = args.max_vlm_frames
        if len(all_frames) > n > 1:
            idxs = sorted({round(i * (len(all_frames) - 1) / (n - 1)) for i in range(n)})
            vlm_frames = [all_frames[i] for i in idxs]
        elif n <= 1:
            vlm_frames = all_frames[:1]
        else:
            vlm_frames = all_frames
    except Exception as e:
        print(f"  [Error] {dir_path}: {e}")
        import traceback; traceback.print_exc()
        return None

    return _analyze_and_save(video_id, str(dir_path), label, dir_path.name, duration,
                              vlm_frames, all_frames, all_frames,
                              processor, model, ocr, args, obj_mapper, t0,
                              is_sequence=True)


# ──────────────────────────────────────────────
# 메인
# ──────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="cybercop pipeline AdotX_v0_3 — CoT + VLM closed-set 증거/객체 탐지")
    inputs = parser.add_mutually_exclusive_group(required=True)
    inputs.add_argument("--url",                  help="단일 YouTube/TikTok URL")
    inputs.add_argument("--file",                 help="단일 로컬 영상/이미지 파일 경로")
    inputs.add_argument("--dir",                  help="영상/이미지 파일 디렉토리 (파일마다 독립 판정)")
    inputs.add_argument("--seq_dir",              help="이어지는 이미지 시퀀스 디렉토리 (전체를 한 건으로 판정)")
    inputs.add_argument("--csv",                  help="label/link 또는 label/url 컬럼 CSV")
    parser.add_argument("--out_dir",              default="./output_AdotX_v0.3")
    parser.add_argument("--model",                default=MODEL_ID)
    parser.add_argument("--sample_sec",           type=float, default=2.0)
    parser.add_argument("--max_frames",           type=int,   default=1000)
    parser.add_argument("--scan_sec",             type=float, default=0.5)
    parser.add_argument("--max_vlm_frames",       type=int,   default=4)
    parser.add_argument("--evidence_sec", "--dino_sec", dest="evidence_sec",
                        type=float, default=3.0,
                        help="VLM 증거 탐지용 균등 샘플링 간격(초)")
    parser.add_argument("--max_evidence_frames", "--max_dino_frames",
                        dest="max_evidence_frames", type=int, default=15,
                        help="VLM 증거 탐지에 넘길 최대 프레임 수")
    parser.add_argument("--escalate_thr",         type=float, default=ESCALATE_THR)
    parser.add_argument("--victim_count",         type=int,   default=None,
                        help="risk_assessment용 피해자 수 수동 override (미지정 시 0=정보없음)")
    parser.add_argument("--total_loss_won",       type=int,   default=None,
                        help="risk_assessment용 총 피해금액(원) 수동 override (미지정 시 OCR에서 추출 시도)")
    args = parser.parse_args()
    if min(args.sample_sec, args.scan_sec, args.evidence_sec) <= 0:
        parser.error("샘플링 간격은 0보다 커야 합니다.")
    if min(args.max_frames, args.max_vlm_frames, args.max_evidence_frames) <= 0:
        parser.error("프레임 수는 0보다 커야 합니다.")
    if not 0.0 <= args.escalate_thr <= 1.0:
        parser.error("--escalate_thr는 0~1 범위여야 합니다.")

    ensure_dir(args.out_dir)

    print("[1/3] VLM 로드...")
    processor, model = load_vlm(args.model)

    print("[2/3] OCR + 임베딩 로드...")
    ocr = load_ocr()
    embed_model = SentenceTransformer(DEFAULT_EMBED_MODEL, device=DEVICE)
    obj_mapper = ObjectMapper(embed_model)  # BGE-m3 + 427종 임베딩 — 일반객체 retrieval용

    print("[3/3] 처리 시작...")

    preds = []
    failed = 0
    if args.url:
        pred = process_single_video(args.url, "manual", processor, model, ocr, args, obj_mapper)
        if pred:
            preds.append(("unknown", pred))
        else:
            failed += 1
    elif args.file:
        file_path = Path(args.file)
        if not file_path.is_file():
            print(f"\n[Error] 파일을 찾을 수 없습니다: {args.file}")
            sys.exit(1)
        pred = process_local_file(file_path, "manual", processor, model, ocr, args, obj_mapper)
        if pred:
            preds.append(("unknown", pred))
        else:
            failed += 1
    elif args.dir:
        dir_path = Path(args.dir)
        if not dir_path.is_dir():
            print(f"\n[Error] 디렉토리를 찾을 수 없습니다: {args.dir}")
            sys.exit(1)
        files = sorted(f for f in dir_path.iterdir()
                       if f.suffix.lower() in VIDEO_EXTS | IMAGE_EXTS)
        if not files:
            print(f"\n[Error] 지원 파일 없음 (지원 확장자: {VIDEO_EXTS | IMAGE_EXTS})")
            sys.exit(1)
        print(f"  총 {len(files)}개 파일 발견")
        for f in files:
            pred = process_local_file(f, "manual", processor, model, ocr, args, obj_mapper)
            if pred:
                preds.append(("unknown", pred))
            else:
                failed += 1
    elif args.seq_dir:
        seq_path = Path(args.seq_dir)
        if not seq_path.is_dir():
            print(f"\n[Error] 디렉토리를 찾을 수 없습니다: {args.seq_dir}")
            sys.exit(1)
        pred = process_image_sequence(seq_path, "manual", processor, model, ocr, args, obj_mapper)
        if pred:
            preds.append(("unknown", pred))
        else:
            failed += 1
    elif args.csv:
        with open(args.csv, "r", encoding="utf-8-sig") as f:
            for row in csv.DictReader(f):
                url = row.get("link") or row.get("url")
                gt  = row.get("label", "unknown")
                if url:
                    pred = process_single_video(url, gt, processor, model, ocr, args, obj_mapper)
                    if pred is not None and gt in ("abnormal", "normal"):
                        preds.append((gt, pred))
                    elif pred is None:
                        failed += 1

    eval_preds = [(gt, pred) for gt, pred in preds if gt in ("abnormal", "normal")]
    if eval_preds:
        # "검토필요"를 사기로 간주해서 성능 계산
        def is_scam(p): return p in ("사기", "검토필요")
        tp = sum(1 for gt, p in eval_preds if is_scam(p) and gt == "abnormal")
        fp = sum(1 for gt, p in eval_preds if is_scam(p) and gt == "normal")
        fn = sum(1 for gt, p in eval_preds if not is_scam(p) and gt == "abnormal")
        tn = sum(1 for gt, p in eval_preds if not is_scam(p) and gt == "normal")
        rec  = tp / (tp + fn) * 100 if (tp + fn) > 0 else 0.0
        prec = tp / (tp + fp) * 100 if (tp + fp) > 0 else 0.0
        f1   = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
        print(f"\n{'='*50}")
        print(f"  [성능] n={len(eval_preds)}  TP={tp} TN={tn} FP={fp} FN={fn}")
        print(f"  Recall={rec:.1f}%  Precision={prec:.1f}%  F1={f1:.1f}%")
        print(f"{'='*50}")

    if failed:
        print(f"\n[Error] {failed}건의 분석이 실패했습니다.")
        sys.exit(1)


if __name__ == "__main__":
    main()
