"""
cybercop_pipeline_AdotX_v0_1.py — CoT 분류 + Grounding DINO 증거 탐지 통합 파이프라인.

기존 cybercop_pipeline_AdotX.py 대비 주요 변경점은 CHANGELOG.md 참고.
이 파일 실행에는 같은 저장소 루트의 pipeline/ 패키지가 필요합니다.
"""
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# PaddleOCR(PaddleX)가 이미 캐시된 모델도 매번 온라인으로 재확인하려다가, 이 환경의 불안정한
# 네트워크(커넥션이 CLOSE-WAIT로 죽은 채 타임아웃 없이 멈춤) 때문에 수십 분씩 멎는 문제가 있었음.
# 필요한 모델(VLM/DINO/PaddleOCR)은 이미 전부 로컬에 있으므로 오프라인 모드를 기본값으로 강제.
# 최초 설치 시 모델을 새로 받아야 한다면 실행 전 `export HF_HUB_OFFLINE=0`으로 풀 것.
os.environ.setdefault("HF_HUB_OFFLINE", "1")

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
import faiss
import torch
from PIL import Image
from transformers import AutoModelForCausalLM, AutoProcessor
from huggingface_hub import snapshot_download
from sentence_transformers import SentenceTransformer
from paddleocr import PaddleOCR

# 새 파이프라인 모듈
from pipeline.frame_sampler import sample_uniform, sample_keyframe
from pipeline.vlm_classify import classify_scam
from pipeline.grounding_detector import GroundingDetector
from pipeline.evidence_aggregator import aggregate, should_escalate
from pipeline.vocab import SCAM_EVIDENCE_VOCAB

# ──────────────────────────────────────────────
# 설정
# ──────────────────────────────────────────────
MODEL_ID            = os.getenv("VLM_MODEL", "skt/A.X-4.0-VL-Light")
GDINO_MODEL_ID      = os.getenv("GDINO_MODEL", "IDEA-Research/grounding-dino-tiny")
DEVICE              = "cuda" if torch.cuda.is_available() else "cpu"
TORCH_DTYPE         = torch.bfloat16 if torch.cuda.is_available() else torch.float32
DEFAULT_EMBED_MODEL = "BAAI/bge-m3"
DEFAULT_DOCS_PATH   = str(Path(__file__).parent / "rag" / "retrieval_docs_v2.json")
ALLOWED_OBJECTS_PATH = Path(__file__).parent / "rag" / "allowed_objects.json"

# Grounding DINO 에스컬레이션 임계값
# CoT="정상"이라도 evidence_score >= 이 값이면 "검토필요" 로 표시
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


# ──────────────────────────────────────────────
# 모델 로드
# ──────────────────────────────────────────────
def load_vlm(model_id: str):
    if os.path.isdir(model_id):
        local_dir = Path(model_id)
    else:
        safe_name = model_id.replace("/", "_").replace(".", "_")
        local_dir = Path("./hf_models") / safe_name
        if not local_dir.exists():
            snapshot_download(repo_id=model_id, local_dir=str(local_dir))
    processor = AutoProcessor.from_pretrained(str(local_dir), trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        str(local_dir), torch_dtype=TORCH_DTYPE, device_map=DEVICE, trust_remote_code=True,
    )
    model.eval()
    return processor, model


def load_paddleocr() -> PaddleOCR:
    # GPU(use_gpu=True)로 시도해봤으나 이 환경엔 paddlepaddle-gpu가 필요로 하는 cuDNN을
    # 못 찾아서(PreconditionNotMet: cudnn_dso_handle) 매 요청마다 500 에러가 났음.
    # cuDNN 경로 문제를 별도로 해결하기 전까진 CPU로 되돌림.
    return PaddleOCR(use_angle_cls=True, lang="korean", use_gpu=False, show_log=False)


# ──────────────────────────────────────────────
# RAG
# ──────────────────────────────────────────────
CRIME_RISK_MAP = {
    # retrieval_docs_v2의 현재 범죄 분류 15종만 사용한다.
    "사이버사기": {"months": 15.6, "risk": 0.28},
    "사이버 금융범죄": {"months": 16.4, "risk": 0.30},
    "개인·위치정보 침해": {"months": 12.3, "risk": 0.22},
    "사이버 저작권 침해": {"months": 13.0, "risk": 0.24},
    "사이버스팸메일": {"months": 12.0, "risk": 0.22},
    "기타 정보통신망 이용 범죄": {"months": 7.0, "risk": 0.13},
    "사이버성폭력": {"months": 25.3, "risk": 0.46},
    "사이버도박": {"months": 13.8, "risk": 0.25},
    "사이버 명예훼손·모욕": {"months": 8.0, "risk": 0.15},
    "사이버스토킹": {"months": 11.0, "risk": 0.20},
    "기타 불법 콘텐츠 범죄": {"months": 8.0, "risk": 0.15},
    "해킹": {"months": 18.0, "risk": 0.33},
    "서비스거부공격(DDoS)": {"months": 18.0, "risk": 0.33},
    "악성프로그램": {"months": 18.0, "risk": 0.33},
    "기타 정보통신망 침해형 범죄": {"months": 18.0, "risk": 0.33},
}


class CrimeRAG:
    def __init__(self, docs_path: str, model_name: str = DEFAULT_EMBED_MODEL):
        with open(docs_path, "r", encoding="utf-8") as f:
            self.docs = json.load(f)
        self.model = SentenceTransformer(model_name, device=DEVICE)
        self.index = None

    def build_index(self):
        texts = [doc["text"] for doc in self.docs]
        embs = self.model.encode(texts, convert_to_numpy=True, normalize_embeddings=True).astype("float32")
        self.index = faiss.IndexFlatIP(embs.shape[1])
        self.index.add(embs)

    def search(self, query: str, top_k: int = 3):
        if not self.index:
            self.build_index()
        q = self.model.encode([query], convert_to_numpy=True, normalize_embeddings=True).astype("float32")
        scores, indices = self.index.search(q, top_k)
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if 0 <= idx < len(self.docs):
                doc = dict(self.docs[idx])
                doc["similarity"] = float(score)
                doc["risk_level"] = CRIME_RISK_MAP.get(doc.get("crime_type", ""), {}).get("risk", 0.0)
                results.append(doc)
        return results


class ObjectMapper:
    """VLM 폴백(vlm_object_fallback)이 자유롭게 뱉은 객체명을 rag/allowed_objects.json의
    고정 어휘(439종)로 정규화. 나중에 정확도 검증/분석할 때 열린 어휘가 아니라 고정된 클래스
    기준으로 볼 수 있게 하기 위함 — DINO의 top_labels(SCAM_EVIDENCE_VOCAB)에는 적용하지 않음
    (성격이 다른 별개 어휘라 섞으면 오히려 의미가 흐려짐)."""

    def __init__(self, embed_model: SentenceTransformer, threshold: float = 0.7):
        # 0.6이면 "금고"→"망고"(0.63), "저장"→"컴퓨터"(0.62), "이모티콘"→"리모컨"(0.62) 처럼
        # 어휘에 없는 개념이 무관한 단어로 강제 매핑됨. 0.7로 올리면 확인된 오매핑은 다 걸러지고
        # "앱"→"텔레그램 앱"(0.73), "확인"→"인증"(0.77) 같은 정상 매핑은 유지됨.
        self.model = embed_model
        self.threshold = threshold
        self.allowed = json.load(open(ALLOWED_OBJECTS_PATH, encoding="utf-8"))
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
def _poly_bounds(poly):
    """PaddleOCR 4점 폴리곤 → 축정렬 사각형 (xmin, ymin, xmax, ymax)."""
    xs = [p[0] for p in poly]
    ys = [p[1] for p in poly]
    return min(xs), min(ys), max(xs), max(ys)


def _merge_same_line(lines, y_tol=10):
    if not lines:
        return []
    lines = sorted(lines, key=lambda x: x[0][0][1])
    merged, cur_line = [], [lines[0]]
    for line in lines[1:]:
        if abs(line[0][0][1] - cur_line[-1][0][0][1]) < y_tol:
            cur_line.append(line)
        else:
            merged.append(cur_line)
            cur_line = [line]
    merged.append(cur_line)
    result = []
    for grp in merged:
        grp.sort(key=lambda x: x[0][0][0])
        text = " ".join(w[1][0] for w in grp)
        score = sum(w[1][1] for w in grp) / len(grp)
        x0, y0, _, y1 = _poly_bounds(grp[0][0])
        _, y2, x1, y3 = _poly_bounds(grp[-1][0])
        bbox = [x0, min(y0, y2), x1, max(y1, y3)]
        result.append({"text": text, "score": score, "bbox": bbox})
    return result


# ──────────────────────────────────────────────
# OCR 프레임간 클러스터링 (HS/cybercop_pipeline_shared/cybercop_pipeline_latest.py 포팅)
# 같은 UI 텍스트가 여러 프레임에서 OCR 오탈자로 조금씩 다르게 읽히는 걸
# IoU(위치)+CER(문자열 유사도)로 묶어 다수결 대표값을 뽑는다.
# 호가창 가격처럼 실제로 매 프레임 바뀌는 숫자값은 클러스터링 대상에서 제외한다.
# ──────────────────────────────────────────────
IOU_THRESHOLD = 0.3
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


def run_paddleocr_on_frame(ocr: PaddleOCR, img: Image.Image):
    arr = np.array(img.convert("RGB"))
    res = ocr.ocr(arr, cls=True)
    if not res or not res[0]:
        return []
    return [[r[0], r[1]] for r in res[0] if r[1][1] > 0.5]


def _ocr_worker(ocr: PaddleOCR, frames: list, out: dict):
    detections, ocr_before = [], []
    for i, (sec, img) in enumerate(frames):
        lines = _merge_same_line(run_paddleocr_on_frame(ocr, img))
        ocr_before.append("\n".join(l["text"] for l in lines))
        for line in lines:
            detections.append({"frame_idx": i, "sec": sec,
                                "text": line["text"], "score": line["score"], "bbox": line["bbox"]})
    out["detections"] = detections
    out["ocr_before"] = ocr_before


DEFAULT_PROMPT = """이 이미지는 동영상의 한 프레임이다.
이미지에 보이는 사물을 한국어로 콤마(,)로 구분해 최대 10개 나열하라.
규칙: 사물 이름만 적고 설명은 쓰지 않는다. 중복 금지. 확신 낮으면 제외. 없으면 "없음".
출력 형식 (반드시 이 형식만 사용, 줄바꿈·불릿·번호 금지):
[OBJECTS] 사물1, 사물2, 사물3""".strip()


def _analyze_one_frame_vlm(processor, model, image: Image.Image, device: str) -> list:
    messages = [{"role": "user", "content": [
        {"type": "image", "image": image.convert("RGB")},
        {"type": "text",  "text": DEFAULT_PROMPT},
    ]}]
    text_input = processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
    inputs = processor(text=[text_input], images=[image.convert("RGB")], return_tensors="pt").to(device)
    with torch.inference_mode():
        out = model.generate(**inputs, max_new_tokens=150, do_sample=False)
    raw = processor.batch_decode(out[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True)[0]
    # [OBJECTS] 태그를 안 붙이고 콤마 리스트만 뱉는 경우가 있어서, 태그 없으면 raw 전체를 그대로 파싱
    m = re.search(r"\[OBJECTS\]\s*(.+)", raw)
    text = m.group(1) if m else raw
    return [o.strip() for o in text.split(",") if o.strip() and o.strip() != "없음"]


def vlm_object_fallback(vlm_frames, processor, model, device: str) -> list:
    """DINO도 OCR도 아무 신호가 없을 때만 쓰는 최후의 폴백 — 자유 서술 객체 나열.
    다른 신호가 없는 상황이라 투표 필터 없이 프레임 중 한 번이라도 언급된 건 전부 채택한다
    (다수결로 걸러내면 폴백 자체가 무의미해짐)."""
    obj_counts = {}
    for _, img in vlm_frames:
        for obj in _analyze_one_frame_vlm(processor, model, img, device):
            obj_counts[obj] = obj_counts.get(obj, 0) + 1
    return sorted(obj_counts, key=obj_counts.get, reverse=True)[:10]


def run_ocr(ocr_frames, ocr: PaddleOCR) -> dict:
    ocr_out = {}
    _ocr_worker(ocr, ocr_frames, ocr_out)
    ocr_spans, ocr_candidates = build_ocr_outputs(ocr_out.get("detections", []))
    return {
        "ocr_before":     ocr_out.get("ocr_before", []),
        "ocr_spans":      ocr_spans,
        "ocr_candidates": ocr_candidates,
    }


def postprocess_ocr_with_vlm(processor, model, ocr_spans: list, device: str) -> list:
    """v0.1의 postprocess_ocr_with_vlm 포팅 — 클러스터링된 OCR 타임라인 전체를
    한 번에 VLM(텍스트 전용)에 넣어 오타/할루시네이션 교정, 문장 연결, 중복 병합."""
    if not ocr_spans:
        return ocr_spans
    input_json = json.dumps(ocr_spans, ensure_ascii=False, indent=2)
    prompt = (
        "다음은 동영상에서 시간대별로 추출된 원시 OCR 텍스트의 JSON 배열이다.\n"
        "1. 오타와 할루시네이션(무의미한 반복)을 교정하라.\n"
        "2. 전체 전후 문맥을 파악하여 끊어진 문장을 자연스럽게 하나로 연결하라.\n"
        "3. 같은 내용이 이어지는 경우 하나로 병합하고 start/end 업데이트.\n"
        "4. @아이디, URL, 해시태그, 전화번호, 계좌번호 등 식별자는 원문 그대로 보존하라.\n"
        "5. 오직 교정된 JSON 배열만 출력하라.\n\n"
        f"[원본 타임라인 데이터]\n{input_json}"
    )
    messages = [{"role": "user", "content": prompt}]
    text_input = processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
    inputs = processor(text=[text_input], return_tensors="pt").to(device)
    with torch.inference_mode():
        output_ids = model.generate(**inputs, max_new_tokens=2048, do_sample=False)
    corrected = processor.batch_decode(
        output_ids[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True
    )[0].strip()
    try:
        m = re.search(r'\[\s*\{.*?\}\s*\]', corrected, re.DOTALL)
        return json.loads(m.group(0) if m else corrected)
    except Exception:
        return ocr_spans  # 파싱 실패 시 교정 전 원본 유지


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
    duration = cap.get(cv2.CAP_PROP_FRAME_COUNT) / (cap.get(cv2.CAP_PROP_FPS) or 30.0)
    cap.release()
    vlm_frames  = sample_keyframe(v_path, args.scan_sec, args.max_vlm_frames)
    dino_frames = sample_uniform(v_path, every_n=args.dino_sec, max_frames=args.max_dino_frames)
    ocr_frames  = sample_uniform(v_path, args.sample_sec, args.max_frames)
    return duration, vlm_frames, dino_frames, ocr_frames


def _get_image_frames(image: Image.Image) -> tuple:
    """정지 이미지 1장 — 시간 축이 없어서 1~3단계 모두 같은 단일 프레임을 그대로 재사용."""
    frame = [(0.0, image)]
    return 0.0, frame, frame, frame


def _analyze_and_save(video_id, source, label, title, duration,
                       vlm_frames, dino_frames, ocr_frames,
                       processor, model, gdino: GroundingDetector,
                       rag: CrimeRAG, ocr: PaddleOCR, args,
                       obj_mapper: ObjectMapper | None, t0: float) -> str | None:
    """1~3단계 분석(CoT 분류 → DINO 증거탐지 → 에스컬레이션 → OCR/객체/RAG) + 결과 저장.
    URL 기반(process_single_video)과 로컬 파일 기반(process_local_file)이 프레임만
    각자 다르게 뽑아서 이 함수를 공통으로 호출한다."""
    res_dir   = ensure_dir(Path(args.out_dir) / "results")
    json_path = res_dir / f"ocr_results_{video_id}.json"
    try:
        # ── 1단계: CoT 분류 ──
        cls_label, cls_summary = classify_scam(processor, model, vlm_frames, title, device=DEVICE)
        print(f"  [분류]  {cls_label}  | {cls_summary[:80]}")

        # ── 2단계: DINO — 모든 영상에서 증거 객체 추출 (정상 판정도 에스컬레이션 체크 위해 실행) ──
        frame_dets  = gdino.detect_batch(dino_frames, SCAM_EVIDENCE_VOCAB)
        evidence    = aggregate(frame_dets, video_duration=duration)
        print(f"  [DINO 증거] score={evidence['evidence_score']:.3f}"
              f"  top={evidence['top_labels'][:3]}")

        final_label = cls_label
        if cls_label == "정상" and should_escalate(evidence, threshold=args.escalate_thr):
            final_label = "검토필요"
            print(f"  [에스컬레이션] 정상 → 검토필요"
                  f"  (evidence_score={evidence['evidence_score']:.3f} >= {args.escalate_thr})")

        base_payload = {
            "id":             video_id,
            "url":            source,
            "title":          title,
            "label":          label,           # GT (CSV/--dir 등에서 받은 것)
            "cls_label":      cls_label,        # 체크리스트 판단
            "final_label":    final_label,     # 최종 판단 (DINO 에스컬레이션 포함)
            "classify_summary": cls_summary,
            "grounding":      evidence,
            "total_inference_time": round(time.time() - t0, 2),
        }

        if final_label not in ("사기", "검토필요"):
            base_payload["skipped"] = True
            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(base_payload, f, ensure_ascii=False, indent=2)
            print(f"  → {final_label} (skip detailed analysis)")
            return final_label

        # ── 3단계: 사기/검토필요 → OCR + 객체 탐지. 객체는 DINO 결과(VOCAB, top_labels)와
        # VLM 폴백 결과(ALLOWED_OBJECTS 매핑)를 둘 다 합침 — DINO는 정밀하지만 좁은 어휘라
        # 놓치는 게 많고 VLM은 넓지만 부정확할 수 있어서, 사기로 확정된 영상은 증거를
        # 최대한 남기기 위해 둘 다 사용 (정상은 위에서 이미 스킵되므로 비용 문제 없음) ──
        detail = run_ocr(ocr_frames, ocr)
        corrected_ocr = postprocess_ocr_with_vlm(processor, model, detail["ocr_spans"], DEVICE)
        ocr_all = " | ".join(s["text"] for s in corrected_ocr)

        fallback_objs = vlm_object_fallback(vlm_frames, processor, model, DEVICE)
        if obj_mapper is not None:
            fallback_objs = obj_mapper.map(fallback_objs)
        objects_out = list(dict.fromkeys(evidence["top_labels"] + fallback_objs))
        obj_all = ", ".join(objects_out)

        mapped  = rag.search(f"[OCR]: {ocr_all} | [Objects]: {obj_all}", top_k=args.top_k)

        full_payload = {
            **base_payload,
            "objects":        objects_out,
            "ocr_before":     detail["ocr_before"],
            "ocr":            corrected_ocr,
            "ocr_text":       ocr_all,
            "ocr_candidates": detail["ocr_candidates"],
            "rag":            mapped,
        }
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(full_payload, f, ensure_ascii=False, indent=2)

        print(f"  저장: {json_path}  ({time.time()-t0:.1f}s)")
        return final_label

    except Exception as e:
        print(f"  [Error] {source}: {e}")
        import traceback; traceback.print_exc()
    return None


def process_single_video(url, label, processor, model, gdino: GroundingDetector,
                          rag: CrimeRAG, ocr: PaddleOCR, args,
                          obj_mapper: ObjectMapper | None = None) -> str | None:
    """YouTube/TikTok URL 처리 — 다운로드 후 _analyze_and_save() 공통 로직 호출."""
    vid_m = re.search(r"(?:v=|video/|shorts/)([a-zA-Z0-9_-]+)", url)
    video_id = vid_m.group(1) if vid_m else str(int(time.time()))
    res_dir   = ensure_dir(Path(args.out_dir) / "results")
    json_path = res_dir / f"ocr_results_{video_id}.json"

    if json_path.exists():
        print(f"  [Skip] {video_id}")
        return None

    print(f"  [Processing] {url}  (label={label})")
    t0 = time.time()
    try:
        v_path, info = download_video(url, Path(args.out_dir))
        title = info.get("title", "")
        duration, vlm_frames, dino_frames, ocr_frames = _get_video_frames(v_path, args)
    except Exception as e:
        print(f"  [Error] {url}: {e}")
        import traceback; traceback.print_exc()
        return None

    return _analyze_and_save(video_id, url, label, title, duration,
                              vlm_frames, dino_frames, ocr_frames,
                              processor, model, gdino, rag, ocr, args, obj_mapper, t0)


def process_local_file(file_path: Path, label, processor, model, gdino: GroundingDetector,
                        rag: CrimeRAG, ocr: PaddleOCR, args,
                        obj_mapper: ObjectMapper | None = None) -> str | None:
    """로컬 영상/이미지 파일 처리 — 다운로드 없이 파일에서 바로 프레임을 뽑아
    _analyze_and_save() 공통 로직 호출. 이미지는 시간 축이 없어 프레임 1장으로 취급."""
    ext = file_path.suffix.lower()
    video_id = file_path.stem
    res_dir   = ensure_dir(Path(args.out_dir) / "results")
    json_path = res_dir / f"ocr_results_{video_id}.json"

    if json_path.exists():
        print(f"  [Skip] {video_id}")
        return None

    print(f"  [Processing] {file_path.name}  (label={label})")
    t0 = time.time()
    try:
        if ext in IMAGE_EXTS:
            image = Image.open(file_path).convert("RGB")
            duration, vlm_frames, dino_frames, ocr_frames = _get_image_frames(image)
        else:
            duration, vlm_frames, dino_frames, ocr_frames = _get_video_frames(file_path, args)
    except Exception as e:
        print(f"  [Error] {file_path.name}: {e}")
        import traceback; traceback.print_exc()
        return None

    return _analyze_and_save(video_id, str(file_path), label, file_path.name, duration,
                              vlm_frames, dino_frames, ocr_frames,
                              processor, model, gdino, rag, ocr, args, obj_mapper, t0)


# ──────────────────────────────────────────────
# 메인
# ──────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="cybercop pipeline AdotX_v0_1 — CoT + Grounding DINO")
    parser.add_argument("--url",                  help="단일 URL")
    parser.add_argument("--file",                 help="단일 로컬 영상/이미지 파일 경로")
    parser.add_argument("--dir",                  help="영상/이미지 파일이 담긴 디렉토리 경로")
    parser.add_argument("--csv",                  default=str(Path(__file__).parent / "data" / "labels.csv"))
    parser.add_argument("--out_dir",              default="./output_AdotX_v0.1")
    parser.add_argument("--model",                default=MODEL_ID)
    parser.add_argument("--gdino_model",          default=GDINO_MODEL_ID)
    parser.add_argument("--sample_sec",           type=float, default=2.0)
    parser.add_argument("--max_frames",           type=int,   default=1000)
    parser.add_argument("--scan_sec",             type=float, default=0.5)
    parser.add_argument("--max_vlm_frames",       type=int,   default=4)
    # DINO는 싸고 빠르므로 VLM보다 더 촘촘하게 샘플링
    parser.add_argument("--dino_sec",             type=float, default=3.0,
                        help="DINO용 균등 샘플링 간격(초). VLM보다 촘촘해도 됨")
    parser.add_argument("--max_dino_frames",      type=int,   default=15,
                        help="DINO에 넘길 최대 프레임 수")
    parser.add_argument("--escalate_thr",         type=float, default=ESCALATE_THR)
    parser.add_argument("--top_k",                type=int,   default=3)
    args = parser.parse_args()

    ensure_dir(args.out_dir)

    print("[1/4] VLM 로드...")
    processor, model = load_vlm(args.model)

    print("[2/4] Grounding DINO 로드...")
    gdino = GroundingDetector(model_id=args.gdino_model)

    print("[3/4] OCR + RAG 로드...")
    ocr = load_paddleocr()
    rag = CrimeRAG(DEFAULT_DOCS_PATH)
    obj_mapper = ObjectMapper(rag.model)

    print("[4/4] 처리 시작...")

    preds = []
    if args.url:
        pred = process_single_video(args.url, "manual", processor, model, gdino, rag, ocr, args, obj_mapper)
        if pred:
            preds.append(("unknown", pred))
    elif args.file:
        file_path = Path(args.file)
        if not file_path.is_file():
            print(f"\n[Error] 파일을 찾을 수 없습니다: {args.file}")
            sys.exit(1)
        pred = process_local_file(file_path, "manual", processor, model, gdino, rag, ocr, args, obj_mapper)
        if pred:
            preds.append(("unknown", pred))
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
            pred = process_local_file(f, "manual", processor, model, gdino, rag, ocr, args, obj_mapper)
            if pred:
                preds.append(("unknown", pred))
    elif args.csv:
        with open(args.csv, "r", encoding="utf-8-sig") as f:
            for row in csv.DictReader(f):
                url = row.get("link") or row.get("url")
                gt  = row.get("label", "unknown")
                if url:
                    pred = process_single_video(url, gt, processor, model, gdino, rag, ocr, args, obj_mapper)
                    if pred is not None and gt in ("abnormal", "normal"):
                        preds.append((gt, pred))

    if preds:
        # "검토필요"를 사기로 간주해서 성능 계산
        def is_scam(p): return p in ("사기", "검토필요")
        tp = sum(1 for gt, p in preds if is_scam(p) and gt == "abnormal")
        fp = sum(1 for gt, p in preds if is_scam(p) and gt == "normal")
        fn = sum(1 for gt, p in preds if not is_scam(p) and gt == "abnormal")
        tn = sum(1 for gt, p in preds if not is_scam(p) and gt == "normal")
        rec  = tp / (tp + fn) * 100 if (tp + fn) > 0 else 0.0
        prec = tp / (tp + fp) * 100 if (tp + fp) > 0 else 0.0
        f1   = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
        print(f"\n{'='*50}")
        print(f"  [성능] n={len(preds)}  TP={tp} TN={tn} FP={fp} FN={fn}")
        print(f"  Recall={rec:.1f}%  Precision={prec:.1f}%  F1={f1:.1f}%")
        print(f"{'='*50}")


if __name__ == "__main__":
    main()
