# app/main_v0_1.py
#
# 현재 파이프라인(CoT 분류 + Grounding DINO 증거탐지 + OCR 클러스터링) 기반 서빙 API.
# cybercop_pipeline_AdotX_v0_1.py의 함수를 그대로 재사용해 CLI 배치 결과(output_AdotX_v0.1/results/*.json)와
# API 응답의 판단 로직이 항상 동일하게 유지되도록 한다.

import os
# 기본은 GPU0이지만, 실행 전 `export CUDA_VISIBLE_DEVICES=1` 등으로 다른 GPU를 지정하면 그걸 우선함
# (하드코딩해서 강제로 덮어쓰면 GPU0가 다른 프로세스로 차 있을 때 우회할 방법이 없었음)
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
# PaddleOCR가 이미 캐시된 모델도 온라인 재확인하려다 죽은 커넥션에 멈추는 문제 방지
# (cybercop_pipeline_AdotX_v0_1 import 시에도 설정되지만 명시적으로 한 번 더 지정)
os.environ.setdefault("HF_HUB_OFFLINE", "1")

import sys
import types
import importlib.machinery

if "librosa" not in sys.modules:
    fake = types.ModuleType("librosa")
    fake.__version__ = "0.0"
    fake.__spec__ = importlib.machinery.ModuleSpec("librosa", loader=None)
    sys.modules["librosa"] = fake

if "soundfile" not in sys.modules:
    fake = types.ModuleType("soundfile")
    fake.__version__ = "0.0"
    fake.__spec__ = importlib.machinery.ModuleSpec("soundfile", loader=None)
    sys.modules["soundfile"] = fake

import csv
import time
from io import StringIO
from pathlib import Path
from contextlib import asynccontextmanager

import cv2
import yt_dlp
from PIL import Image

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import JSONResponse
from fastapi.concurrency import run_in_threadpool

# ──────────────────────────────────────────────
# 경로 설정 — 저장소 루트를 모듈 검색 경로에 추가해 파이프라인 함수를 그대로 재사용
# ──────────────────────────────────────────────
BASE_DIR    = Path(__file__).resolve().parent.parent
OUT_DIR     = BASE_DIR / "downloaded_videos"
UPLOAD_DIR  = BASE_DIR / "app" / "uploaded_files"

sys.path.insert(0, str(BASE_DIR))

from cybercop_pipeline_AdotX_v0_1 import (  # noqa: E402  (경로 설정 이후 import)
    DEVICE, ESCALATE_THR, MODEL_ID, GDINO_MODEL_ID,
    load_vlm, load_paddleocr, CrimeRAG, ObjectMapper, DEFAULT_DOCS_PATH,
    run_ocr, postprocess_ocr_with_vlm, vlm_object_fallback,
)
from pipeline.frame_sampler import sample_uniform, sample_keyframe   # noqa: E402
from pipeline.vlm_classify import classify_scam                       # noqa: E402
from pipeline.grounding_detector import GroundingDetector             # noqa: E402
from pipeline.evidence_aggregator import aggregate, should_escalate   # noqa: E402
from pipeline.vocab import SCAM_EVIDENCE_VOCAB                        # noqa: E402

VIDEO_EXTS = {".mp4", ".avi", ".mkv", ".mov", ".webm", ".flv", ".ts"}
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".webp", ".tiff"}

# 프레임 샘플링 기본값 — CLI(cybercop_pipeline_AdotX_v0_1.py)의 argparse 기본값과 동일하게 맞춤
DEFAULT_SCAN_SEC       = 0.5   # 분류용 키프레임 스캔 간격
DEFAULT_MAX_VLM_FRAMES = 4
DEFAULT_DINO_SEC       = 3.0   # DINO 증거탐지 프레임 간격
DEFAULT_MAX_DINO_FRAMES = 15

# ──────────────────────────────────────────────
# 전역 모델 상태 (1회 로드)
# ──────────────────────────────────────────────
_processor  = None
_model      = None
_gdino      = None
_ocr        = None
_rag        = None
_obj_mapper = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _processor, _model, _gdino, _ocr, _rag, _obj_mapper
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

    print("[startup] VLM 로딩 중...")
    _processor, _model = await run_in_threadpool(load_vlm, MODEL_ID)
    print("[startup] Grounding DINO 로딩 중...")
    _gdino = await run_in_threadpool(GroundingDetector, GDINO_MODEL_ID)
    print("[startup] PaddleOCR 로딩 중...")
    _ocr = await run_in_threadpool(load_paddleocr)
    print("[startup] RAG 인덱스 구성 중...")
    _rag = await run_in_threadpool(CrimeRAG, DEFAULT_DOCS_PATH)
    await run_in_threadpool(_rag.build_index)
    _obj_mapper = await run_in_threadpool(ObjectMapper, _rag.model)
    print("[startup] 준비 완료.")
    yield


app = FastAPI(title="CyberCOP Video Analysis API (AdotX_v0_1: CoT + Grounding DINO)", lifespan=lifespan)


# ──────────────────────────────────────────────
# 단일 영상/이미지 분석 — cybercop_pipeline_AdotX_v0_1.process_single_video()와 동일한 판단 로직
# ──────────────────────────────────────────────
def _get_video_frames(v_path: Path) -> tuple:
    """비디오 파일에서 1~3단계에 필요한 프레임 세트 + 재생시간을 뽑는다."""
    cap = cv2.VideoCapture(str(v_path))
    duration = cap.get(cv2.CAP_PROP_FRAME_COUNT) / (cap.get(cv2.CAP_PROP_FPS) or 30.0)
    cap.release()
    vlm_frames  = sample_keyframe(v_path, DEFAULT_SCAN_SEC, DEFAULT_MAX_VLM_FRAMES)
    dino_frames = sample_uniform(v_path, every_n=DEFAULT_DINO_SEC, max_frames=DEFAULT_MAX_DINO_FRAMES)
    ocr_frames  = sample_uniform(v_path, DEFAULT_SCAN_SEC * 4, 1000)
    return duration, vlm_frames, dino_frames, ocr_frames


def _get_image_frames(image: Image.Image) -> tuple:
    """정지 이미지 1장 — 시간 축이 없어서 1~3단계 모두 같은 단일 프레임을 그대로 재사용."""
    frame = [(0.0, image)]
    return 0.0, frame, frame, frame


def _analyze_frames(duration: float, vlm_frames: list, dino_frames: list, ocr_frames: list,
                     title: str, top_k: int) -> dict:
    # ── 1단계: CoT 분류 ──
    cls_label, cls_summary = classify_scam(_processor, _model, vlm_frames, title, device=DEVICE)

    # ── 2단계: DINO 증거탐지 (모든 입력에서 실행, 정상→검토필요 에스컬레이션 위해) ──
    frame_dets  = _gdino.detect_batch(dino_frames, SCAM_EVIDENCE_VOCAB)
    evidence    = aggregate(frame_dets, video_duration=duration)

    final_label = cls_label
    if cls_label == "정상" and should_escalate(evidence, threshold=ESCALATE_THR):
        final_label = "검토필요"

    result = {
        "cls_label":        cls_label,
        "final_label":      final_label,
        "classify_summary": cls_summary,
        "grounding":         evidence,
        "objects":          [],
        "ocr":              [],
        "ocr_text":         "",
        "ocr_candidates":   [],
        "rag":              [],
    }

    # ── 3단계: 사기/검토필요 판정 영상만 OCR + 상세 오브젝트 분석 ──
    # 객체는 DINO 결과(VOCAB, evidence["top_labels"])와 VLM 폴백 결과(ALLOWED_OBJECTS 매핑)를
    # 둘 다 합침 — DINO는 정밀하지만 좁은 어휘라 놓치는 게 많고, VLM은 넓지만 부정확할 수 있어서
    # 사기로 확정된 영상은 증거를 최대한 남기기 위해 둘 다 사용 (정상은 3단계 자체를 스킵하니 비용 문제 없음)
    if final_label in ("사기", "검토필요"):
        detail = run_ocr(ocr_frames, _ocr)
        corrected_ocr = postprocess_ocr_with_vlm(_processor, _model, detail["ocr_spans"], DEVICE)
        ocr_all = " | ".join(s["text"] for s in corrected_ocr)

        fallback_objs = vlm_object_fallback(vlm_frames, _processor, _model, DEVICE)
        if _obj_mapper is not None:
            fallback_objs = _obj_mapper.map(fallback_objs)
        objects_out = list(dict.fromkeys(evidence["top_labels"] + fallback_objs))
        obj_all = ", ".join(objects_out)

        mapped  = _rag.search(f"[OCR]: {ocr_all} | [Objects]: {obj_all}", top_k=top_k)
        result.update({
            "objects":        objects_out,
            "ocr":            corrected_ocr,
            "ocr_text":       ocr_all,
            "ocr_candidates": detail["ocr_candidates"],
            "rag":            mapped,
        })

    # 외부 소비 편의를 위한 2단계 요약 라벨 (사기/검토필요 → abnormal, 정상 → normal)
    result["label"] = "abnormal" if final_label in ("사기", "검토필요") else "normal"
    return result


def _analyze_video(v_path: Path, title: str, top_k: int) -> dict:
    duration, vlm_frames, dino_frames, ocr_frames = _get_video_frames(v_path)
    return _analyze_frames(duration, vlm_frames, dino_frames, ocr_frames, title, top_k)


def _analyze_image(image: Image.Image, title: str, top_k: int) -> dict:
    duration, vlm_frames, dino_frames, ocr_frames = _get_image_frames(image)
    return _analyze_frames(duration, vlm_frames, dino_frames, ocr_frames, title, top_k)


def _download(url: str) -> tuple[Path, dict]:
    outtmpl = str(OUT_DIR / "%(title).80s_%(id)s.%(ext)s")
    with yt_dlp.YoutubeDL({"outtmpl": outtmpl, "format": "best[ext=mp4]/best",
                            "noplaylist": True, "quiet": True, "no_warnings": True}) as ydl:
        info = ydl.extract_info(url, download=True)
    v_path = Path(ydl.prepare_filename(info))
    if not v_path.exists():
        v_path = list(OUT_DIR.glob(f"*{info['id']}*"))[0]
    return v_path, info


# ──────────────────────────────────────────────
# 엔드포인트
# ──────────────────────────────────────────────
@app.post("/api/video")
async def analyze_url(
    url:    str = Form(..., description="YouTube / TikTok 영상 URL"),
    top_k:  int = Form(3, description="RAG 검색 상위 k"),
):
    """YouTube / TikTok URL → 사이버범죄 분석 JSON 반환 (CoT + DINO)"""
    def _process():
        v_path, info = _download(url)
        result = _analyze_video(v_path, info.get("title", ""), top_k)
        return {"id": info.get("id", str(int(time.time()))), "title": info.get("title"), **result}

    try:
        payload = await run_in_threadpool(_process)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    return JSONResponse({"message": "success", "result": payload})


@app.post("/api/video/upload")
async def analyze_upload(
    file:   UploadFile = File(..., description="영상 또는 이미지 파일"),
    top_k:  int         = Form(3),
):
    """영상 / 이미지 파일 업로드 → 사이버범죄 분석 JSON 반환 (CoT + DINO)"""
    if not file.filename:
        raise HTTPException(status_code=400, detail="파일 이름이 없습니다.")
    ext = Path(file.filename).suffix.lower()
    if ext not in VIDEO_EXTS | IMAGE_EXTS:
        raise HTTPException(status_code=400, detail=f"지원하지 않는 파일 형식입니다. ({ext})")

    save_path = UPLOAD_DIR / file.filename
    with save_path.open("wb") as buf:
        import shutil
        shutil.copyfileobj(file.file, buf)

    def _process():
        if ext in IMAGE_EXTS:
            image = Image.open(save_path).convert("RGB")
            result = _analyze_image(image, file.filename, top_k)
        else:
            result = _analyze_video(save_path, file.filename, top_k)
        return {"id": save_path.stem, "title": file.filename, **result}

    try:
        payload = await run_in_threadpool(_process)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    return JSONResponse({"message": "success", "result": payload})


@app.post("/api/video/csv")
async def analyze_csv(
    file:  UploadFile = File(..., description="link, label 컬럼을 가진 CSV 파일"),
    top_k: int         = Form(3),
):
    """CSV 파일 배치 업로드 (link, label 컬럼) → 전체 분석 결과 JSON 반환"""
    if not file.filename or Path(file.filename).suffix.lower() != ".csv":
        raise HTTPException(status_code=400, detail="CSV 파일(.csv)만 지원합니다.")

    content = (await file.read()).decode("utf-8")
    reader  = csv.DictReader(StringIO(content))
    rows    = list(reader)
    if not rows:
        raise HTTPException(status_code=400, detail="CSV 파일이 비어 있습니다.")
    if "link" not in rows[0]:
        raise HTTPException(status_code=400, detail="CSV에 'link' 컬럼이 필요합니다.")

    def _process_all():
        results = []
        for row in rows:
            url   = row["link"]
            gt    = row.get("label", "unknown")
            try:
                v_path, info = _download(url)
                result = _analyze_video(v_path, info.get("title", ""), top_k)
                results.append({
                    "id":       info.get("id", str(int(time.time()))),
                    "title":    info.get("title"),
                    "gt_label": gt,
                    **result,
                    "error": None,
                })
            except Exception as e:
                results.append({"url": url, "gt_label": gt, "error": str(e)})
        return results

    try:
        results = await run_in_threadpool(_process_all)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    return JSONResponse({"message": "success", "count": len(results), "results": results})
