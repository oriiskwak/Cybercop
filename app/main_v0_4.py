"""Production API for the CyberCOP v0.4 pipeline.

v0.3 API(legacy/app/main_v0_3.py) 대비 변경점:
  - 파이프라인을 cybercop_pipeline_AdotX_v0_4로 교체. 자체 RAG(`rag` 필드, `top_k` 인자)가 사라지고
    risk_agent 위험도·사기유형(`risk_assessment`)이 들어감.
  - OCR 프레임 인자(`sample_sec`, `max_frames`)를 받음. 위험도의 피해자 수·피해금액은 받지 않고
    파이프라인 기본값(피해자 0=정보 없음, 금액은 OCR에서 추출 시도)으로 계산함.
  - 응답의 탐지 객체를 3필드(`scam_objects`/`general_objects`/`icon_emoji_objects`)로 나누고,
    OCR은 전체 텍스트(`ocr_after`)만 담음(v0.3의 `ocr`/`ocr_before`/`ocr_candidates`는 응답에서 뺌).
  - 검토필요 판정이면 그 사유를 `review_reason`에 한 문장으로 담음(그 외에는 null).
  - 업로드 파일은 원래 파일명으로 저장해 VLM 분류 프롬프트의 제목이 CLI(--file)와 같게 함.
"""

from __future__ import annotations

import asyncio
import csv
import json
import logging
import os
import re
import secrets
import tempfile
import time
import uuid
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from io import BytesIO, StringIO
from pathlib import Path
from types import SimpleNamespace
from typing import Annotated
from urllib.parse import urlparse

from fastapi import Depends, FastAPI, File, Form, Header, HTTPException, UploadFile, status
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import JSONResponse
from PIL import Image, UnidentifiedImageError
from sentence_transformers import SentenceTransformer

import cybercop_pipeline_AdotX_v0_4 as core
import text_risk


logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO").upper(),
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)
logger = logging.getLogger("cybercop.api")


def _env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.lower() in {"1", "true", "yes", "on"}


WORK_DIR = Path(os.getenv("CYBERCOP_WORK_DIR", "/tmp/cybercop"))
LOAD_MODELS_ON_STARTUP = _env_bool("LOAD_MODELS_ON_STARTUP", True)
MAX_UPLOAD_BYTES = int(os.getenv("MAX_UPLOAD_BYTES", str(250 * 1024 * 1024)))
MAX_CSV_ROWS = int(os.getenv("MAX_CSV_ROWS", "100"))
MAX_CONCURRENT_ANALYSES = int(os.getenv("MAX_CONCURRENT_ANALYSES", "1"))
# OCR 프레임 인자 기본값은 CLI(--sample_sec/--max_frames) 기본값과 동일.
DEFAULT_SAMPLE_SEC = float(os.getenv("DEFAULT_SAMPLE_SEC", "2.0"))
DEFAULT_MAX_FRAMES = int(os.getenv("DEFAULT_MAX_FRAMES", "1000"))
MAX_FRAMES_LIMIT = int(os.getenv("MAX_FRAMES_LIMIT", "1000"))
API_KEY = os.getenv("CYBERCOP_API_KEY")
ALLOWED_URL_HOSTS = {
    host.strip().lower()
    for host in os.getenv(
        "ALLOWED_URL_HOSTS",
        "youtube.com,youtu.be,tiktok.com",
    ).split(",")
    if host.strip()
}
Image.MAX_IMAGE_PIXELS = int(os.getenv("MAX_IMAGE_PIXELS", "50000000"))

if min(
    MAX_UPLOAD_BYTES,
    MAX_CSV_ROWS,
    MAX_CONCURRENT_ANALYSES,
    DEFAULT_SAMPLE_SEC,
    DEFAULT_MAX_FRAMES,
    MAX_FRAMES_LIMIT,
) <= 0:
    raise RuntimeError("API 크기/동시성/프레임 제한 환경변수는 0보다 커야 합니다.")
if DEFAULT_MAX_FRAMES > MAX_FRAMES_LIMIT:
    raise RuntimeError("DEFAULT_MAX_FRAMES는 MAX_FRAMES_LIMIT 이하여야 합니다.")


@dataclass
class ResourceState:
    processor: object | None = None
    model: object | None = None
    ocr: object | None = None
    object_mapper: core.ObjectMapper | None = None
    risk_agent: dict = field(default_factory=dict)
    ready: bool = False
    startup_error: str | None = None


resources = ResourceState()
analysis_slots = asyncio.Semaphore(MAX_CONCURRENT_ANALYSES)


def _load_resources() -> None:
    resources.ready = False
    resources.startup_error = None
    try:
        processor, model = core.load_vlm(core.MODEL_ID)
        ocr = core.load_ocr()
        embed_model = SentenceTransformer(core.DEFAULT_EMBED_MODEL, device=core.DEVICE)
        object_mapper = core.ObjectMapper(embed_model)
        # risk_agent 모델은 첫 assess_risk() 호출 때 로드됨. 첫 요청이 느려지지 않도록 미리 한 번 호출.
        warmup = text_risk.assess_risk({"classify_summary": "warmup"})
    except Exception as exc:
        resources.startup_error = f"{type(exc).__name__}: {exc}"
        raise

    resources.processor = processor
    resources.model = model
    resources.ocr = ocr
    resources.object_mapper = object_mapper
    resources.risk_agent = {
        "available": bool(warmup.get("available")),
        "text_risk": bool((warmup.get("text_risk") or {}).get("available")),
        "guideline_risk": bool((warmup.get("guideline_risk") or {}).get("available")),
        "error": warmup.get("error"),
    }
    resources.ready = True


@asynccontextmanager
async def lifespan(_: FastAPI):
    WORK_DIR.mkdir(parents=True, exist_ok=True)
    if LOAD_MODELS_ON_STARTUP:
        logger.info("CyberCOP 모델을 로드합니다.")
        try:
            await run_in_threadpool(_load_resources)
        except Exception:
            logger.exception("CyberCOP 초기화에 실패했습니다.")
            raise
        logger.info("CyberCOP API가 준비되었습니다. risk_agent=%s", resources.risk_agent)
    else:
        logger.warning("LOAD_MODELS_ON_STARTUP=0: API가 비준비 상태로 시작됩니다.")
    try:
        yield
    finally:
        resources.ready = False


app = FastAPI(
    title="CyberCOP Analysis API",
    version="0.4.0",
    description="Gemma 4 VLM, RapidOCR, BGE-m3, risk_agent 기반 사이버 범죄 분석 API",
    lifespan=lifespan,
)


async def require_api_key(x_api_key: str | None = Header(default=None)) -> None:
    if API_KEY and (x_api_key is None or not secrets.compare_digest(x_api_key, API_KEY)):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="invalid API key")


AUTH = [Depends(require_api_key)]


@app.get("/health/live", tags=["health"])
async def health_live() -> dict:
    return {"status": "ok"}


@app.get("/health/ready", tags=["health"])
async def health_ready():
    payload = {
        "status": "ready" if resources.ready else "not_ready",
        "device": core.DEVICE,
        "risk_agent": resources.risk_agent,
        "startup_error": resources.startup_error,
    }
    return JSONResponse(
        payload,
        status_code=status.HTTP_200_OK if resources.ready else status.HTTP_503_SERVICE_UNAVAILABLE,
    )


@app.get("/api/info", dependencies=AUTH)
async def api_info() -> dict:
    return {
        "pipeline_version": "0.4",
        "vlm_model": core.MODEL_ID,
        "embedding_model": core.DEFAULT_EMBED_MODEL,
        "ocr": "RapidOCR PP-OCRv6 detection(MEDIUM) + PP-OCRv5 Korean recognition, ONNX Runtime CPU",
        "scam_object_vocab_size": len(core.SCAM_EVIDENCE_LABELS),
        "ocr_defaults": {"sample_sec": DEFAULT_SAMPLE_SEC, "max_frames": DEFAULT_MAX_FRAMES,
                         "max_frames_limit": MAX_FRAMES_LIMIT},
        "risk_agent": resources.risk_agent,
        "device": core.DEVICE,
        "ready": resources.ready,
    }


def _ensure_ready() -> None:
    if not resources.ready:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="analysis models are not ready",
        )


@dataclass(frozen=True)
class AnalysisOptions:
    sample_sec: float = DEFAULT_SAMPLE_SEC
    max_frames: int = DEFAULT_MAX_FRAMES


def _pipeline_args(out_dir: Path, opts: AnalysisOptions) -> SimpleNamespace:
    # 판정 쪽 프레임(분류용 대표 4장, 사기 객체·이모티콘아이콘용 영상 전체 고르게 15장)은 고정하고,
    # 요청으로 받는 sample_sec/max_frames는 OCR 프레임에만 사용.
    return SimpleNamespace(
        out_dir=str(out_dir),
        scan_sec=0.5,
        max_vlm_frames=4,
        evidence_sec=3.0,
        max_evidence_frames=15,
        sample_sec=opts.sample_sec,
        max_frames=opts.max_frames,
        escalate_thr=core.ESCALATE_THR,
        victim_count=None,
        total_loss_won=None,
    )


def _read_result(out_dir: Path) -> dict:
    paths = list((out_dir / "results").glob("ocr_results_*.json"))
    if len(paths) != 1:
        raise RuntimeError(f"expected one result JSON, found {len(paths)}")
    with paths[0].open(encoding="utf-8") as f:
        return json.load(f)


def _normalize_result(
    payload: dict,
    opts: AnalysisOptions,
    *,
    result_id: str | None = None,
    title: str | None = None,
    source: str | None = None,
) -> dict:
    input_label = payload.pop("label", "manual")
    final_label = payload.get("final_label")
    payload["input_label"] = input_label
    payload["label"] = "abnormal" if final_label in {"사기", "검토필요"} else "normal"
    payload["id"] = result_id or payload.get("id")
    if title is not None:
        payload["title"] = title
    if source is not None:
        payload["url"] = source
    for key, default in {
        "objects": [],
        "scam_evidence": [],
        "ocr_after": "",
        "risk_assessment": None,
        "skipped": False,
    }.items():
        payload.setdefault(key, default)
    # OCR은 전체 텍스트(ocr_after, 위험도 입력과 같은 값)만 응답. 프레임별/줄 단위 OCR은
    # 같은 글자가 반복돼 응답만 커져서 제외(파이프라인 결과 JSON에는 남아 있음).
    n_ocr_frames = len(payload.get("ocr_frames") or [])
    for key in ("ocr_before", "ocr_frames", "ocr", "ocr_candidates"):
        payload.pop(key, None)

    # 탐지 객체 3필드. scam_objects는 scam_evidence와 같은 값(빈도순 상위 5개)이고,
    # 탐지된 사기 객체 전체와 빈도는 grounding.labels에 있음.
    scam_objects = list(payload["scam_evidence"])
    payload["scam_objects"] = scam_objects
    payload.setdefault("general_objects", [o for o in payload["objects"] if o not in set(scam_objects)])
    payload.setdefault("icon_emoji_objects", {"emoticons": [], "icons": []})
    payload["ocr_settings"] = {
        "sample_sec": opts.sample_sec,
        "max_frames": opts.max_frames,
        "n_frames": n_ocr_frames,
    }
    payload["review_reason"] = _review_reason(payload)
    return payload


def _review_reason(payload: dict) -> str | None:
    """final_label이 검토필요일 때 왜 그렇게 됐는지 한 문장으로 설명. 판정 규칙은 파이프라인
    _analyze_and_save()의 에스컬레이션과 동일(임계값 core.ESCALATE_THR). 검토필요가 아니면 None."""
    if payload.get("final_label") != "검토필요":
        return None
    cls_label = payload.get("cls_label")
    grounding = payload.get("grounding") or {}
    score = float(grounding.get("evidence_score") or 0.0)
    labels = grounding.get("top_labels") or []
    thr = core.ESCALATE_THR
    if cls_label == "정상":
        return (f"1차 분류는 정상이었지만 사기 객체 점수가 {score:.2f}점으로 기준({thr}점) 이상이라 "
                f"검토필요로 올렸습니다. 근거: {', '.join(labels[:3])}")
    if cls_label == "사기":
        found = f"탐지된 사기 객체: {', '.join(labels[:3])}" if labels else "탐지된 사기 객체 없음"
        return (f"1차 분류는 사기였지만 사기 객체 점수가 {score:.2f}점으로 기준({thr}점)에 못 미쳐 "
                f"검토필요로 내렸습니다(뒷받침 증거 부족). {found}")
    return "1차 분류 결과를 해석하지 못해 검토필요로 표시했습니다."


def _run_local(path: Path, opts: AnalysisOptions, out_dir: Path) -> dict:
    prediction = core.process_local_file(
        path,
        "manual",
        resources.processor,
        resources.model,
        resources.ocr,
        _pipeline_args(out_dir, opts),
        resources.object_mapper,
    )
    if prediction is None:
        raise RuntimeError("local analysis did not produce a result")
    payload = _normalize_result(_read_result(out_dir), opts, result_id=path.stem, title=path.name)
    payload["url"] = None
    return payload


def _run_url(url: str, opts: AnalysisOptions, out_dir: Path) -> dict:
    prediction = core.process_single_video(
        url,
        "manual",
        resources.processor,
        resources.model,
        resources.ocr,
        _pipeline_args(out_dir, opts),
        resources.object_mapper,
    )
    if prediction is None:
        raise RuntimeError("URL analysis did not produce a result")
    return _normalize_result(_read_result(out_dir), opts, source=url)


def _validate_url(url: str) -> str:
    value = url.strip()
    parsed = urlparse(value)
    host = (parsed.hostname or "").lower().rstrip(".")
    try:
        port = parsed.port
    except ValueError as exc:
        raise HTTPException(status_code=400, detail="유효하지 않은 URL 포트입니다.") from exc
    if (
        parsed.scheme != "https"
        or not host
        or parsed.username
        or parsed.password
        or port not in {None, 443}
    ):
        raise HTTPException(status_code=400, detail="HTTPS YouTube/TikTok URL만 허용합니다.")
    allowed = any(host == base or host.endswith("." + base) for base in ALLOWED_URL_HOSTS)
    if not allowed:
        raise HTTPException(status_code=400, detail="허용되지 않은 URL 호스트입니다.")
    return value


def _safe_name(filename: str | None) -> str:
    name = Path(filename or "").name
    if not name:
        raise HTTPException(status_code=400, detail="파일 이름이 없습니다.")
    return name


def _safe_stem(value: str) -> str:
    return re.sub(r"[^\w.-]+", "_", value, flags=re.UNICODE).strip("._")[:80]


async def _read_limited(upload: UploadFile, limit: int) -> bytes:
    chunks: list[bytes] = []
    size = 0
    while chunk := await upload.read(1024 * 1024):
        size += len(chunk)
        if size > limit:
            raise HTTPException(status_code=413, detail=f"업로드 제한({limit} bytes)을 초과했습니다.")
        chunks.append(chunk)
    return b"".join(chunks)


async def _save_limited(upload: UploadFile, path: Path, limit: int) -> int:
    size = 0
    with path.open("wb") as output:
        while chunk := await upload.read(1024 * 1024):
            size += len(chunk)
            if size > limit:
                raise HTTPException(status_code=413, detail=f"업로드 제한({limit} bytes)을 초과했습니다.")
            output.write(chunk)
    return size


def _validate_image(data: bytes, filename: str) -> None:
    if Path(filename).suffix.lower() not in core.IMAGE_EXTS:
        raise HTTPException(status_code=400, detail=f"이미지 형식만 지원합니다: {filename}")
    try:
        with Image.open(BytesIO(data)) as image:
            image.verify()
    except (UnidentifiedImageError, OSError, Image.DecompressionBombError) as exc:
        raise HTTPException(status_code=400, detail=f"유효하지 않은 이미지입니다: {filename}") from exc


def _public_error(request_id: str, exc: Exception) -> HTTPException:
    logger.exception("분석 실패 request_id=%s", request_id)
    return HTTPException(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        detail={"message": "analysis failed", "request_id": request_id},
    )


def _options(sample_sec: float, max_frames: int) -> AnalysisOptions:
    return AnalysisOptions(sample_sec=sample_sec, max_frames=max_frames)


SampleSec = Annotated[float, Form(gt=0, le=60, description="OCR 프레임 추출 간격(초). 영상 길이 ÷ 이 값 = 프레임별 OCR 개수")]
MaxFrames = Annotated[int, Form(ge=1, le=MAX_FRAMES_LIMIT, description="OCR 최대 프레임 수. 넘으면 영상 앞부분부터 이 수만큼만 OCR")]


@app.post("/api/video", dependencies=AUTH)
async def analyze_url(
    url: Annotated[str, Form(description="YouTube/TikTok HTTPS URL")],
    sample_sec: SampleSec = DEFAULT_SAMPLE_SEC,
    max_frames: MaxFrames = DEFAULT_MAX_FRAMES,
):
    _ensure_ready()
    safe_url = _validate_url(url)
    opts = _options(sample_sec, max_frames)
    request_id = uuid.uuid4().hex
    try:
        async with analysis_slots:
            with tempfile.TemporaryDirectory(prefix="url-", dir=WORK_DIR) as temp:
                payload = await run_in_threadpool(_run_url, safe_url, opts, Path(temp) / "output")
    except HTTPException:
        raise
    except Exception as exc:
        raise _public_error(request_id, exc) from exc
    return {"message": "success", "request_id": request_id, "result": payload}


@app.post("/api/video/upload", dependencies=AUTH)
async def analyze_upload(
    file: UploadFile = File(..., description="영상 또는 이미지 파일"),
    sample_sec: SampleSec = DEFAULT_SAMPLE_SEC,
    max_frames: MaxFrames = DEFAULT_MAX_FRAMES,
):
    _ensure_ready()
    original_name = _safe_name(file.filename)
    ext = Path(original_name).suffix.lower()
    if ext not in core.VIDEO_EXTS | core.IMAGE_EXTS:
        raise HTTPException(status_code=400, detail=f"지원하지 않는 파일 형식입니다: {ext}")
    opts = _options(sample_sec, max_frames)
    request_id = uuid.uuid4().hex
    try:
        with tempfile.TemporaryDirectory(prefix="upload-", dir=WORK_DIR) as temp:
            temp_path = Path(temp)
            input_dir = temp_path / "input"
            input_dir.mkdir()
            # 원래 파일명으로 저장 — 파이프라인이 파일명을 영상 제목으로 VLM 분류 프롬프트에 넣음(CLI --file과 동일).
            upload_path = input_dir / f"{_safe_stem(Path(original_name).stem) or uuid.uuid4().hex}{ext}"
            await _save_limited(file, upload_path, MAX_UPLOAD_BYTES)
            if ext in core.IMAGE_EXTS:
                _validate_image(upload_path.read_bytes(), original_name)
            async with analysis_slots:
                payload = await run_in_threadpool(_run_local, upload_path, opts, temp_path / "output")
    except HTTPException:
        raise
    except Exception as exc:
        raise _public_error(request_id, exc) from exc
    finally:
        await file.close()
    payload["title"] = original_name
    return {"message": "success", "request_id": request_id, "result": payload}


@app.post("/api/video/csv", dependencies=AUTH)
async def analyze_csv(
    file: UploadFile = File(..., description="label/link 또는 label/url 컬럼 CSV"),
    sample_sec: SampleSec = DEFAULT_SAMPLE_SEC,
    max_frames: MaxFrames = DEFAULT_MAX_FRAMES,
):
    _ensure_ready()
    if Path(_safe_name(file.filename)).suffix.lower() != ".csv":
        raise HTTPException(status_code=400, detail="CSV 파일만 지원합니다.")
    try:
        raw = await _read_limited(file, 2 * 1024 * 1024)
        try:
            text = raw.decode("utf-8-sig")
        except UnicodeDecodeError as exc:
            raise HTTPException(status_code=400, detail="CSV는 UTF-8 인코딩이어야 합니다.") from exc
        rows = list(csv.DictReader(StringIO(text)))
    finally:
        await file.close()

    if not rows:
        raise HTTPException(status_code=400, detail="CSV가 비어 있습니다.")
    if len(rows) > MAX_CSV_ROWS:
        raise HTTPException(status_code=413, detail="CSV 행 제한을 초과했습니다.")
    if not ({"link", "url"} & set(rows[0])):
        raise HTTPException(status_code=400, detail="CSV에 link 또는 url 컬럼이 필요합니다.")

    opts = _options(sample_sec, max_frames)
    request_id = uuid.uuid4().hex

    def process_rows() -> list[dict]:
        results = []
        for index, row in enumerate(rows, start=2):
            candidate = row.get("link") or row.get("url") or ""
            try:
                safe_url = _validate_url(candidate)
                with tempfile.TemporaryDirectory(prefix="csv-", dir=WORK_DIR) as temp:
                    payload = _run_url(safe_url, opts, Path(temp) / "output")
                payload["input_label"] = row.get("label", "unknown")
                results.append(payload)
            except HTTPException as exc:
                results.append({"row": index, "url": candidate, "error": exc.detail})
            except Exception:
                logger.exception("CSV 행 분석 실패 request_id=%s row=%s", request_id, index)
                results.append({"row": index, "url": candidate, "error": "analysis failed"})
        return results

    try:
        async with analysis_slots:
            started = time.monotonic()
            results = await run_in_threadpool(process_rows)
            elapsed = round(time.monotonic() - started, 2)
    except Exception as exc:
        raise _public_error(request_id, exc) from exc

    return {
        "message": "success",
        "request_id": request_id,
        "count": len(results),
        "elapsed_seconds": elapsed,
        "results": results,
    }
