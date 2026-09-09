"""Production API for the CyberCOP v0.3 pipeline."""

from __future__ import annotations

import asyncio
import csv
import logging
import os
import re
import secrets
import tempfile
import time
import uuid
import zipfile
from contextlib import asynccontextmanager
from dataclasses import dataclass
from io import BytesIO, StringIO
from pathlib import Path
from types import SimpleNamespace
from urllib.parse import urlparse

from fastapi import Depends, FastAPI, File, Form, Header, HTTPException, UploadFile, status
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import JSONResponse
from PIL import Image, UnidentifiedImageError

import cybercop_pipeline_AdotX_v0_3 as core


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
MAX_ARCHIVE_BYTES = int(os.getenv("MAX_ARCHIVE_BYTES", str(100 * 1024 * 1024)))
MAX_SEQUENCE_BYTES = int(os.getenv("MAX_SEQUENCE_BYTES", str(200 * 1024 * 1024)))
MAX_SEQUENCE_IMAGES = int(os.getenv("MAX_SEQUENCE_IMAGES", "50"))
MAX_CSV_ROWS = int(os.getenv("MAX_CSV_ROWS", "100"))
MAX_CONCURRENT_ANALYSES = int(os.getenv("MAX_CONCURRENT_ANALYSES", "1"))
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
    MAX_ARCHIVE_BYTES,
    MAX_SEQUENCE_BYTES,
    MAX_SEQUENCE_IMAGES,
    MAX_CSV_ROWS,
    MAX_CONCURRENT_ANALYSES,
) <= 0:
    raise RuntimeError("API 크기/동시성 제한 환경변수는 0보다 커야 합니다.")


@dataclass
class ResourceState:
    processor: object | None = None
    model: object | None = None
    ocr: object | None = None
    rag: core.CrimeRAG | None = None
    object_mapper: core.ObjectMapper | None = None
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
        rag = core.CrimeRAG(core.DEFAULT_DOCS_PATH)
        rag.build_index()
        object_mapper = core.ObjectMapper(rag.model)
    except Exception as exc:
        resources.startup_error = f"{type(exc).__name__}: {exc}"
        raise

    resources.processor = processor
    resources.model = model
    resources.ocr = ocr
    resources.rag = rag
    resources.object_mapper = object_mapper
    resources.ready = True


@asynccontextmanager
async def lifespan(_: FastAPI):
    WORK_DIR.mkdir(parents=True, exist_ok=True)
    if LOAD_MODELS_ON_STARTUP:
        logger.info("CyberCOP 모델과 RAG 인덱스를 로드합니다.")
        try:
            await run_in_threadpool(_load_resources)
        except Exception:
            logger.exception("CyberCOP 초기화에 실패했습니다.")
            raise
        logger.info("CyberCOP API가 준비되었습니다.")
    else:
        logger.warning("LOAD_MODELS_ON_STARTUP=0: API가 비준비 상태로 시작됩니다.")
    try:
        yield
    finally:
        resources.ready = False


app = FastAPI(
    title="CyberCOP Analysis API",
    version="0.3.0",
    description="Gemma 4 VLM, RapidOCR, BGE-m3/FAISS 기반 사이버 범죄 분석 API",
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
        "startup_error": resources.startup_error,
    }
    return JSONResponse(
        payload,
        status_code=status.HTTP_200_OK if resources.ready else status.HTTP_503_SERVICE_UNAVAILABLE,
    )


@app.get("/api/info", dependencies=AUTH)
async def api_info() -> dict:
    return {
        "pipeline_version": "0.3",
        "vlm_model": core.MODEL_ID,
        "embedding_model": core.DEFAULT_EMBED_MODEL,
        "ocr": "RapidOCR PP-OCRv5 (Korean recognition, ONNX Runtime CPU)",
        "device": core.DEVICE,
        "ready": resources.ready,
    }


def _ensure_ready() -> None:
    if not resources.ready:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="analysis models are not ready",
        )


def _pipeline_args(out_dir: Path, top_k: int) -> SimpleNamespace:
    return SimpleNamespace(
        out_dir=str(out_dir),
        scan_sec=0.5,
        max_vlm_frames=4,
        evidence_sec=3.0,
        max_evidence_frames=15,
        sample_sec=2.0,
        max_frames=1000,
        escalate_thr=core.ESCALATE_THR,
        top_k=top_k,
    )


def _read_result(out_dir: Path) -> dict:
    paths = list((out_dir / "results").glob("ocr_results_*.json"))
    if len(paths) != 1:
        raise RuntimeError(f"expected one result JSON, found {len(paths)}")
    import json

    with paths[0].open(encoding="utf-8") as f:
        return json.load(f)


def _normalize_result(
    payload: dict,
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
    for field, default in {
        "objects": [],
        "scam_evidence": [],
        "ocr_before": [],
        "ocr": [],
        "ocr_after": "",
        "ocr_candidates": [],
        "rag": [],
        "skipped": False,
    }.items():
        payload.setdefault(field, default)
    return payload


def _run_local(path: Path, original_name: str, top_k: int, out_dir: Path) -> dict:
    args = _pipeline_args(out_dir, top_k)
    prediction = core.process_local_file(
        path,
        "manual",
        resources.processor,
        resources.model,
        resources.rag,
        resources.ocr,
        args,
        resources.object_mapper,
    )
    if prediction is None:
        raise RuntimeError("local analysis did not produce a result")
    payload = _normalize_result(
        _read_result(out_dir),
        result_id=Path(original_name).stem,
        title=original_name,
    )
    payload["url"] = None
    return payload


def _run_url(url: str, top_k: int, out_dir: Path) -> dict:
    args = _pipeline_args(out_dir, top_k)
    prediction = core.process_single_video(
        url,
        "manual",
        resources.processor,
        resources.model,
        resources.rag,
        resources.ocr,
        args,
        resources.object_mapper,
    )
    if prediction is None:
        raise RuntimeError("URL analysis did not produce a result")
    return _normalize_result(_read_result(out_dir), source=url)


def _run_sequence(seq_dir: Path, sequence_id: str, top_k: int, out_dir: Path) -> dict:
    args = _pipeline_args(out_dir, top_k)
    prediction = core.process_image_sequence(
        seq_dir,
        "manual",
        resources.processor,
        resources.model,
        resources.rag,
        resources.ocr,
        args,
        resources.object_mapper,
    )
    if prediction is None:
        raise RuntimeError("sequence analysis did not produce a result")
    payload = _normalize_result(
        _read_result(out_dir),
        result_id=sequence_id,
        title=sequence_id,
    )
    payload["url"] = None
    payload["n_images"] = len(
        [p for p in seq_dir.iterdir() if p.suffix.lower() in core.IMAGE_EXTS]
    )
    return payload


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


def _safe_sequence_id(value: str | None) -> str:
    if not value:
        return f"seq_{uuid.uuid4().hex[:12]}"
    cleaned = re.sub(r"[^\w.-]+", "_", value, flags=re.UNICODE).strip("._")[:80]
    if not cleaned:
        raise HTTPException(status_code=400, detail="유효한 sequence_id가 필요합니다.")
    return cleaned


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


@app.post("/api/video", dependencies=AUTH)
async def analyze_url(
    url: str = Form(..., description="YouTube/TikTok HTTPS URL"),
    top_k: int = Form(3, ge=1, le=15),
):
    _ensure_ready()
    safe_url = _validate_url(url)
    request_id = uuid.uuid4().hex
    try:
        async with analysis_slots:
            with tempfile.TemporaryDirectory(prefix="url-", dir=WORK_DIR) as temp:
                out_dir = Path(temp) / "output"
                payload = await run_in_threadpool(_run_url, safe_url, top_k, out_dir)
    except HTTPException:
        raise
    except Exception as exc:
        raise _public_error(request_id, exc) from exc
    return {"message": "success", "request_id": request_id, "result": payload}


@app.post("/api/video/upload", dependencies=AUTH)
async def analyze_upload(
    file: UploadFile = File(..., description="영상 또는 이미지 파일"),
    top_k: int = Form(3, ge=1, le=15),
):
    _ensure_ready()
    original_name = _safe_name(file.filename)
    ext = Path(original_name).suffix.lower()
    if ext not in core.VIDEO_EXTS | core.IMAGE_EXTS:
        raise HTTPException(status_code=400, detail=f"지원하지 않는 파일 형식입니다: {ext}")
    request_id = uuid.uuid4().hex
    try:
        with tempfile.TemporaryDirectory(prefix="upload-", dir=WORK_DIR) as temp:
            temp_path = Path(temp)
            upload_path = temp_path / f"{uuid.uuid4().hex}{ext}"
            await _save_limited(file, upload_path, MAX_UPLOAD_BYTES)
            if ext in core.IMAGE_EXTS:
                _validate_image(upload_path.read_bytes(), original_name)
            async with analysis_slots:
                payload = await run_in_threadpool(
                    _run_local,
                    upload_path,
                    original_name,
                    top_k,
                    temp_path / "output",
                )
    except HTTPException:
        raise
    except Exception as exc:
        raise _public_error(request_id, exc) from exc
    finally:
        await file.close()
    return {"message": "success", "request_id": request_id, "result": payload}


@app.post("/api/video/upload/sequence", dependencies=AUTH)
async def analyze_upload_sequence(
    files: list[UploadFile] | None = File(default=None, description="순서대로 처리할 이미지 파일"),
    archive: UploadFile | None = File(default=None, description="이미지가 들어 있는 zip 파일"),
    sequence_id: str | None = Form(default=None),
    top_k: int = Form(3, ge=1, le=15),
):
    _ensure_ready()
    if archive is not None and files:
        raise HTTPException(status_code=400, detail="files와 archive 중 하나만 보내세요.")
    if archive is None and not files:
        raise HTTPException(status_code=400, detail="files 또는 archive가 필요합니다.")

    safe_id = _safe_sequence_id(sequence_id)
    request_id = uuid.uuid4().hex
    opened_uploads = [*(files or []), *([archive] if archive is not None else [])]
    try:
        with tempfile.TemporaryDirectory(prefix="sequence-", dir=WORK_DIR) as temp:
            temp_path = Path(temp)
            seq_dir = temp_path / "sequence"
            seq_dir.mkdir()
            items: list[tuple[str, bytes]] = []

            if archive is not None:
                if Path(_safe_name(archive.filename)).suffix.lower() != ".zip":
                    raise HTTPException(status_code=400, detail="archive는 zip 파일이어야 합니다.")
                raw = await _read_limited(archive, MAX_ARCHIVE_BYTES)
                try:
                    with zipfile.ZipFile(BytesIO(raw)) as zf:
                        members = [
                            info
                            for info in zf.infolist()
                            if not info.is_dir()
                            and Path(info.filename).suffix.lower() in core.IMAGE_EXTS
                            and "__MACOSX" not in info.filename
                            and not Path(info.filename).name.startswith(".")
                        ]
                        members.sort(key=lambda info: info.filename)
                        if not members:
                            raise HTTPException(status_code=400, detail="zip 안에 이미지가 없습니다.")
                        if len(members) > MAX_SEQUENCE_IMAGES:
                            raise HTTPException(status_code=413, detail="이미지 개수 제한을 초과했습니다.")
                        if sum(info.file_size for info in members) > MAX_SEQUENCE_BYTES:
                            raise HTTPException(status_code=413, detail="압축 해제 크기 제한을 초과했습니다.")
                        items = [(Path(info.filename).name, zf.read(info)) for info in members]
                except zipfile.BadZipFile as exc:
                    raise HTTPException(status_code=400, detail="유효하지 않은 zip 파일입니다.") from exc
            else:
                assert files is not None
                if len(files) > MAX_SEQUENCE_IMAGES:
                    raise HTTPException(status_code=413, detail="이미지 개수 제한을 초과했습니다.")
                total = 0
                for upload in files:
                    name = _safe_name(upload.filename)
                    data = await _read_limited(upload, MAX_SEQUENCE_BYTES)
                    total += len(data)
                    if total > MAX_SEQUENCE_BYTES:
                        raise HTTPException(status_code=413, detail="이미지 전체 크기 제한을 초과했습니다.")
                    items.append((name, data))

            for index, (name, data) in enumerate(items):
                _validate_image(data, name)
                suffix = Path(name).suffix.lower()
                (seq_dir / f"{index:04d}{suffix}").write_bytes(data)

            async with analysis_slots:
                payload = await run_in_threadpool(
                    _run_sequence,
                    seq_dir,
                    safe_id,
                    top_k,
                    temp_path / "output",
                )
    except HTTPException:
        raise
    except Exception as exc:
        raise _public_error(request_id, exc) from exc
    finally:
        for upload in opened_uploads:
            await upload.close()
    return {"message": "success", "request_id": request_id, "result": payload}


@app.post("/api/video/csv", dependencies=AUTH)
async def analyze_csv(
    file: UploadFile = File(..., description="label/link 또는 label/url 컬럼 CSV"),
    top_k: int = Form(3, ge=1, le=15),
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

    request_id = uuid.uuid4().hex

    def process_rows() -> list[dict]:
        results = []
        for index, row in enumerate(rows, start=2):
            candidate = row.get("link") or row.get("url") or ""
            try:
                safe_url = _validate_url(candidate)
                with tempfile.TemporaryDirectory(prefix="csv-", dir=WORK_DIR) as temp:
                    payload = _run_url(safe_url, top_k, Path(temp) / "output")
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
