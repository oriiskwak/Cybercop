"""
risk_api.py
- risk_utils의 피해금액 파싱 / 위험도 계산 기능을 REST API로 제공하는 서버.
- 실행:
    python risk_api.py
- 기본 주소: http://localhost:8003

엔드포인트:
    GET  /health         - 서버 상태 확인
    GET  /logs           - SSE 실시간 로그 스트리밍
    POST /parse_money    - 피해금액 문자열 → 원(KRW) 정수
    POST /compute_risk   - (피해자 수, 총 피해금액) → 위험도
    POST /summary        - metadata 리스트(JSON 본문) → 집계 + 위험도 (편의 엔드포인트)
    POST /risk_calcuation - metadata JSON 파일 복수 업로드 → 집계 + 위험도
"""

import asyncio
import datetime
import json
import logging
import os
import queue
import re
import sys
import tempfile
from pathlib import Path
from typing import Any, List, Optional

import uvicorn
from fastapi import FastAPI, File, Form, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel
from sse_starlette.sse import EventSourceResponse
from starlette.background import BackgroundTask

# =====================================================
# 설정
# =====================================================
HOST = "0.0.0.0"
PORT = 8003

# =====================================================
# risk_utils 임포트
# =====================================================
sys.path.insert(0, str(Path(__file__).resolve().parent))
from risk_utils import compute_risk, parse_money_to_won

logger = logging.getLogger("risk_api")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

# =====================================================
# SSE 로그 브로드캐스터
# =====================================================
_log_subscribers: List[queue.Queue] = []


class SSELogHandler(logging.Handler):
    """모든 로그 메시지를 SSE 구독자에게 전달."""
    def emit(self, record):
        msg = self.format(record)
        for q in _log_subscribers:
            try:
                q.put_nowait(msg)
            except queue.Full:
                pass


_sse_handler = SSELogHandler()
_sse_handler.setLevel(logging.INFO)
_sse_handler.setFormatter(logging.Formatter("%(asctime)s [%(name)s] %(message)s", datefmt="%H:%M:%S"))
logging.getLogger().addHandler(_sse_handler)

# =====================================================
# FastAPI 앱
# =====================================================
app = FastAPI(title="risk_agent API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# =====================================================
# Pydantic 모델
# =====================================================
class ParseMoneyRequest(BaseModel):
    text: str


class ComputeRiskRequest(BaseModel):
    victim_count: int
    total_loss_won: int


class SummaryRequest(BaseModel):
    """cyber_preprocess의 결과 리스트를 그대로 받아 집계한다."""
    results: List[dict]  # [{file, ok, metadata:{진정인, 진정의 요지}, ...}, ...]


# =====================================================
# 엔드포인트
# =====================================================
@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/logs")
async def logs():
    """SSE 엔드포인트: 실시간 로그 스트리밍."""
    q: queue.Queue = queue.Queue(maxsize=200)
    _log_subscribers.append(q)

    async def event_generator():
        try:
            while True:
                try:
                    msg = q.get_nowait()
                    yield {"event": "log", "data": msg}
                except queue.Empty:
                    await asyncio.sleep(0.1)
        except asyncio.CancelledError:
            pass
        finally:
            _log_subscribers.remove(q)

    return EventSourceResponse(event_generator())


@app.post("/parse_money")
def parse_money(body: ParseMoneyRequest):
    """
    피해금액 문자열을 원(KRW) 정수로 변환.

    Request:  {"text": "3억 2천만원"}
    Response: {"ok": true, "won": 320000000}  또는  {"ok": true, "won": null}
    """
    try:
        won = parse_money_to_won(body.text)
        return {"ok": True, "won": won}
    except Exception as e:
        logger.error(f"[parse_money] 실패: text={body.text!r} → {e}")
        return {"ok": False, "error": str(e), "won": None}


@app.post("/compute_risk")
def compute_risk_endpoint(body: ComputeRiskRequest):
    """
    피해자 수와 총 피해금액 기준으로 위험도 산출.

    Request:  {"victim_count": 3, "total_loss_won": 1200000}
    Response: {"ok": true, "risk": { "final_risk": "medium", ... }}
    """
    try:
        risk = compute_risk(victim_count=body.victim_count, total_loss_won=body.total_loss_won)
        logger.info(
            f"[compute_risk] 피해자 {body.victim_count}명, "
            f"총 {body.total_loss_won:,}원 → {risk['final_risk']}"
        )
        return {"ok": True, "risk": risk}
    except Exception as e:
        logger.error(f"[compute_risk] 실패: {e}")
        return {"ok": False, "error": str(e)}


def _extract_metadata(item: Any) -> Optional[dict]:
    """집계 대상 1건에서 실제 metadata dict 를 꺼낸다.

    허용 형태:
      - {"ok": true, "metadata": {...}}  (cyber_preprocess 결과 항목)
      - {"진정인": {...}, "진정의 요지": {...}, ...}  (per-file metadata 자체)
    ok 키가 있고 False 면 None(스킵). ok 가 없으면 처리 대상으로 간주.
    """
    if not isinstance(item, dict):
        return None
    if item.get("ok") is False:
        return None
    meta = item.get("metadata")
    if isinstance(meta, dict):
        return meta
    # metadata 키가 없으면 item 자체를 metadata 로 취급 (단일 metadata JSON 업로드 케이스)
    return item


def _extract_crime_type(meta: dict) -> str:
    """metadata 에서 범죄유형 문자열을 꺼낸다.

    우선순위:
      1) 레거시 한글 키 `범죄유형` (문자열)
      2) 신규 구조 `범죄분류.crime_type.value` (또는 문자열)
    없으면 "" 반환.
    """
    if not isinstance(meta, dict):
        return ""
    ct = meta.get("범죄유형")
    if isinstance(ct, str) and ct.strip():
        return ct.strip()
    block = meta.get("범죄분류")
    if isinstance(block, dict):
        c = block.get("crime_type")
        if isinstance(c, dict):
            v = c.get("value")
            if isinstance(v, str) and v.strip():
                return v.strip()
        elif isinstance(c, str) and c.strip():
            return c.strip()
    return ""


def _join_values(values: list[Any]) -> str:
    parts: list[str] = []
    for value in values:
        if value is None:
            continue
        if isinstance(value, list):
            text = ", ".join(str(v).strip() for v in value if str(v).strip())
        elif isinstance(value, dict):
            text = ", ".join(str(v).strip() for v in value.values() if str(v).strip())
        else:
            text = str(value).strip()
        if text:
            parts.append(text)
    return " ".join(parts)


def _build_statement_text(metas: list[dict]) -> str:
    """metadata에서 text-only 위험도 모델 입력용 진술 텍스트를 구성한다."""
    docs: list[str] = []
    for meta in metas:
        if not isinstance(meta, dict):
            continue
        summary = meta.get("진정의 요지") or {}
        site = meta.get("범죄관련사이트") or {}
        digital = meta.get("디지털식별자") or {}
        victim = meta.get("진정인") or {}
        suspect = meta.get("피진정인") or {}
        crime_block = meta.get("범죄분류") or {}
        crime_methods = crime_block.get("crime_method") if isinstance(crime_block, dict) else None
        crime_method_values = []
        if isinstance(crime_methods, list):
            for item in crime_methods:
                if isinstance(item, dict):
                    crime_method_values.append(item.get("value"))
                else:
                    crime_method_values.append(item)

        text = _join_values([
            f"관련 파일 {meta.get('관련파일이름')}" if meta.get("관련파일이름") else "",
            f"범죄유형 {meta.get('범죄유형')}" if meta.get("범죄유형") else "",
            f"범죄수법 {meta.get('범죄수법')}" if meta.get("범죄수법") else "",
            f"세부 수법 {_join_values(crime_method_values)}" if crime_method_values else "",
            f"사이트 {site.get('사이트명')} {site.get('URL')}" if isinstance(site, dict) else "",
            f"플랫폼 {digital.get('플랫폼')} ID {digital.get('ID')}" if isinstance(digital, dict) else "",
            f"용의자 ID {meta.get('용의자 ID')}" if meta.get("용의자 ID") else "",
            f"피해자 {victim.get('성명')}" if isinstance(victim, dict) and victim.get("성명") else "",
            f"피진정인 {suspect.get('성명')}" if isinstance(suspect, dict) and suspect.get("성명") else "",
            f"피해일시 {summary.get('피해일시')}" if isinstance(summary, dict) and summary.get("피해일시") else "",
            f"피해금액 {summary.get('피해금액')}" if isinstance(summary, dict) and summary.get("피해금액") else "",
            summary.get("기타") if isinstance(summary, dict) else "",
            f"범죄 관련 키워드 {_join_values(meta.get('범죄관련키워드') or [])}",
            f"광고 상품명 {meta.get('범죄광고상품명')}" if meta.get("범죄광고상품명") else "",
        ])
        if text:
            docs.append(text)
    return "\n".join(docs)


def _predict_text_risk(statement_text: str, total_loss_won: int = 0) -> dict:
    """패키지 내부 text-only 모델로 진술서 기반 위험도를 예측한다."""
    if not statement_text.strip():
        return {"available": False, "error": "metadata에서 모델 입력 텍스트를 구성할 수 없습니다.", "input_text": ""}
    try:
        model_dir = Path(__file__).resolve().parent.parent / "text_risk_model"
        if str(model_dir) not in sys.path:
            sys.path.insert(0, str(model_dir))
        from predictor import predict_text_risk
        return predict_text_risk(statement_text, total_loss_won)
    except Exception as e:
        return {"available": False, "error": str(e), "input_text": statement_text}


def _predict_guideline_multitask_risk(
    statement_text: str,
    victim_count: int,
    total_loss_won: int,
) -> dict:
    """S1/S2 원시값과 텍스트 기반 S3~S6 예측을 가이드라인으로 결합한다."""
    if not statement_text.strip():
        return {
            "available": False,
            "error": "metadata에서 S1~S6 결합 모델 입력 텍스트를 구성할 수 없습니다.",
            "input_text": "",
        }
    try:
        model_dir = Path(__file__).resolve().parent.parent / "guideline_multitask_model"
        if str(model_dir) not in sys.path:
            sys.path.insert(0, str(model_dir))
        from guideline_predictor import predict_guideline_multitask_risk
        return predict_guideline_multitask_risk(
            statement_text,
            victim_count=victim_count,
            total_loss_won=total_loss_won,
        )
    except Exception as e:
        return {"available": False, "error": str(e), "input_text": statement_text}


def _aggregate_results(results: list) -> dict:
    """metadata 항목 리스트에서 피해자 수 / 총 피해금액 / 범죄유형 / 위험도를 집계해 dict 로 반환.

    /summary 와 /risk_calcuation 이 공유하는 핵심 로직.
    """
    victim_names: set[str] = set()
    total_loss = 0
    loss_unknown_files: list[Any] = []
    crime_types: set[str] = set()
    # 파일별 산출 근거 — 어떤 파일에 어떤 피해자/피해금액이 있었는지(프론트 "산출 근거" 표시용).
    per_file: list[dict] = []
    metas_for_risk: list[dict] = []

    for item in results or []:
        meta = _extract_metadata(item)
        if meta is None:
            continue
        metas_for_risk.append(meta)
        file_name = item.get("file") if isinstance(item, dict) else None
        victim_name = (meta.get("진정인") or {}).get("성명")
        vname = str(victim_name).strip() if victim_name else ""
        if vname:
            victim_names.add(vname)
        crime_type = _extract_crime_type(meta)
        if crime_type:
            crime_types.add(crime_type)
        loss_text = (meta.get("진정의 요지") or {}).get("피해금액", "")
        won = parse_money_to_won(loss_text)
        per_file.append({
            "file": file_name,
            "victim_name": vname,
            "loss_text": loss_text or "",
            "loss_won": won,   # None 이면 피해금액 파싱 실패(미상)
        })
        if won is None:
            loss_unknown_files.append(file_name)
            continue
        total_loss += won

    victim_count = len(victim_names)
    statement_text = _build_statement_text(metas_for_risk)
    text_risk = _predict_text_risk(statement_text, total_loss)
    risk = _predict_guideline_multitask_risk(
        statement_text,
        victim_count=victim_count,
        total_loss_won=total_loss,
    )
    crime_type_list = sorted(crime_types)
    return {
        "victim_count": victim_count,
        "victim_names": sorted(victim_names),
        "total_loss_won": total_loss,
        "loss_unknown_files": loss_unknown_files,
        # 파일별 산출 근거 — [{file, victim_name, loss_text, loss_won}]
        "per_file": per_file,
        # 범죄유형 — 업로드된 metadata 들에서 추출한 고유 범죄유형 목록 + 단건 편의 필드
        "crime_types": crime_type_list,
        "crime_type": crime_type_list[0] if len(crime_type_list) == 1 else None,
        "risk": risk,
        "text_risk": text_risk,
    }


def _normalize_to_results(data: Any) -> list:
    """업로드된 JSON(다양한 형태)을 집계용 항목 리스트로 정규화.

      - {"results": [...]}        → 그 리스트
      - [ {...}, {...} ]           → 그대로 (metadata 또는 결과 항목들)
      - { 단일 metadata dict }     → [그 dict]
    """
    if isinstance(data, dict) and isinstance(data.get("results"), list):
        return data["results"]
    if isinstance(data, list):
        return data
    if isinstance(data, dict):
        return [data]
    return []


@app.post("/summary")
def summary(body: SummaryRequest):
    """
    cyber_preprocess 결과 리스트에서 피해자/피해금액/위험도를 한 번에 집계.

    Request:  {"results": [{"file": "...", "ok": true, "metadata": {...}}, ...]}
    Response: {
        "ok": true,
        "victim_count": int, "victim_names": [str],
        "total_loss_won": int, "loss_unknown_files": [str],
        "risk": {...}
    }
    """
    try:
        out = _aggregate_results(body.results)
        logger.info(
            f"[summary] 피해자 {out['victim_count']}명, "
            f"총 {out['total_loss_won']:,}원 → {out['risk'].get('final_risk', 'unavailable')}"
        )
        return {"ok": True, **out}
    except Exception as e:
        logger.error(f"[summary] 실패: {e}")
        return {"ok": False, "error": str(e)}


def _risk_filename(filenames: list) -> str:
    """위험도 결과 다운로드 파일명 — `RISK_<stems>_<ts>.json`.

    업로드한 metadata 파일들의 stem 을 이어붙여 식별성을 준다(길면 60자로 컷).
    """
    stems = [os.path.splitext(os.path.basename(n))[0] for n in filenames if n]
    combined = "_".join(stems).strip("_") or "result"
    if len(combined) > 60:
        combined = combined[:60]
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    return f"RISK_{combined}_{ts}.json"


def _safe_remove(path: str) -> None:
    try:
        os.remove(path)
    except OSError:
        pass


@app.post(
    "/risk_calcuation",
    responses={
        200: {
            "description": "위험도 결과 RISK_*.json 다운로드(response_format=file) 또는 JSON(response_format=json)",
            # format=binary 로 선언해야 Swagger UI 가 'Download file' 버튼을 렌더한다.
            "content": {
                "application/octet-stream": {"schema": {"type": "string", "format": "binary"}}
            },
        },
    },
)
async def risk_calcuation(
    files: List[UploadFile] = File(
        ...,
        description=(
            "metadata JSON 파일 (복수 선택 가능). 각 파일은 단일 metadata / metadata 리스트 / "
            "{\"results\":[...]} 형태 모두 허용. 업로드한 모든 파일의 metadata 를 합쳐서 집계한다."
        ),
    ),
    response_format: str = Form(
        "file",
        description=(
            '결과 반환 형식: "file"(기본)=RISK_<stems>_<ts>.json 파일 다운로드 | '
            '"json"=응답 본문에 결과 JSON 직접 반환'
        ),
    ),
):
    """
    **metadata JSON 파일 여러 개를 업로드**하면 모든 파일의 metadata 를 합쳐 피해자 수와
    총 피해금액을 추출해 위험도를 산출한다.

    `/summary` 와 동일한 집계 로직을 쓰되, 입력을 JSON 본문이 아니라 **파일 업로드(복수)** 로 받는다.
    Swagger 에서 파일 여러 개 선택만으로 바로 테스트 가능.

    각 파일에 허용되는 JSON 형태:
      - 단일 metadata 객체: `{"진정인": {"성명": "..."}, "진정의 요지": {"피해금액": "3억"}}`
      - metadata 객체 리스트: `[{...}, {...}]`
      - 결과 리스트: `{"results": [{"file": "...", "ok": true, "metadata": {...}}, ...]}`

    Response (response_format):
      • "file" (기본) — 결과를 `RISK_<stems>_<ts>.json` 파일로 다운로드.
      • "json"        — 아래 결과 JSON 을 본문에 직접 반환.
      입력 오류/집계 실패 시에는 형식과 무관하게 {"ok": false, "error": "..."} (JSON) 반환.

    결과 JSON: {
        "ok": true, "file_count": int, "source_filenames": [str], "item_count": int,
        "file_errors": [{"filename","error"}],
        "victim_count": int, "victim_names": [str],
        "total_loss_won": int, "loss_unknown_files": [...],
        "crime_types": [str], "crime_type": str|null,   # metadata 의 범죄유형(고유 목록 + 단건이면 그 값)
        "risk": {...}
    }
    """
    if not files:
        return {"ok": False, "error": "업로드된 파일이 없습니다."}

    all_results: list = []
    used_filenames: list[str] = []
    file_errors: list[dict] = []

    for f in files:
        if not f or not f.filename:
            continue
        try:
            raw = await f.read()
            if not raw:
                file_errors.append({"filename": f.filename, "error": "빈 파일"})
                continue
            data = json.loads(raw.decode("utf-8"))
        except json.JSONDecodeError as e:
            file_errors.append({"filename": f.filename, "error": f"JSON 파싱 실패: {e}"})
            continue
        except Exception as e:
            file_errors.append({"filename": f.filename, "error": f"읽기 실패: {e}"})
            continue
        items = _normalize_to_results(data)
        for item in items:
            if isinstance(item, dict) and "file" not in item:
                item["file"] = f.filename
        all_results.extend(items)
        used_filenames.append(f.filename)

    if not all_results:
        return {
            "ok": False,
            "error": "집계할 metadata 가 없습니다 (모든 파일이 비었거나 파싱 실패).",
            "file_errors": file_errors,
        }

    try:
        out = _aggregate_results(all_results)
        logger.info(
            f"[risk_calcuation] 파일 {len(used_filenames)}개, 항목 {len(all_results)}개 → "
            f"피해자 {out['victim_count']}명, 총 {out['total_loss_won']:,}원 → "
            f"{out['risk'].get('final_risk', 'unavailable')}"
        )
        result = {
            "ok": True,
            "file_count": len(used_filenames),
            "source_filenames": used_filenames,
            "item_count": len(all_results),
            "file_errors": file_errors,
            **out,
        }
    except Exception as e:
        logger.error(f"[risk_calcuation] 집계 실패 → {e}")
        return {"ok": False, "error": str(e), "file_errors": file_errors}

    # 기본은 RISK_*.json 파일 다운로드. response_format=json 이면 본문에 직접 반환.
    if (response_format or "file").strip().lower() == "json":
        return result

    download_name = _risk_filename(used_filenames)
    fd, out_path = tempfile.mkstemp(prefix="risk_", suffix=".json")
    with os.fdopen(fd, "w", encoding="utf-8") as fh:
        json.dump(result, fh, ensure_ascii=False, indent=2)
    # application/octet-stream + filename= 으로 Swagger 의 'Download file' 버튼이 RISK_*.json 으로 저장.
    return FileResponse(
        out_path,
        media_type="application/octet-stream",
        filename=download_name,
        background=BackgroundTask(_safe_remove, out_path),
    )


# =====================================================
# (추가) 메타그래프 노드 기반 사이버범죄 유형 평가
#   - 기존 엔드포인트는 그대로 두고, 새 기능만 추가.
#   - 메타그래프(entities/relations)의 노드 구성 + CrimeType/CrimeMethod/Keyword 노드 값을
#     규칙(키워드/별칭/노드타입 가중치)으로 채점해 사이버범죄 유형을 평가한다.
#   - 범죄유형/수법 라벨은 metadata_agent 의 통제 어휘(provo_metadata.CRIME_TYPES /
#     CRIME_METHODS, ETRI Ver.0.8)를 그대로 따른다 — metadata_agent 와 동일 기준.
# =====================================================
# metadata_agent 통제 어휘를 단일 출처로 import (범죄유형 대분류 / 수법 라벨).
_META_AGENT_DIR = Path(__file__).resolve().parent.parent / "metadata_agent"
if str(_META_AGENT_DIR) not in sys.path:
    sys.path.insert(0, str(_META_AGENT_DIR))
try:
    from provo_metadata import CRIME_TYPES, CRIME_METHODS  # 통제 어휘 (ETRI Ver.0.8)
except Exception as _e:  # metadata_agent 가 없어도 서버는 기동 (라벨 검증만 생략)
    CRIME_TYPES, CRIME_METHODS = [], []
    logger.warning(f"[classify_crime] provo_metadata 통제 어휘 import 실패 → 라벨 검증 생략: {_e}")

# 각 규칙은 metadata_agent 통제 어휘의 (crime_type 대분류, crime_method 수법) 한 쌍에 매핑된다.
#   - crime_type   : provo_metadata.CRIME_TYPES 중 하나 (대분류)
#   - crime_method : provo_metadata.CRIME_METHODS 중 하나 (수법)
#   - aliases      : 범죄유형/수법을 직접 가리키는 구어/별칭 (강한 신호)
#   - keywords     : 노드 값/힌트에서 찾을 키워드
#   - node_boost   : 노드 타입 구성 가중치 (구조적 단서)
RISK_CRIME_RULES = [
    {
        "crime_type": "사이버 사기",
        "crime_method": "직거래사기",
        "aliases": ["중고거래", "직거래사기", "중고나라사기", "직거래", "물품사기"],
        "keywords": ["중고", "직거래", "당근", "번개장터", "중고나라", "택배", "판매", "구매", "물품", "입금", "선입금"],
        "node_boost": {"Account": 2, "Product": 3, "App": 1, "Website": 1},
    },
    {
        "crime_type": "사이버 사기",
        "crime_method": "쇼핑몰 사기",
        "aliases": ["쇼핑몰사기", "온라인사기", "오픈마켓사기"],
        "keywords": ["쇼핑몰", "사이트", "결제", "주문", "스마트스토어", "오픈마켓", "구매대행", "사칭사이트"],
        "node_boost": {"Website": 3, "Account": 1, "Product": 2},
    },
    {
        "crime_type": "사이버금융범죄",
        "crime_method": "메신저피싱",
        "aliases": ["메신저피싱", "지인사칭", "카톡피싱"],
        "keywords": ["카카오톡", "카톡", "메신저", "지인", "가족", "사칭", "문화상품권", "기프트", "상품권"],
        "node_boost": {"App": 2, "Account": 1, "Person": 1},
    },
    {
        "crime_type": "사이버금융범죄",
        "crime_method": "기타 전기통신금융사기",
        "aliases": ["보이스피싱", "기관사칭", "전화금융사기"],
        "keywords": ["검찰", "경찰", "금융감독원", "수사", "명의도용", "안전계좌", "공공기관", "대포통장", "검사"],
        "node_boost": {"Phone": 3, "Account": 2},
    },
    {
        "crime_type": "사이버금융범죄",
        "crime_method": "사이버투자사기",
        "aliases": ["투자사기", "코인사기", "리딩방", "가상자산사기"],
        "keywords": ["투자", "코인", "비트코인", "리딩", "주식", "선물", "수익", "재테크", "가상자산", "거래소", "원금"],
        "node_boost": {"App": 2, "Website": 2, "Account": 1},
    },
    {
        "crime_type": "사이버 사기",
        "crime_method": "연애빙자사기",
        "aliases": ["로맨스스캠", "로맨스사기"],
        "keywords": ["연애", "로맨스", "외국", "군인", "선물", "통관", "환전", "달러", "사랑", "이성"],
        "node_boost": {"App": 2, "Person": 1, "Account": 1},
    },
    {
        "crime_type": "사이버성폭력",
        "crime_method": "몸캠피싱",
        "aliases": ["몸캠피싱", "영상협박", "섹스토션"],
        "keywords": ["몸캠", "영상통화", "협박", "유포", "지인목록", "음란", "나체"],
        "node_boost": {"App": 2, "Phone": 1},
    },
    {
        "crime_type": "사이버금융범죄",
        "crime_method": "기타 전기통신금융사기",
        "aliases": ["대출사기"],
        "keywords": ["대출", "저금리", "정부지원", "신용등급", "수수료", "보증금", "대환", "한도"],
        "node_boost": {"Phone": 2, "Account": 2, "App": 1},
    },
    {
        "crime_type": "사이버금융범죄",
        "crime_method": "스미싱(Smishing)",
        "aliases": ["스미싱", "피싱", "악성앱"],
        "keywords": ["문자", "링크", "택배조회", "청첩장", "부고", "앱설치", "원격", "악성", "url", "http"],
        "node_boost": {"Website": 3, "Phone": 1},
    },
]

# 규칙 라벨이 통제 어휘에 속하는지 검증 (metadata_agent 와 불일치 방지). 어휘 import 실패 시 생략.
if CRIME_TYPES and CRIME_METHODS:
    _bad_labels = [
        (r["crime_type"], r["crime_method"])
        for r in RISK_CRIME_RULES
        if r["crime_type"] not in CRIME_TYPES or r["crime_method"] not in CRIME_METHODS
    ]
    if _bad_labels:
        logger.warning(f"[classify_crime] 통제 어휘 밖 라벨 발견(확인 필요): {_bad_labels}")

# 범죄 신호가 강한 노드 타입(값을 가중 채점). 영문/한글(vt_*) 모두 포함.
_CRIME_SIGNAL_TYPES = {"CrimeType", "CrimeMethod", "Keyword", "App", "vt_app"}


def _mg_entities_relations(metagraph, entities, relations):
    """입력(metagraph dict 또는 entities/relations 직접)에서 노드/관계 리스트를 정규화."""
    mg = metagraph or {}
    ents = entities if entities is not None else (mg.get("entities") or mg.get("nodes") or [])
    rels = relations if relations is not None else (mg.get("relations") or mg.get("edges") or [])
    return (ents or []), (rels or [])


def _node_value(n):
    if not isinstance(n, dict):
        return ""
    return str(n.get("value") or n.get("name") or n.get("label") or "").strip()


def classify_crime_from_metagraph(entities, relations, hints=None):
    """메타그래프 노드를 규칙 기반으로 채점해 사이버범죄 유형/수법을 평가한다.

    각 규칙은 metadata_agent 통제 어휘의 (crime_type 대분류, crime_method 수법) 한 쌍이다.
    규칙별 점수를 매긴 뒤, crime_type 과 crime_method 별로 합산해 각각 순위를 산출한다.
    반환의 crime_types(대분류) / crime_methods(수법) 라벨은 metadata_agent 와 동일한 통제 어휘다.
    """
    ents = entities or []
    type_counts: dict = {}
    all_texts: list = []
    signal_texts: list = []   # 범죄 신호가 강한 노드 값(CrimeType/CrimeMethod/Keyword/App)

    for n in ents:
        if not isinstance(n, dict):
            continue
        t = str(n.get("type") or "").strip()
        type_counts[t] = type_counts.get(t, 0) + 1
        v = _node_value(n)
        if v:
            all_texts.append(v)
            if t in _CRIME_SIGNAL_TYPES:
                signal_texts.append(v)

    for h in (hints or []):
        if h:
            all_texts.append(str(h))
            signal_texts.append(str(h))

    blob = " ".join(all_texts).lower()
    sig = " ".join(signal_texts).lower()

    # --- 규칙별 채점 ---
    rule_scores: list = []
    for rule in RISK_CRIME_RULES:
        score = 0.0
        evidence = set()
        # 별칭 + 통제 어휘 라벨(crime_type/method) 직접 매칭 — 가장 강한 신호.
        # 통제 어휘 라벨을 포함시켜 metadata 의 범죄유형 힌트가 그대로 매칭되게 한다.
        for al in list(rule.get("aliases", [])) + [rule["crime_type"], rule["crime_method"]]:
            a = al.lower()
            if a in sig:
                score += 4.0
                evidence.add(al)
            elif a in blob:
                score += 2.0
                evidence.add(al)
        # 키워드 매칭
        for kw in rule["keywords"]:
            k = kw.lower()
            cb = blob.count(k)
            cs = sig.count(k)
            if cs:
                score += 1.5 * cs
                evidence.add(kw)
            if cb:
                score += 1.0 * cb
                evidence.add(kw)
        # 노드 타입 구성 가중치(구조적 단서)
        for nt, w in rule.get("node_boost", {}).items():
            cnt = type_counts.get(nt, 0)
            if cnt:
                score += w * min(cnt, 3) * 0.5
        if score > 0:
            rule_scores.append({
                "crime_type": rule["crime_type"],
                "crime_method": rule["crime_method"],
                "score": score,
                "evidence": evidence,
            })

    # --- crime_type(대분류) / crime_method(수법) 별 합산 + 신뢰도 ---
    def _aggregate(key_field: str, out_field: str) -> list:
        agg: dict = {}
        for rs in rule_scores:
            k = rs[key_field]
            slot = agg.get(k)
            if slot is None:
                slot = {out_field: k, "score": 0.0, "evidence": set()}
                if out_field == "method":
                    slot["crime_type"] = rs["crime_type"]   # 수법이 속한 대분류 표기
                agg[k] = slot
            slot["score"] += rs["score"]
            slot["evidence"].update(rs["evidence"])
        # 근거(evidence) 가 1개도 없는 항목은 분류에서 제외한다.
        #   - node_boost(구조적 노드 가중치)만으로 점수가 붙은 항목은 키워드/별칭 근거가 없으므로 표시하지 않음.
        items = [x for x in agg.values() if x["evidence"]]
        items.sort(key=lambda x: x["score"], reverse=True)
        # 신뢰도는 표시되는(근거 있는) 항목들 기준으로 재정규화.
        total = sum(x["score"] for x in items) or 1.0
        for x in items:
            x["confidence"] = round(x["score"] / total, 3)
            x["score"] = round(x["score"], 2)
            x["evidence"] = sorted(x["evidence"])
        return items

    crime_types = _aggregate("crime_type", "type")      # [{type, score, confidence, evidence}]
    crime_methods = _aggregate("crime_method", "method")  # [{method, crime_type, score, confidence, evidence}]

    return {
        "top_crime_type": crime_types[0]["type"] if crime_types else "미분류",
        "top_crime_method": crime_methods[0]["method"] if crime_methods else "",
        "crime_types": crime_types,
        "crime_methods": crime_methods,
        "node_type_counts": type_counts,
        "node_count": len(ents),
        "relation_count": len(relations or []),
    }


class CrimeTypeRequest(BaseModel):
    """메타그래프(entities/relations) 또는 metagraph dict 를 받아 범죄유형을 평가한다."""
    metagraph: Optional[dict] = None
    entities: Optional[list] = None
    relations: Optional[list] = None
    crime_type_hints: Optional[list] = None   # (선택) metadata 의 범죄유형 등 힌트


@app.post("/classify_crime")
def classify_crime(body: CrimeTypeRequest):
    """
    **메타그래프 노드 기반 사이버범죄 유형 평가** (신규 엔드포인트, 기존 API 와 독립).

    Request 예:
      {"metagraph": {"entities": [{"type":"Account","value":"110-..."}, ...],
                     "relations": [...]},
       "crime_type_hints": ["중고거래사기"]}
    또는 {"entities":[...], "relations":[...]}.

    범죄유형/수법 라벨은 metadata_agent 통제 어휘(CRIME_TYPES / CRIME_METHODS)를 따른다.

    Response: {
        "ok": true,
        "top_crime_type":   "사이버 사기",        # CRIME_TYPES 중 대분류
        "top_crime_method": "직거래 사기",         # CRIME_METHODS 중 수법
        "crime_types":   [{"type","score","confidence","evidence":[...]}, ...],            # 대분류별 합산
        "crime_methods": [{"method","crime_type","score","confidence","evidence":[...]}],  # 수법별 합산
        "node_type_counts": {"Account":3, "Product":1, ...},
        "node_count": int, "relation_count": int
    }
    """
    try:
        ents, rels = _mg_entities_relations(body.metagraph, body.entities, body.relations)
        out = classify_crime_from_metagraph(ents, rels, body.crime_type_hints)
        logger.info(
            f"[classify_crime] 노드 {out['node_count']}개 → "
            f"{out['top_crime_type']}"
            + (f" / {out['top_crime_method']}" if out.get('top_crime_method') else "")
        )
        return {"ok": True, **out}
    except Exception as e:
        logger.error(f"[classify_crime] 실패: {e}")
        return {"ok": False, "error": str(e)}


# =====================================================
# 서버 실행
# =====================================================
if __name__ == "__main__":
    print("=" * 50)
    print("  risk_agent API Server")
    print(f"  주소      : http://{HOST}:{PORT}")
    print("=" * 50)
    print("  엔드포인트:")
    print("    GET  /health         - 서버 상태 확인")
    print("    GET  /logs           - SSE 실시간 로그 스트리밍")
    print("    POST /parse_money    - 피해금액 문자열 → 원(KRW) 정수")
    print("    POST /compute_risk   - (피해자 수, 총 피해금액) → 위험도")
    print("    POST /summary        - metadata 리스트(JSON 본문) → 집계 + 위험도")
    print("    POST /risk_calcuation - metadata JSON 파일 복수 업로드 → 집계 + 위험도")
    print("    POST /classify_crime - 메타그래프 노드 기반 사이버범죄 유형 평가 (신규)")
    print("=" * 50)

    uvicorn.run(app, host=HOST, port=PORT)
