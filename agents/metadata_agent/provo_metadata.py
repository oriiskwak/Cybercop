"""
provo_metadata.py
- ETRI 사이버범죄 메타데이터 체계 (Ver.0.8) 에 맞춘 PROV-O JSON-LD 빌더

핵심 개념:
  * @context : prov, rdfs, xsd, ex (사례 식별자) 접두사
  * @graph   : Entity / Activity / Agent 의 평탄한 배열
  * Entity   : file, file_meta, crime_meta, dv_meta, dc_meta, dd_meta, pp_meta, kg_meta, mg_meta, risk_meta
  * Activity : 추출 활동 (예: file_meta_extract, ner)
  * Agent    : 추출 모듈 (예: module_ner, module_file_meta)

NER 필드 규칙:
  * NER_text  : OCR/원문에 그대로 존재하는 문자열
  * NER_start : 0-기반 문자 오프셋
  * NER_end   : 배제적 끝 오프셋 (slice [start:end] = NER_text)

사용 예:
    from provo_metadata import build_provo_doc
    doc = build_provo_doc(
        file_id="진정서1.png",
        file_label="진정서1.png",
        file_meta={...},          # file_metadata.extract_file_metadata 결과
        crime_items={             # metadata_extractor 가 LLM 으로 채움
            "crime_type":   {"value": "사이버 사기", "ner": {...}},
            "crime_method": [{"value": "사이버투자사기", "ner": {...}}],
            "crime_step":   [{"value": "...", "ner": {...}}],
        },
        ocr_text="...",           # NER offset 검증에 사용 (선택)
    )
"""
from __future__ import annotations

import re
import uuid
from datetime import datetime, timezone
from typing import Any, Optional


# =====================================================
# JSON-LD @context
# =====================================================
PROV_CONTEXT: dict = {
    "prov":  "http://www.w3.org/ns/prov#",
    "rdfs":  "http://www.w3.org/2000/01/rdf-schema#",
    "xsd":   "http://www.w3.org/2001/XMLSchema#",
    "ex":    "https://etri.re.kr/cybercrime/",
    # 자주 쓰는 prov 술어를 짧게 노출 — 문서 가독성용
    "type":            "@type",
    "id":              "@id",
    "label":           "rdfs:label",
    "wasGeneratedBy":  {"@id": "prov:wasGeneratedBy", "@type": "@id"},
    "wasAttributedTo": {"@id": "prov:wasAttributedTo", "@type": "@id"},
    "wasDerivedFrom":  {"@id": "prov:wasDerivedFrom",  "@type": "@id"},
    "used":            {"@id": "prov:used",            "@type": "@id"},
    "wasAssociatedWith": {"@id": "prov:wasAssociatedWith", "@type": "@id"},
    "startedAtTime":   {"@id": "prov:startedAtTime",   "@type": "xsd:dateTime"},
    "endedAtTime":     {"@id": "prov:endedAtTime",     "@type": "xsd:dateTime"},
    "generatedAtTime": {"@id": "prov:generatedAtTime", "@type": "xsd:dateTime"},
    # 도메인 전용 술어
    "metaType":   "ex:metaType",
    "metaValue":  "ex:metaValue",
    "NER_text":   "ex:NER_text",
    "NER_start":  "ex:NER_start",
    "NER_end":    "ex:NER_end",
    "items":      "ex:items",
    "schemaVer":  "ex:schemaVer",
}


# =====================================================
# 통제 어휘 (controlled vocabulary)
# - ETRI 사이버범죄 메타데이터 체계 (Ver.0.8) 부속
# - 매핑 시 "정확한 한글 라벨" 을 사용
# =====================================================
CRIME_TYPES: list = [
    "사이버 사기",
    "사이버금융범죄",
    "개인•위치정보침해",
    "사이버 저작권 침해",
    "사이버스팸메일",
    "기타 정보통신망 이용형 범죄",
    "사이버성폭력",
    "사이버도박",
    "사이버 명예훼손·모욕",
    "사이버스토킹",
    "기타 불법 콘텐츠 범죄",
    "해킹",
    "서비스거부공격(DDoS등)",
    "악성프로그램",
    "기타 정보통신망 침해형 범죄",
]

CRIME_METHODS: list = [
    "계정도용",
    "단순침입",
    "자료유출",
    "자료훼손",
    "직거래사기",
    "쇼핑몰 사기",
    "게임 사기",
    "연애빙자사기",
    "사이버투자사기",
    "이메일 무역사기",
    "기타 사이버사기",
    "피싱(Phishing)",
    "파밍(Pharming)",
    "스미싱(Smishing)",
    "메모리해킹",
    "몸캠피싱",
    "메신저피싱",
    "기타 전기통신금융사기",
    "컴퓨터 등 사용사기",
    "전자화폐 등에 의한 거래 행위",
    "정보통신망 인증 관련 위반 행위",
    "스포츠토토",
    "경마,경륜,경정",
    "기타 인터넷 도박",
    "사이버 명예훼손·모욕",
    "사이버스토킹",
]

CRIME_STEPS: list = [
    "금융기관 가장한 이메일 발송",
    "이메일에서 안내",
    "인터넷주소 클릭",
    "가짜 은행사이트로 접속 유도",
    "보안카드번호 전부 입력 요구",
    "금융정보 탈취",
    "범행계좌로 이체",
    "악성코드에 감염",
    "피싱(가짜) 사이트로 유도",
    "무료쿠폰 제공 등의 문자메시지",
    "인터넷주소를 클릭",
    "악성코드가 스마트폰에 설치",
    "소액결제 피해 발생",
    "개인 정보 탈취",
    "사진 도용",
    "랜던 채팅 어플",
    "특정파일 설치 요구",
    "동영상 유포 협박",
]


# 어휘 set (빠른 매핑/검증)
_CRIME_TYPES_SET   = set(CRIME_TYPES)
_CRIME_METHODS_SET = set(CRIME_METHODS)
_CRIME_STEPS_SET   = set(CRIME_STEPS)

# 범죄수법(crime_method) → 범죄유형(crime_type) 대분류 매핑 (ETRI Ver.0.8 분류 체계).
# LLM 이 crime_type 칸에 수법(예: '사이버투자사기')을 잘못 넣었을 때 올바른 대분류를 추론하는 데 사용.
CRIME_METHOD_TO_TYPE: dict = {
    # 사이버 사기
    "직거래사기": "사이버 사기", "쇼핑몰 사기": "사이버 사기", "게임 사기": "사이버 사기",
    "연애빙자사기": "사이버 사기", "사이버투자사기": "사이버 사기", "이메일 무역사기": "사이버 사기",
    "기타 사이버사기": "사이버 사기",
    # 사이버금융범죄
    "피싱(Phishing)": "사이버금융범죄", "파밍(Pharming)": "사이버금융범죄", "스미싱(Smishing)": "사이버금융범죄",
    "메모리해킹": "사이버금융범죄", "메신저피싱": "사이버금융범죄", "기타 전기통신금융사기": "사이버금융범죄",
    "컴퓨터 등 사용사기": "사이버금융범죄", "전자화폐 등에 의한 거래 행위": "사이버금융범죄",
    "정보통신망 인증 관련 위반 행위": "사이버금융범죄",
    # 사이버성폭력
    "몸캠피싱": "사이버성폭력",
    # 정보통신망 침해형(해킹)
    "계정도용": "해킹", "단순침입": "해킹", "자료유출": "해킹", "자료훼손": "해킹",
    # 사이버도박
    "스포츠토토": "사이버도박", "경마,경륜,경정": "사이버도박", "기타 인터넷 도박": "사이버도박",
    # 불법 콘텐츠(수법명=유형명 동일)
    "사이버 명예훼손·모욕": "사이버 명예훼손·모욕",
    "사이버스토킹": "사이버스토킹",
}


def normalize_crime_items(crime_items: Optional[dict]) -> dict:
    """LLM 이 통제 어휘를 잘못 배치한 범죄분류를 교정한다.

    - crime_type 이 CRIME_TYPES 에 없고 CRIME_METHODS 에 해당하면(예: '사이버투자사기')
      → crime_method 로 옮기고, CRIME_METHOD_TO_TYPE 로 올바른 대분류를 채운다.
    - crime_type 이 비면 crime_method 들로부터 대분류를 추론한다.
    - 끝까지 유효한 대분류를 못 정하면 crime_type 은 None (잘못된 값을 남기지 않음).
    - crime_method 는 통제 어휘로 정규화, crime_step 은 예시 어휘라 원본 보존.

    Returns: {"crime_type": {value,ner}|None, "crime_method": [{value,ner}...], "crime_step": [...]}
    """
    ci = dict(crime_items or {})

    def _v(x):
        if isinstance(x, dict):
            return str(x.get("value") or "").strip()
        if isinstance(x, str):
            return x.strip()
        return ""

    def _ner(x):
        return x.get("ner") if isinstance(x, dict) else None

    def _as_list(raw):
        if raw is None:
            return []
        return raw if isinstance(raw, list) else [raw]

    # 1) crime_method 정규화(통제 어휘 매핑) + 중복 제거
    method_objs: list = []
    seen_m: set = set()
    for m in _as_list(ci.get("crime_method")):
        val = _normalize_label(_v(m), _CRIME_METHODS_SET)
        if val and val not in seen_m:
            seen_m.add(val)
            method_objs.append({"value": val, "ner": _ner(m)})

    # 2) crime_type 검증/교정
    ct_raw = ci.get("crime_type")
    ct_val = _v(ct_raw)
    ct_norm = _normalize_label(ct_val, _CRIME_TYPES_SET) if ct_val else ""
    type_obj = None
    if ct_norm in _CRIME_TYPES_SET:
        type_obj = {"value": ct_norm, "ner": _ner(ct_raw)}
    elif ct_val:
        # crime_type 칸에 들어온 값이 사실 '수법'이면 → 수법으로 이동.
        as_method = _normalize_label(ct_val, _CRIME_METHODS_SET)
        if as_method in _CRIME_METHODS_SET and as_method not in seen_m:
            seen_m.add(as_method)
            method_objs.insert(0, {"value": as_method, "ner": _ner(ct_raw)})

    # 3) crime_type 이 없으면 수법들로부터 대분류 추론
    if type_obj is None:
        for mo in method_objs:
            t = CRIME_METHOD_TO_TYPE.get(mo["value"])
            if t:
                type_obj = {"value": t, "ner": None}
                break

    # 4) crime_step — 예시 어휘이므로 정규화하되 원본 보존(드롭하지 않음)
    step_objs: list = []
    seen_s: set = set()
    for s in _as_list(ci.get("crime_step")):
        val = _normalize_label(_v(s), _CRIME_STEPS_SET)
        if val and val not in seen_s:
            seen_s.add(val)
            step_objs.append({"value": val, "ner": _ner(s)})

    return {"crime_type": type_obj, "crime_method": method_objs, "crime_step": step_objs}


# =====================================================
# 헬퍼
# =====================================================
def _now_iso() -> str:
    """xsd:dateTime — 로컬 TZ 포함 ISO8601."""
    return datetime.now().astimezone().replace(microsecond=0).isoformat()


def _file_key_from_id(file_id: str) -> str:
    """
    PROV @id 에 안전한 키로 변환.
    예: '진정서 1.png' → '%EC%A7%84%EC%A0%95%EC%84%9C_1.png'
    """
    if not file_id:
        return uuid.uuid4().hex
    # 공백 → _, 위험 문자 제거
    key = file_id.strip().replace(" ", "_")
    key = re.sub(r"[\\/?#\[\]@!$&'()*+,;=]", "", key)
    return key or uuid.uuid4().hex


def _validate_ner(ocr_text: str, ner: Optional[dict]) -> Optional[dict]:
    """
    NER 객체의 offset 을 ocr_text 와 대조해 검증/보정한다.
    - text 없으면 None 반환
    - start/end 가 어긋나면 ocr_text 에서 첫 occurrence 로 다시 찾는다.
    - 그래도 못 찾으면 offset 만 제거한 채 text 만 반환.
    """
    if not ner or not isinstance(ner, dict):
        return None
    text = (ner.get("NER_text") or ner.get("text") or "").strip()
    if not text:
        return None

    start = ner.get("NER_start", ner.get("start"))
    end   = ner.get("NER_end",   ner.get("end"))

    # 1) offset 그대로 검증
    if (
        ocr_text and isinstance(start, int) and isinstance(end, int)
        and 0 <= start < end <= len(ocr_text)
        and ocr_text[start:end] == text
    ):
        return {"NER_text": text, "NER_start": int(start), "NER_end": int(end)}

    # 2) ocr_text 에서 다시 검색
    if ocr_text:
        idx = ocr_text.find(text)
        if idx >= 0:
            return {"NER_text": text, "NER_start": idx, "NER_end": idx + len(text)}

    # 3) text 만 (offset 검증 실패)
    return {"NER_text": text}


def _normalize_label(value: str, vocab_set: set) -> str:
    """
    LLM 이 살짝 변형해서 출력한 값을 통제 어휘에 매핑.
    - 정확히 일치하면 그대로
    - 공백/괄호 차이만 다르면 정규화 후 매핑
    - 못 찾으면 원본 그대로 (단, 빈 문자열은 빈 문자열)
    """
    if not value:
        return ""
    v = value.strip()
    if v in vocab_set:
        return v
    # 괄호 안 영문 제거 (예: '피싱(Phishing)' ↔ '피싱')
    v_simple = re.sub(r"\s*\([^)]*\)\s*", "", v).strip()
    for cand in vocab_set:
        cand_simple = re.sub(r"\s*\([^)]*\)\s*", "", cand).strip()
        if v_simple == cand_simple:
            return cand
    # 부분 일치 (LLM 이 잘라먹은 경우)
    for cand in vocab_set:
        if v in cand or cand in v:
            return cand
    return v  # 원본 보존 — 검증 단계에서 표시


# =====================================================
# Entity / Activity / Agent 빌더
# =====================================================
def _agent(agent_id: str, label: str, kind: str = "prov:SoftwareAgent") -> dict:
    return {
        "id": f"ex:agent/{agent_id}",
        "type": ["prov:Agent", kind],
        "label": label,
    }


def _activity(act_id: str, label: str, started: str, ended: str,
              associated_with: Optional[str] = None,
              used: Optional[list] = None) -> dict:
    a: dict = {
        "id":   f"ex:act/{act_id}",
        "type": "prov:Activity",
        "label": label,
        "startedAtTime": started,
        "endedAtTime":   ended,
    }
    if associated_with:
        a["wasAssociatedWith"] = f"ex:agent/{associated_with}"
    if used:
        a["used"] = used
    return a


def _file_entity(file_key: str, label: str, file_meta: Optional[dict]) -> dict:
    """원본 파일 자체를 Entity 로."""
    e: dict = {
        "id":   f"ex:file/{file_key}",
        "type": ["prov:Entity", "ex:File"],
        "label": label,
    }
    if file_meta:
        # 자주 쓰는 식별자만 평탄하게 노출 — 전체는 file_meta Entity 로 따로
        fs = (file_meta.get("파일시스템") or {})
        if fs.get("절대경로"):
            e["ex:absolutePath"] = fs["절대경로"]
        if fs.get("크기바이트") is not None:
            e["ex:sizeBytes"] = int(fs["크기바이트"])
        h = (file_meta.get("해시") or {})
        if h.get("sha256"):
            e["ex:sha256"] = h["sha256"]
    return e


def _file_meta_entity(file_key: str, file_meta: dict, gen_act: str,
                      gen_agent: str, gen_time: str) -> dict:
    """파일 자체 메타데이터를 file_meta Entity 로."""
    return {
        "id":   f"ex:file_meta/{file_key}/1",
        "type": ["prov:Entity", "ex:file_meta"],
        "label": "파일 메타데이터",
        "metaType":  "file_meta",
        "metaValue": file_meta,
        "wasGeneratedBy":   f"ex:act/{gen_act}",
        "wasAttributedTo":  f"ex:agent/{gen_agent}",
        "generatedAtTime":  gen_time,
        "wasDerivedFrom":   f"ex:file/{file_key}",
    }


def _crime_meta_entities(file_key: str, crime_items: dict,
                         ocr_text: str, gen_act: str, gen_agent: str,
                         gen_time: str) -> list:
    """
    crime_meta Entity 들 — crime_type / crime_method / crime_step 별로 발행.

    crime_items 형식 (LLM 출력 후 정규화):
        {
          "crime_type":   {"value": "...", "ner": {...}}  | str | None,
          "crime_method": [{"value": "...", "ner": {...}}, ...] | str | [str],
          "crime_step":   [{"value": "...", "ner": {...}}, ...] | str | [str],
        }
    """
    entities: list = []
    seq = 0

    def _norm_one(raw: Any, vocab_set: set) -> Optional[dict]:
        """단일 항목 → {value, ner} 표준 형태."""
        if raw is None:
            return None
        if isinstance(raw, str):
            v = _normalize_label(raw, vocab_set)
            return {"value": v, "ner": None} if v else None
        if isinstance(raw, dict):
            v = _normalize_label(raw.get("value", ""), vocab_set)
            if not v:
                return None
            ner = _validate_ner(ocr_text, raw.get("ner"))
            return {"value": v, "ner": ner}
        return None

    def _norm_list(raw: Any, vocab_set: set) -> list:
        if raw is None:
            return []
        if isinstance(raw, (str, dict)):
            n = _norm_one(raw, vocab_set)
            return [n] if n else []
        if isinstance(raw, list):
            out = []
            seen = set()
            for r in raw:
                n = _norm_one(r, vocab_set)
                if n and n["value"] not in seen:
                    seen.add(n["value"])
                    out.append(n)
            return out
        return []

    bundle = {
        "crime_type":   _norm_list(crime_items.get("crime_type"),   _CRIME_TYPES_SET),
        "crime_method": _norm_list(crime_items.get("crime_method"), _CRIME_METHODS_SET),
        "crime_step":   _norm_list(crime_items.get("crime_step"),   _CRIME_STEPS_SET),
    }

    for kind, items in bundle.items():
        for item in items:
            seq += 1
            ent: dict = {
                "id":   f"ex:crime_meta/{file_key}/{seq}",
                "type": ["prov:Entity", "ex:crime_meta"],
                "label": f"{kind}: {item['value']}",
                "metaType":  kind,
                "metaValue": item["value"],
                "wasGeneratedBy":  f"ex:act/{gen_act}",
                "wasAttributedTo": f"ex:agent/{gen_agent}",
                "generatedAtTime": gen_time,
                "wasDerivedFrom":  f"ex:file/{file_key}",
            }
            if item["ner"]:
                ent["NER_text"]  = item["ner"]["NER_text"]
                if "NER_start" in item["ner"]:
                    ent["NER_start"] = item["ner"]["NER_start"]
                    ent["NER_end"]   = item["ner"]["NER_end"]
            entities.append(ent)

    return entities


# =====================================================
# 공개 API
# =====================================================
def _generic_meta_entity(
    file_key: str, kind: str, label: str, value: Any,
    seq: int, gen_act: str, gen_agent: str, gen_time: str,
    rdf_type: str = "ex:meta",
) -> dict:
    """file_meta / forgery_meta / video_meta 등 단일 도메인 Entity 빌더."""
    return {
        "id":   f"ex:{kind}/{file_key}/{seq}",
        "type": ["prov:Entity", rdf_type],
        "label": label,
        "metaType":  kind,
        "metaValue": value,
        "wasGeneratedBy":  f"ex:act/{gen_act}",
        "wasAttributedTo": f"ex:agent/{gen_agent}",
        "generatedAtTime": gen_time,
        "wasDerivedFrom":  f"ex:file/{file_key}",
    }


def build_provo_doc(
    file_id: str,
    file_label: str,
    file_meta: Optional[dict] = None,
    crime_items: Optional[dict] = None,
    forgery_meta: Optional[dict] = None,
    video_meta:   Optional[dict] = None,
    ocr_text: str = "",
    schema_version: str = "ETRI-Cyber-Meta-v0.8",
) -> dict:
    """
    PROV-O JSON-LD 문서 한 개를 만든다.

    Returns:
        {
          "@context": {...},
          "@graph":   [Entity, Activity, Agent, ...],
          "ex:schemaVer": "ETRI-Cyber-Meta-v0.8",
        }
    """
    file_key = _file_key_from_id(file_id)
    now = _now_iso()
    graph: list = []

    # --- Agents (필요한 것만) ---
    agents_added: set = set()
    def _ensure_agent(agent_id: str, label: str):
        if agent_id in agents_added:
            return
        graph.append(_agent(agent_id, label))
        agents_added.add(agent_id)

    # --- File entity (항상 발행) ---
    graph.append(_file_entity(file_key, file_label or file_id, file_meta))

    # --- file_meta Entity + Activity ---
    if file_meta:
        _ensure_agent("module_file_meta", "파일 메타데이터 추출 모듈")
        act_id = f"file_meta_extract/{file_key}"
        graph.append(_activity(
            act_id, "파일 메타데이터 추출", now, now,
            associated_with="module_file_meta",
            used=[f"ex:file/{file_key}"],
        ))
        graph.append(_file_meta_entity(
            file_key, file_meta,
            gen_act=act_id, gen_agent="module_file_meta", gen_time=now,
        ))

    # --- crime_meta Entities + Activity ---
    if crime_items and any(crime_items.get(k) for k in ("crime_type", "crime_method", "crime_step")):
        _ensure_agent("module_ner", "OCR/NER 기반 범죄 메타데이터 추출 모듈")
        act_id = f"ner/{file_key}"
        graph.append(_activity(
            act_id, "OCR 기반 범죄 NER", now, now,
            associated_with="module_ner",
            used=[f"ex:file/{file_key}"],
        ))
        graph.extend(_crime_meta_entities(
            file_key, crime_items, ocr_text,
            gen_act=act_id, gen_agent="module_ner", gen_time=now,
        ))

    # --- forgery_meta Entity + Activity (위변조 탐지) ---
    if forgery_meta:
        _ensure_agent("module_forgery", "이미지 위변조 탐지 모듈")
        act_id = f"forgery_detect/{file_key}"
        graph.append(_activity(
            act_id, "이미지 위변조 탐지", now, now,
            associated_with="module_forgery",
            used=[f"ex:file/{file_key}"],
        ))
        graph.append(_generic_meta_entity(
            file_key, kind="forgery_meta", label="위변조 판정",
            value=forgery_meta, seq=1,
            gen_act=act_id, gen_agent="module_forgery", gen_time=now,
            rdf_type="ex:forgery_meta",
        ))

    # --- video_meta Entity + Activity (영상 분석) ---
    if video_meta:
        _ensure_agent("module_video", "영상 분석 (OCR + 객체검출 + RAG) 모듈")
        act_id = f"video_analyze/{file_key}"
        graph.append(_activity(
            act_id, "영상 분석", now, now,
            associated_with="module_video",
            used=[f"ex:file/{file_key}"],
        ))
        graph.append(_generic_meta_entity(
            file_key, kind="video_meta", label="영상 분석 결과",
            value=video_meta, seq=1,
            gen_act=act_id, gen_agent="module_video", gen_time=now,
            rdf_type="ex:video_meta",
        ))

    return {
        "@context":     PROV_CONTEXT,
        "@graph":       graph,
        "ex:schemaVer": schema_version,
    }


# =====================================================
# attach_provo — save_body 를 introspection 해서 @provo 채워줌
# =====================================================
def attach_provo(
    save_body: dict,
    file_id: str,
    file_label: str = "",
    ocr_text: str = "",
) -> dict:
    """
    metadata_*.json 으로 저장될 dict 에 PROV-O JSON-LD 를 (재)생성해 추가한다.
    save_body 의 도메인 키를 자동 인식:
      - 파일메타데이터 → file_meta Entity
      - 범죄분류        → crime_meta Entities
      - 위변조판정      → forgery_meta Entity
      - 영상분석        → video_meta Entity (top-level keys: video_id/ocr/objects/rag/...)
    """
    if not isinstance(save_body, dict):
        return save_body

    file_meta   = save_body.get("파일메타데이터")
    crime_items = save_body.get("범죄분류")

    forgery_meta: Optional[dict] = None
    if save_body.get("위변조판정"):
        forgery_meta = {
            "위변조판정":  save_body.get("위변조판정"),
            "탐지모델":    save_body.get("탐지모델"),
            "산출물":      save_body.get("산출물"),
        }

    video_meta: Optional[dict] = None
    # video_agent 결과는 top-level 에 video_id / ocr / objects / rag / title 가 있음
    if any(k in save_body for k in ("video_id", "rag", "objects")):
        video_meta = {
            "video_id":    save_body.get("video_id"),
            "title":       save_body.get("title"),
            "objects":     save_body.get("objects"),
            "ocr":         save_body.get("ocr"),
            "rag":         save_body.get("rag"),
            "duration":    save_body.get("duration"),
        }
        # null 키 제거
        video_meta = {k: v for k, v in video_meta.items() if v is not None}
        if not video_meta:
            video_meta = None

    save_body["@provo"] = build_provo_doc(
        file_id      = file_id,
        file_label   = file_label or file_id,
        file_meta    = file_meta,
        crime_items  = crime_items if isinstance(crime_items, dict) else None,
        forgery_meta = forgery_meta,
        video_meta   = video_meta,
        ocr_text     = ocr_text,
    )
    return save_body


# =====================================================
# 자체 테스트
# =====================================================
if __name__ == "__main__":
    import json
    sample_ocr = "이 사건은 카카오뱅크 200-9999-999999 계좌로 송금하라는 보이스피싱이었다."
    doc = build_provo_doc(
        file_id="진정서1.png",
        file_label="진정서1.png",
        file_meta={
            "파일시스템": {"절대경로": "C:/cases/진정서1.png", "크기바이트": 12345},
            "해시":      {"sha256": "abc..."},
        },
        crime_items={
            "crime_type":   "사이버금융범죄",
            "crime_method": [{"value": "보이스피싱(전기통신금융사기)",
                              "ner": {"NER_text": "보이스피싱", "NER_start": 31, "NER_end": 36}}],
            "crime_step":   [{"value": "전화 통화로 송금 유도",
                              "ner": {"NER_text": "송금하라는", "NER_start": 22, "NER_end": 27}}],
        },
        ocr_text=sample_ocr,
    )
    print(json.dumps(doc, ensure_ascii=False, indent=2))
