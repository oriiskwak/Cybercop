import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import sys
import types
import importlib.machinery
import re
import json
import time
import base64
import argparse
import difflib
import csv
from io import BytesIO
from pathlib import Path

# Windows 콘솔 인코딩 문제 해결 (유니코드 출력 지원)
if sys.platform == "win32":
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.detach(), encoding="utf-8", line_buffering=True)
    sys.stderr = io.TextIOWrapper(sys.stderr.detach(), encoding="utf-8", line_buffering=True)

# OCR stub (librosa / soundfile 미사용이지만 하위 호환 유지)
if "librosa" not in sys.modules:
    fake_librosa = types.ModuleType("librosa")
    fake_librosa.__version__ = "0.0"
    fake_librosa.__spec__ = importlib.machinery.ModuleSpec("librosa", loader=None)
    sys.modules["librosa"] = fake_librosa

if "soundfile" not in sys.modules:
    fake_soundfile = types.ModuleType("soundfile")
    fake_soundfile.__version__ = "0.0"
    fake_soundfile.__spec__ = importlib.machinery.ModuleSpec("soundfile", loader=None)
    sys.modules["soundfile"] = fake_soundfile

import cv2
import yt_dlp
import numpy as np
import faiss
import torch
from PIL import Image
from transformers import AutoModelForCausalLM, AutoProcessor
from huggingface_hub import snapshot_download
from sentence_transformers import SentenceTransformer

# ──────────────────────────────────────────────
# 설정
# ──────────────────────────────────────────────

MODEL_ID            = os.getenv("VLM_MODEL", "skt/A.X-4.0-VL-Light")
DEVICE              = "cuda" if torch.cuda.is_available() else "cpu"
TORCH_DTYPE         = torch.bfloat16 if torch.cuda.is_available() else torch.float32
DEFAULT_EMBED_MODEL = "BAAI/bge-m3"
DEFAULT_DOCS_PATH   = "./rag/retrieval_docs.json"

# Objects365 한국어 허용 클래스명 리스트 (외부 JSON에서 로드)
_ALLOWED_OBJECTS_PATH = Path(__file__).parent / "rag" / "allowed_objects.json"
ALLOWED_OBJECTS: list = json.load(open(_ALLOWED_OBJECTS_PATH, encoding="utf-8"))

# [보완] 범죄유형별 위험도 수치 맵 (이미지 데이터 기반)
CRIME_RISK_MAP = {
    "직거래 사기": {"months": 12.0, "risk": 0.22},
    "쇼핑몰 사기": {"months": 12.0, "risk": 0.22},
    "게임 사기": {"months": 12.0, "risk": 0.22},
    "이메일 무역사기": {"months": 30.0, "risk": 0.55},
    "기타 사이버 사기": {"months": 12.0, "risk": 0.22},
    "피싱": {"months": 12.0, "risk": 0.22},
    "파밍": {"months": 12.0, "risk": 0.22},
    "스미싱": {"months": 12.0, "risk": 0.22},
    "메모리 해킹": {"months": 18.0, "risk": 0.33},
    "몸캠피싱": {"months": 39.0, "risk": 0.71},
    "메신저 이용사기": {"months": 12.0, "risk": 0.22},
    "기타 사이버 금융범죄": {"months": 10.0, "risk": 0.18},
    "개인·위치정보 침해": {"months": 12.3, "risk": 0.22},
    "사이버저작권 침해": {"months": 13.0, "risk": 0.24},
    "기타 정보통신망 이용범죄": {"months": 7.0, "risk": 0.13},
    "아동성 착취물": {"months": 55.0, "risk": 1.00},
    "불법 촬영물": {"months": 24.3, "risk": 0.44},
    "허위 영상물": {"months": 15.0, "risk": 0.27},
    "불법성 영상물": {"months": 7.0, "risk": 0.13},
    "스포츠 토토": {"months": 16.0, "risk": 0.29},
    "경마·경륜·경정": {"months": 13.0, "risk": 0.24},
    "카지노": {"months": 13.0, "risk": 0.24},
    "기타 사이버 도박": {"months": 13.0, "risk": 0.24},
    "명예훼손": {"months": 11.0, "risk": 0.20},
    "모욕": {"months": 5.0, "risk": 0.09},
    "기타 불법콘텐츠 범죄": {"months": 8.0, "risk": 0.15}
}

ABNORMAL_THRESHOLD = 0.5  # RAG 최고 유사도가 이 이상이면 abnormal 판정

# ──────────────────────────────────────────────

# 프롬프트
# ──────────────────────────────────────────────
DEFAULT_PROMPT = """
이 이미지는 동영상의 한 프레임이다. 다음 두 가지 작업을 수행하라.

1. [OCR]: 화면에 보이는 모든 한국어/영어 텍스트를 추출하라.
    - 규칙: 원문 그대로 적고, 줄바꿈을 유지하며, 읽기 어려운 부분은 [불명확]으로 표시하라. 텍스트가 없으면 "텍스트 없음"이라 한다.

2. [OBJECTS]: 이미지 내에 보이는 사물들을 한국어로 자유롭게 나열하라.
    - 규칙:
      - 콤마(,)로 구분하여 최대 10개만 나열한다.
      - 설명문 없이 사물명만 적는다.
      - 같은 사물을 중복해서 쓰지 않는다.
      - 확신이 낮은 객체는 제외한다.
      - 적절한 객체가 없으면 "없음"이라 한다.
      - 예시: 사람, 의자, 자동차
      - 잘못된 예시: Person, chair, tv, table_tennis

두 섹션을 반드시 [OCR]과 [OBJECTS] 제목으로 구분하라.
""".strip()



# ──────────────────────────────────────────────
# 유틸
# ──────────────────────────────────────────────
def ensure_dir(path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path

def format_ts(seconds: float) -> str:
    seconds = max(0, int(seconds))
    h, m, s = seconds // 3600, (seconds % 3600) // 60, seconds % 60
    return f"{h:02d}:{m:02d}:{s:02d}" if h > 0 else f"{m:02d}:{s:02d}"


# ──────────────────────────────────────────────
# 모델 로드
# ──────────────────────────────────────────────
def load_model(model_id: str):
    print(f"  [INFO] 모델 로드 중: {model_id}  (device={DEVICE}, dtype={TORCH_DTYPE})")
    
    if os.path.isdir(model_id):
        local_dir = Path(model_id)
    else:
        # 모델 이름에 점(.)이 포함되면 transformers 모듈 경로가 꼬이므로 로컬에 먼저 다운로드
        safe_name = model_id.replace("/", "_").replace(".", "_")
        local_dir = Path("./hf_models") / safe_name
        if not local_dir.exists():
            print(f"  [INFO] 로컬 캐시 없음 — HuggingFace에서 다운로드 중...")
            snapshot_download(repo_id=model_id, local_dir=str(local_dir))
            
    print(f"  [INFO] 로컬 경로에서 로드: {local_dir}")
    processor = AutoProcessor.from_pretrained(
        str(local_dir),
        trust_remote_code=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        str(local_dir),
        torch_dtype=TORCH_DTYPE,
        device_map="auto",
        trust_remote_code=True,
    )
    model.eval()
    print("  [OK] 모델 로드 완료.")
    return processor, model


# ──────────────────────────────────────────────
# RAG
# ──────────────────────────────────────────────
class CrimeRAG:
    def __init__(self, docs_path: str, model_name: str = DEFAULT_EMBED_MODEL):
        self.docs_path = Path(docs_path)
        with open(self.docs_path, "r", encoding="utf-8") as f:
            self.docs = json.load(f)
        self.model = SentenceTransformer(model_name)
        self.index = None

    def build_index(self):
        texts = [doc["text"] for doc in self.docs]
        embeddings = self.model.encode(
            texts, convert_to_numpy=True, normalize_embeddings=True
        ).astype("float32")
        dim = embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dim)
        self.index.add(embeddings)

    def search(self, query: str, top_k: int = 3):
        if not self.index:
            self.build_index()
        q_emb = self.model.encode(
            [query], convert_to_numpy=True, normalize_embeddings=True
        ).astype("float32")
        scores, indices = self.index.search(q_emb, top_k)
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0 or idx >= len(self.docs):
                continue
            doc = self.docs[idx].copy()
            doc["similarity"] = float(score)
            
            # 위험도 매핑 로직 추가
            crime_type = doc.get("crime_type", "기타")
            risk_data = CRIME_RISK_MAP.get(crime_type, {"months": 0.0, "risk": 0.0})
            doc["risk_level"] = risk_data["risk"]
            
            results.append(doc)
        return results


# ──────────────────────────────────────────────
# 객체 매퍼 (임베딩 기반 ALLOWED_OBJECTS 매핑)
# ──────────────────────────────────────────────
class ObjectMapper:
    def __init__(self, embed_model: SentenceTransformer, threshold: float = 0.6):
        self.model = embed_model
        self.threshold = threshold
        self.allowed = ALLOWED_OBJECTS
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
# 영상 처리
# ──────────────────────────────────────────────
def download_video(url: str, out_dir: Path) -> tuple[Path, dict]:
    outtmpl = str(out_dir / "%(title).80s_%(id)s.%(ext)s")
    ydl_opts = {
        "outtmpl": outtmpl,
        "format": "best[ext=mp4]/best",
        "noplaylist": True,
        "quiet": True,
        "no_warnings": True,
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=True)
    video_path = Path(ydl.prepare_filename(info))
    if not video_path.exists():
        video_path = list(out_dir.glob(f"*{info['id']}*"))[0]
    return video_path, info

def sample_frames(video_path: Path, every_n: float, max_f: int) -> list[tuple[float, Image.Image]]:
    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    step = max(int(round(fps * every_n)), 1)
    frames, idx = [], 0
    while True:
        ok, frame_bgr = cap.read()
        if not ok:
            break
        if idx % step == 0:
            image = Image.fromarray(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
            frames.append((idx / fps, image))
            if max_f > 0 and len(frames) >= max_f:
                break
        idx += 1
    cap.release()
    return frames


# ──────────────────────────────────────────────
# 프레임 분석 (transformers)
# ──────────────────────────────────────────────
def analyze_one_frame(
    processor: AutoProcessor,
    model: AutoModelForCausalLM,
    image: Image.Image,
    prompt: str,
    obj_mapper: "ObjectMapper",
    max_new_tokens: int = 1024,
) -> dict:
    # PIL 이미지를 messages 안에 직접 포함해야 apply_chat_template이 이미지 토큰을 삽입함
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image.convert("RGB")},
                {"type": "text", "text": prompt},
            ],
        }
    ]

    text_input = processor.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=False,
    )

    inputs = processor(
        text=[text_input],
        images=[image.convert("RGB")],
        return_tensors="pt",
    ).to(DEVICE)

    # 생성
    start_time = time.time()
    with torch.inference_mode():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=1.0,
            top_p=0.95,
        )

    # 입력 토큰 제거 후 디코딩
    input_len   = inputs["input_ids"].shape[1]
    new_ids     = output_ids[:, input_len:]
    text_resp   = processor.batch_decode(new_ids, skip_special_tokens=True)[0].strip()

    elapsed     = time.time() - start_time
    # 디버깅용 원문 출력
    print(f"\n--- RAW RESPONSE ({elapsed:.2f}s) ---\n{text_resp}\n---------------------")

    ocr_text, obj_list = "텍스트 없음", []

    # OCR 추출
    ocr_match = re.search(
        r"\[?OCR\]?[:\s\n]+(.*?)(?=\[?OBJECTS\]?|$)", text_resp, re.S | re.I
    )
    if ocr_match:
        ocr_text = ocr_match.group(1).strip()

    # 객체 추출 및 임베딩 기반 매핑
    obj_match = re.search(r"\[?OBJECTS\]?[:\s\n]+(.*?)$", text_resp, re.S | re.I)
    if obj_match:
        raw = obj_match.group(1).strip()
        if raw and "없음" not in raw:
            candidates = [o.strip() for o in re.split(r"[,|\n·\-\.]", raw) if o.strip()]
            obj_list = obj_mapper.map(candidates)
            print(f"    [Post-Process] {candidates} → {obj_list}")

    return {"ocr_text": ocr_text, "object_list": obj_list, "inference_time": elapsed}


# ──────────────────────────────────────────────
# OCR VLM 후처리
# ──────────────────────────────────────────────
def postprocess_ocr_with_vlm(processor, model, merged_ocr: list) -> list:
    if not merged_ocr:
        return merged_ocr
    input_json = json.dumps(merged_ocr, ensure_ascii=False, indent=2)
    prompt = (
        "다음은 동영상에서 시간대별로 추출된 원시 OCR 텍스트의 JSON 배열이다.\n"
        "1. 오타와 할루시네이션(무의미한 반복)을 교정하라.\n"
        "2. 전체 전후 문맥을 파악하여 끊어진 문장을 자연스럽게 하나로 연결하라.\n"
        "3. 같은 내용이 이어지는 경우 하나로 병합하고 start/end 업데이트.\n"
        "4. @아이디, URL, 해시태그, 전화번호, 계좌번호 등 식별자는 원문 그대로 보존하라.\n"
        "5. 오직 교정된 JSON 배열만 출력하라.\n\n"
        f"[원본 타임라인 데이터]\n{input_json}"
    )
    messages = [{"role": "user", "content": [{"type": "text", "text": prompt}]}]
    text_input = processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
    inputs = processor(text=[text_input], return_tensors="pt").to(DEVICE)
    with torch.inference_mode():
        output_ids = model.generate(**inputs, max_new_tokens=2048, do_sample=False)
    corrected = processor.batch_decode(output_ids[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True)[0].strip()
    try:
        m = re.search(r'\[\s*\{.*?\}\s*\]', corrected, re.DOTALL)
        return json.loads(m.group(0) if m else corrected)
    except Exception:
        return merged_ocr  # fallback


# ──────────────────────────────────────────────
# 단일 영상 처리 파이프라인
# ──────────────────────────────────────────────
def process_single_video(
    url: str,
    label: str,
    processor: AutoProcessor,
    model: AutoModelForCausalLM,
    rag: CrimeRAG,
    obj_mapper: ObjectMapper,
    args,
):
    video_id_match = re.search(r"(?:v=|video/|shorts/)([a-zA-Z0-9_-]+)", url)
    video_id = video_id_match.group(1) if video_id_match else str(int(time.time()))

    out_dir  = Path(args.out_dir)
    res_dir  = ensure_dir(out_dir / "results")
    json_path = res_dir / f"ocr_results_{video_id}.json"
    txt_path  = res_dir / f"ocr_merged_{video_id}.txt"

    if json_path.exists():
        print(f"  [Skip] Already completed: {video_id}")
        return

    print(f"  [Processing] {url} (Label: {label})")
    video_start = time.time()
    try:
        v_path, info = download_video(url, out_dir)
        frames = sample_frames(v_path, args.sample_sec, args.max_frames)

        raw_results, obj_counts = [], {}
        for i, (sec, img) in enumerate(frames, 1):
            res = analyze_one_frame(processor, model, img, DEFAULT_PROMPT, obj_mapper)
            print(f"    - Frame {i}/{len(frames)} ({format_ts(sec)}) [{res['inference_time']:.2f}s]")
            raw_results.append({
                "sec": round(sec, 3),
                "ts": format_ts(sec),
                "text": res["ocr_text"],
                "objects": res["object_list"],
            })
            for obj in res["object_list"]:
                obj_counts[obj] = obj_counts.get(obj, 0) + 1

        # 프레임별 OCR + 병합 (유사도 0.5 미만이면 새 구간, 이상이면 구간 확장 + 고유 줄 누적)
        ocr_frames, merged_ocr = [], []
        for r in raw_results:
            if not r["text"] or r["text"] in ["텍스트 없음", "[불명확]"]:
                continue
            ocr_frames.append({"ts": r["ts"], "text": r["text"]})
            new_lines = [l.strip() for l in r["text"].split("\n") if l.strip()]
            if not merged_ocr or difflib.SequenceMatcher(None, merged_ocr[-1]["_raw"], r["text"]).ratio() < 0.5:
                merged_ocr.append({"start": r["ts"], "end": r["ts"], "_raw": r["text"], "_lines": new_lines})
            else:
                merged_ocr[-1]["end"] = r["ts"]
                merged_ocr[-1]["_raw"] = r["text"]
                existing = set(merged_ocr[-1]["_lines"])
                for line in new_lines:
                    if line not in existing:
                        merged_ocr[-1]["_lines"].append(line)
                        existing.add(line)

        for m in merged_ocr:
            m["text"] = "\n".join(m.pop("_lines"))
            m.pop("_raw")

        # VLM 후처리: 오타/할루시네이션 교정 및 중복 병합
        merged_ocr = postprocess_ocr_with_vlm(processor, model, merged_ocr)

        # Voting: 2프레임 이상 등장한 객체만 유지 (프레임이 1개인 경우는 예외)
        if len(frames) > 1:
            final_objs = [obj for obj, count in obj_counts.items() if count >= 2]
        else:
            final_objs = list(obj_counts.keys())

        # RAG 검색 + 레이블 자동 예측
        ocr_all = " | ".join(m["text"] for m in merged_ocr)
        obj_all = ", ".join(final_objs)
        rag_candidates = rag.search(f"[OCR]: {ocr_all} | [Objects]: {obj_all}", top_k=args.top_k)
        max_sim = max((r["similarity"] for r in rag_candidates), default=0.0)
        if max_sim >= ABNORMAL_THRESHOLD:
            predicted_label = "abnormal"
            mapped = rag_candidates
        else:
            predicted_label = "normal"
            mapped = []

        # 결과 저장
        video_elapsed = time.time() - video_start
        payload = {
            "id": video_id,
            "url": url,
            "title": info.get("title"),
            "gt_label": label,
            "label": predicted_label,
            "objects": final_objs,
            "ocr_frames": ocr_frames,
            "ocr_merged": merged_ocr,
            "rag": mapped,
            "total_inference_time": round(video_elapsed, 2)
        }
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        with open(txt_path, "w", encoding="utf-8") as f:
            f.write(f"VIDEO: {info.get('title')}\nURL: {url}\nGT_LABEL: {label}\nPREDICTED: {predicted_label}\n\n[Mapped Crimes]\n")
            for c in mapped:
                f.write(f"- {c['crime_type']} ({c['similarity']:.4f})\n")
            f.write(f"\n[Detected Objects]\n{obj_all}\n\n[OCR Timeline]\n")
            for m in merged_ocr:
                f.write(f"[{m['start']}~{m['end']}] {m['text']}\n")
        print(f"  [RAG] 매칭된 범죄 유형 (예측: {predicted_label}):")
        for c in mapped:
            print(f"    - {c['crime_type']} (score: {c['similarity']:.4f})")
        video_elapsed = time.time() - video_start
        print(f"  [Done] Result saved: {json_path} (Total: {video_elapsed:.2f}s)")

    except Exception as e:
        print(f"  [Error] {url}: {e}")


VIDEO_EXTS = {".mp4", ".avi", ".mkv", ".mov", ".webm", ".flv", ".ts"}
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".webp", ".tiff"}


# ──────────────────────────────────────────────
# 로컬 파일 처리 (영상 or 이미지)
# ──────────────────────────────────────────────
def process_local_file(
    file_path: Path,
    label: str,
    processor: AutoProcessor,
    model: AutoModelForCausalLM,
    rag: CrimeRAG,
    obj_mapper: ObjectMapper,
    args,
):
    file_id = file_path.stem
    ext = file_path.suffix.lower()
    res_dir = ensure_dir(Path(args.out_dir) / "results")
    json_path = res_dir / f"ocr_results_{file_id}.json"
    txt_path  = res_dir / f"ocr_merged_{file_id}.txt"

    if json_path.exists():
        print(f"  [Skip] Already completed: {file_id}")
        return

    print(f"  [Processing] {file_path.name} (Label: {label})")
    file_start = time.time()
    try:
        if ext in IMAGE_EXTS:
            image = Image.open(file_path).convert("RGB")
            frames = [(0.0, image)]
        else:
            frames = sample_frames(file_path, args.sample_sec, args.max_frames)

        raw_results, obj_counts = [], {}
        for i, (sec, img) in enumerate(frames, 1):
            res = analyze_one_frame(processor, model, img, DEFAULT_PROMPT, obj_mapper)
            print(f"    - Frame {i}/{len(frames)} ({format_ts(sec)}) [{res['inference_time']:.2f}s]")
            raw_results.append({
                "sec": round(sec, 3),
                "ts": format_ts(sec),
                "text": res["ocr_text"],
                "objects": res["object_list"],
            })
            for obj in res["object_list"]:
                obj_counts[obj] = obj_counts.get(obj, 0) + 1

        # 프레임별 OCR + 병합 (유사도 0.5 미만이면 새 구간, 이상이면 구간 확장 + 고유 줄 누적)
        ocr_frames, merged_ocr = [], []
        for r in raw_results:
            if not r["text"] or r["text"] in ["텍스트 없음", "[불명확]"]:
                continue
            ocr_frames.append({"ts": r["ts"], "text": r["text"]})
            new_lines = [l.strip() for l in r["text"].split("\n") if l.strip()]
            if not merged_ocr or difflib.SequenceMatcher(None, merged_ocr[-1]["_raw"], r["text"]).ratio() < 0.5:
                merged_ocr.append({"start": r["ts"], "end": r["ts"], "_raw": r["text"], "_lines": new_lines})
            else:
                merged_ocr[-1]["end"] = r["ts"]
                merged_ocr[-1]["_raw"] = r["text"]
                existing = set(merged_ocr[-1]["_lines"])
                for line in new_lines:
                    if line not in existing:
                        merged_ocr[-1]["_lines"].append(line)
                        existing.add(line)

        for m in merged_ocr:
            m["text"] = "\n".join(m.pop("_lines"))
            m.pop("_raw")

        # VLM 후처리: 오타/할루시네이션 교정 및 중복 병합
        merged_ocr = postprocess_ocr_with_vlm(processor, model, merged_ocr)

        # Voting: 2프레임 이상 등장한 객체만 유지 (프레임이 1개인 경우는 예외)
        if len(frames) > 1:
            final_objs = [obj for obj, count in obj_counts.items() if count >= 2]
        else:
            final_objs = list(obj_counts.keys())

        # RAG 검색 + 레이블 자동 예측
        ocr_all = " | ".join(m["text"] for m in merged_ocr)
        obj_all = ", ".join(final_objs)
        rag_candidates = rag.search(f"[OCR]: {ocr_all} | [Objects]: {obj_all}", top_k=args.top_k)
        max_sim = max((r["similarity"] for r in rag_candidates), default=0.0)
        if max_sim >= ABNORMAL_THRESHOLD:
            predicted_label = "abnormal"
            mapped = rag_candidates
        else:
            predicted_label = "normal"
            mapped = []

        payload = {
            "id": file_id,
            "url": str(file_path),
            "title": file_path.name,
            "gt_label": label,
            "label": predicted_label,
            "objects": final_objs,
            "ocr_frames": ocr_frames,
            "ocr_merged": merged_ocr,
            "rag": mapped,
        }
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        with open(txt_path, "w", encoding="utf-8") as f:
            f.write(f"FILE: {file_path.name}\nGT_LABEL: {label}\nPREDICTED: {predicted_label}\n\n[Mapped Crimes]\n")
            for c in mapped:
                f.write(f"- {c['crime_type']} ({c['similarity']:.4f})\n")
            f.write(f"\n[Detected Objects]\n{obj_all}\n\n[OCR Timeline]\n")
            for m in merged_ocr:
                f.write(f"[{m['start']}~{m['end']}] {m['text']}\n")
        print(f"  [RAG] 매칭된 범죄 유형 (예측: {predicted_label}):")
        for c in mapped:
            print(f"    - {c['crime_type']} (score: {c['similarity']:.4f})")
        file_elapsed = time.time() - file_start
        print(f"  [Done] Result saved: {json_path} (Total: {file_elapsed:.2f}s)")

    except Exception as e:
        print(f"  [Error] {file_path.name}: {e}")


# ──────────────────────────────────────────────
# 메인
# ──────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="Video OCR + RAG pipeline (HuggingFace Transformers backend)"
    )
    parser.add_argument("--url",        help="단일 YouTube URL")
    parser.add_argument("--file",       help="단일 로컬 영상/이미지 파일 경로")
    parser.add_argument("--csv",        default="./data/labels.csv", help="link,label 컬럼을 가진 CSV 파일 경로")
    parser.add_argument("--dir",        help="영상/이미지 파일이 담긴 디렉토리 경로")
    parser.add_argument("--out_dir",    default="./output_thecheat2")
    parser.add_argument("--sample_sec", type=float, default=2.0,  help="프레임 샘플링 간격(초)")
    parser.add_argument("--max_frames", type=int,   default=1000, help="최대 프레임 수")
    parser.add_argument("--top_k",      type=int,   default=3,    help="RAG top-k")
    parser.add_argument("--model",      default=MODEL_ID,         help="HuggingFace 모델 ID")
    args = parser.parse_args()

    ensure_dir(args.out_dir)

    # 모델 로드
    print("[1/3] Loading VLM model via HuggingFace Transformers...")
    processor, model = load_model(args.model)

    # RAG 인덱스 + ObjectMapper
    print("[2/3] Preparing RAG Index & Object Mapper...")
    rag = CrimeRAG(DEFAULT_DOCS_PATH)
    obj_mapper = ObjectMapper(rag.model, threshold=0.75)

    # 배치 처리
    print("[3/3] Starting Batch Analysis...")
    if args.url:
        process_single_video(args.url, "manual", processor, model, rag, obj_mapper, args)
    elif args.file:
        file_path = Path(args.file)
        if not file_path.is_file():
            print(f"\n[Error] 파일을 찾을 수 없습니다: {args.file}")
            sys.exit(1)
        process_local_file(file_path, "manual", processor, model, rag, obj_mapper, args)
    elif args.csv:
        with open(args.csv, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                url = row.get("link") or row.get("url")
                if not url:
                    continue
                process_single_video(
                    url, row.get("label", "unknown"),
                    processor, model, rag, obj_mapper, args
                )
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
            process_local_file(f, "manual", processor, model, rag, obj_mapper, args)
    else:
        print(f"\n[Error] --url / --csv / --dir 중 하나가 필요합니다.")
        sys.exit(1)


if __name__ == "__main__":
    main()