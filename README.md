<h1 align="center">CyberCOP</h1>
<p align="center">
  <b>VLM 기반 사이버 범죄 징후 탐지 및 위험도·사기유형 판정 파이프라인</b>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Version-0.4-blue" />
  <img src="https://img.shields.io/badge/Python-3.11%20%7C%203.12-blue" />
  <img src="https://img.shields.io/badge/CUDA-12.6-76B900" />
  <img src="https://img.shields.io/badge/Domain-Cybercrime%20Detection-orange" />
</p>

---

VLM(Gemma 4) 기반 GPU 추론 파이프라인. 영상·이미지·이미지 시퀀스에서 사기/정상 여부를 판별하고,
별도 구축된 risk_agent 모델로 위험도·사기유형까지 판정. 결과는 하나의 JSON으로 통합.

운영 기준 코드: `cybercop_pipeline_AdotX_v0_4.py` (CLI), `app/main_v0_4.py` (API 서버)

## Main features

- VLM(Gemma 4) 기반 사기/정상 분류 + 36개 사기증거 어휘 closed-set 확인 (영상 전체에서 고르게 최대 15장)
- BGE-m3 임베딩으로 427개 클래스 중 일반 객체 후보 검색 → VLM으로 등장 확인
- 이모티콘 9종·아이콘 50종 closed-set 탐지 (표시용, 판정에는 미사용)
- RapidOCR(PP-OCRv5 Korean) 기반 텍스트 추출
- 독립된 위험도 모델 2종(`text_risk`, `guideline` S1~S6) + 규칙 기반 사기유형 분류
- 정상 판정 시 이후 단계(OCR·객체·위험도) 생략 → 비용 절감

---

## 📁 Repository structure

```text
Cybercop/
├─ cybercop_pipeline_AdotX_v0_4.py  # 현재 CLI/핵심 파이프라인
├─ text_risk.py                     # 위험도·사기유형 분류 연결
├─ download_models.py               # 위험도 체크포인트 다운로드
├─ agents/                          # 위험도·사기유형 모델 코드 (체크포인트 별도 다운로드)
│   ├─ text_risk_model/             # 텍스트 전용 3-class 위험도
│   ├─ guideline_multitask_model/   # S1~S6 가이드라인 결합 위험도
│   ├─ risk_agent/                  # 사기유형 규칙 분류기, 금액 파서
│   └─ metadata_agent/              # 범죄유형·수법 통제어휘
├─ app/                             # FastAPI 서버
│   ├─ main_v0_4.py                 # 현재 API (v0.4 파이프라인 연결)
│   ├─ main.py, main_v0_1.py ~ main_v0_3.py  # 이전 버전 API — 보관용
│   └─ __init__.py                  # app 폴더를 파이썬 패키지로 만드는 파일
├─ pipeline/                        # 프레임 샘플링, 분류, 증거 집계, 어휘
│   ├─ frame_sampler.py             # 프레임 추출 (균등 간격 / 장면 전환 / 영상 전체 고르게)
│   └─ vocab.py                     # 사기증거 36종, 이모티콘 9종, 아이콘 50종
├─ rag/allowed_objects.json         # 일반 객체 427종 (v0.4에서도 사용)
├─ data/                            # 예제 입력과 CSV
├─ requirements.txt
├─ install_requirements.py          # 의존성 라이브러리 설치 프로그램
├─ docs/                            # 버전별 변경 이력 (CHANGELOG*.md)
└─ legacy/                          # 이전 버전 파이프라인 — 재현·비교용, 실행 대상 아님
    └─ cybercop_pipeline_AdotX.py, _v0_1.py, _v0_2.py, _v0_3.py
```

- `app/__init__.py`: 내용은 설명 한 줄뿐이지만 삭제 금지. 이 파일이 있어야 `app` 폴더가 패키지로
  인식되어 `uvicorn app.main_v0_4:app` 처럼 `app.파일명` 형태로 서버 실행 가능.
- `app/main.py`, `main_v0_1.py` ~ `main_v0_3.py`: 버전별 API 기록 보관용. 각자 같은 버전의 파이프라인
  (`legacy/cybercop_pipeline_AdotX_v0_x.py`)을 import하므로 그대로는 실행 불가. 운영은 `main_v0_4.py`만 사용.

## ⚙️ Installation

```bash
sudo apt install python3.12-dev   # INT4 VLM 실행에 필요 (시스템 패키지)
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip

# 의존성 라이브러리 다운로드
python install_requirements.py

# 위험도 모델 체크포인트 다운로드 (약 848MB)
python download_models.py
```

VLM(Gemma 4), BGE-m3, RapidOCR는 첫 실행 시 자동 다운로드. 캐시 완료 후에만 `HF_HUB_OFFLINE=1` 사용.

### risk model checkpoint

체크포인트 2개는 각 400MB↑로 GitHub 파일 크기 제한(100MB) 초과 → GitHub Releases로 배포.
`download_models.py`가 아래 위치에 다운로드.

```text
agents/text_risk_model/checkpoints/rule14b1_final_risk_level_text_classifier_best.pt   (423MB)
agents/guideline_multitask_model/checkpoints/guideline_multitask_best.pt               (425MB)
```

체크포인트가 없으면 위험도(`text_risk`/`guideline_risk`, 하/중/상)가 `available: false`로 비어서 나옴.
사기/정상 판정·객체·OCR·사기유형 분류는 그대로 동작하므로, 위험도가 비어 있으면 체크포인트 설치 여부부터 확인.

---

## 🧩 Components

| 역할 | 모델/엔진 | 위치 |
|---|---|---|
| VLM (분류·증거·객체 확인) | [Intel/gemma-4-26B-A4B-it-int4-AutoRound](https://huggingface.co/Intel/gemma-4-26B-A4B-it-int4-AutoRound), 26B(A4B MoE) INT4 auto-round 양자화 | HF 자동 다운로드 |
| 임베딩 (일반객체 후보 검색) | [BAAI/bge-m3](https://huggingface.co/BAAI/bge-m3) | HF 자동 다운로드 |
| OCR | RapidOCR 3.9.2, PP-OCRv6 Detection, PP-OCR-v5 Korean Recognition, ONNX Runtime CPU | 첫 실행 시 자동 다운로드 |
| 위험도 (text_risk / guideline S1~S6) | KoSimCSE-RoBERTa 기반 분류기 2종 | `agents/` + `download_models.py` |
| 사기유형·수법 분류 | 키워드/규칙 기반 매칭 | `agents/risk_agent/risk_api.py` |

## 🔄 Workflow

1. 대표 프레임 선택 — 영상: 장면 전환 기반 최대 4장 / 이미지: 1장 / 이미지 시퀀스: 전 구간 고르게 4장
2. VLM이 관찰 내용·체크리스트 판정 분리 → 사기/정상 출력 (파싱 실패 시 불명확)
3. VLM이 영상 전체에서 고르게 뽑은 최대 15장에서 36개 사기증거 어휘 확인 → `evidence_score` 산출
4. 검토필요 전환 조건
   - 분류 결과 불명확
   - 정상 + `evidence_score >= 0.15`
   - 사기 + `evidence_score < 0.15`
5. 정상 확정 → 이후 단계 생략 (`skipped: true`, 비용 절감)
6. 사기/검토필요 → OCR → 일반객체 확인 → 이모티콘·아이콘 확인(3단계와 같은 프레임) → 위험도·사기유형 분류 수행



## 💻 Requirements

- Python 3.11 또는 3.12 / Linux 권장
- CUDA 12.6 지원 NVIDIA 드라이버·GPU (requirements.txt: PyTorch 2.12.0 + cu126 기준)
- VRAM 합계 20GB↑ 권장 (`device_map="auto"`로 다중 GPU 분산 지원)
- 여유 디스크 30GB↑ 권장 (VLM 약 50GB + 임베딩 약 4GB + 위험도 체크포인트 약 0.9GB)

## 🔧 Settings

| 변수 | 기본값 | 설명 |
|---|---|---|
| `VLM_MODEL` | google/gemma-4-26B-A4B-it | VLM ID 또는 로컬 경로 |
| `VLM_GPU_RESERVE_GIB` | 4 | GPU별 추론용 예약 메모리(GiB) |
| `EMBED_MODEL` | BAAI/bge-m3 | SentenceTransformer 임베딩 모델 |
| `MODEL_CACHE_DIR` | 프로젝트/hf_models | VLM 로컬 저장 위치 |
| `HF_HOME` | HF 기본 캐시 | BGE-m3 등 Hub 캐시 위치 |
| `HF_HUB_OFFLINE` | 미설정 | 1이면 네트워크 없이 캐시만 사용 |
| `RISK_AGENT_DIR` | 저장소 루트 | 위험도 모델 코드(`agents/`)의 상위 경로 |
| `CUDA_VISIBLE_DEVICES` | 미설정 | 사용할 GPU 선택 |

---

## ▶️ Run

입력 옵션 5개 중 정확히 1개 필요.

```bash
# 이미지 한 장
python cybercop_pipeline_AdotX_v0_4.py --file data/eximg.png

# 로컬 영상 한 개
python cybercop_pipeline_AdotX_v0_4.py --file /path/to/video.mp4

# 폴더 안 파일 각각 독립 분석
python cybercop_pipeline_AdotX_v0_4.py --dir /path/to/media

# 이어지는 스크린샷을 한 건으로 분석
python cybercop_pipeline_AdotX_v0_4.py --seq_dir data/img/kakao1

# 단일 URL
python cybercop_pipeline_AdotX_v0_4.py --url "https://www.youtube.com/shorts/VIDEO_ID"

# CSV 배치 (label, url 또는 link 컬럼)
python cybercop_pipeline_AdotX_v0_4.py --csv data/labels.csv
```

**Option**

| 옵션 | 기본값 | 설명 |
|---|---:|---|
| `--out_dir` | ./output_AdotX_v0.3 | 결과·다운로드 저장 경로 (이름 레거시, v0.4도 동일 사용) |
| `--model` | Intel/gemma-4-26B-A4B-it-int4-AutoRound | VLM ID 또는 로컬 경로 |
| `--scan_sec` | 0.5 | 대표 프레임 탐색 간격 |
| `--max_vlm_frames` | 4 | 분류 대표 프레임 최대 수 |
| `--evidence_sec` | 3.0 | 사용 안 함 (이전 자동화 호환용) |
| `--max_evidence_frames` | 15 | 사기증거·이모티콘·아이콘 탐지 프레임 수 (영상 전체에 고르게) |
| `--sample_sec` | 2.0 | OCR 샘플링 간격 |
| `--max_frames` | 1000 | OCR 최대 프레임 |
| `--escalate_thr` | 0.15 | 검토필요 전환 증거 점수 |

결과 저장 위치: `<out_dir>/results/ocr_results_<ID>.json`. 처리 실패 1건 이상 시 종료 코드 1 반환.

## 📊 Result format

```json
{
  "id": "eximg",
  "url": "data/eximg.png",
  "title": "eximg.png",
  "label": "manual",
  "cls_label": "사기",
  "final_label": "검토필요",
  "classify_summary": "관찰 내용 ... / 판단근거: ...",
  "grounding": { "labels": {}, "evidence_score": 0.5, "n_detected_frames": 1, "top_labels": ["비트코인 화면"] },
  "objects": ["비트코인 화면", "안내 문구"],
  "scam_evidence": ["비트코인 화면"],
  "general_objects": ["안내 문구"],
  "icon_emoji_objects": { "emoticons": ["하트"], "icons": ["화살표"] },
  "ocr_before": ["..."],
  "ocr_frames": [{ "frame_idx": 0, "sec": 0.0, "time": "00:00", "text": "..." }],
  "ocr": [{ "start": "00:00", "end": "00:00", "text": "..." }],
  "ocr_after": "...",
  "ocr_candidates": [],
  "skipped": false,
  "total_inference_time": 12.3,
  "risk_assessment": {
    "available": true,
    "statement_text": "위험도 모델에 실제로 입력된 텍스트",
    "victim_count_used": 0,
    "total_loss_won_used": 0,
    "amount_source": "none",
    "text_risk": { "pred_label": "상", "probabilities": { "하": 0.01, "중": 0.06, "상": 0.93 } },
    "guideline_risk": { "signal_based_level": "중", "score": 41.5, "indicator_scores": { "S1": 100, "S2": 0, "S3": 0, "S4": 100, "S5": 50, "S6": 0 } },
    "crime_classification": { "top_crime_type": "사이버금융범죄", "top_crime_method": "사이버투자사기" }
  }
}
```

- 탐지 객체는 `scam_evidence`(사기증거, 등장 비율 상위 5개 — 전체는 `grounding.labels`) / `general_objects`(일반 객체)
  / `icon_emoji_objects`(이모티콘·아이콘)로 구분. `objects`는 앞의 둘을 합친 기존 호환 필드.
- `ocr_frames`: OCR 프레임마다 읽은 텍스트와 시각. 글자가 없는 프레임도 빈 `text`로 포함.
- `final_label` 정상 → `skipped: true`, 상세 분석 배열 비어 있음, `risk_assessment` 필드 없음
- `text_risk`·`guideline_risk`는 서로 독립된 두 모델 결과. 합쳐진 단일 등급 없음
- `guideline_risk`는 S1(피해금액)·S2(피해자 수)가 전체 가중치의 60% 차지. 영상 단독 분석으로는 값을 알 수 없어 기본 0 처리 → 등급이 보수적(낮게)으로 나오는 경향. 현재 구성에서는 `text_risk`가 더 신뢰 가능한 지표.

---

## 🌐 API server (v0.4)

```bash
# 저장소 루트에서 실행 (app/__init__.py가 있어야 app.main_v0_4 로 불러올 수 있음)
uvicorn app.main_v0_4:app --host 0.0.0.0 --port 8000 --workers 1
```

시작 시 VLM·OCR·임베딩·위험도 모델을 모두 로드한 뒤 `/health/ready`가 200 반환. 요청 화면은 `http://<서버>:8000/docs`.
위 Settings의 환경변수(`VLM_MODEL`, `MODEL_CACHE_DIR`, `HF_HOME` 등) 그대로 적용. API 전용 설정은 아래와 같음.

| 변수 | 기본값 | 설명 |
|---|---|---|
| `CYBERCOP_API_KEY` | 미설정 | 설정하면 `/health/*`를 뺀 모든 요청에 `X-API-Key` 헤더 필요 |
| `CYBERCOP_WORK_DIR` | /tmp/cybercop | 업로드·다운로드 임시 폴더 (요청이 끝나면 삭제) |
| `MAX_CONCURRENT_ANALYSES` | 1 | 동시에 처리할 분석 수 (GPU 1장 기준 1) |
| `MAX_UPLOAD_BYTES` | 250MB | 업로드 파일 크기 제한 |
| `MAX_SEQUENCE_IMAGES` | 50 | 이미지 시퀀스 최대 장수 |
| `MAX_CSV_ROWS` | 100 | CSV 일괄 분석 최대 행 수 |
| `ALLOWED_URL_HOSTS` | youtube.com,youtu.be,tiktok.com | 허용 URL 호스트 |
| `DEFAULT_SAMPLE_SEC` / `DEFAULT_MAX_FRAMES` / `MAX_FRAMES_LIMIT` | 2.0 / 1000 / 1000 | OCR 프레임 인자 기본값·상한 |

| 메서드 | 경로 | 입력 | 설명 |
|---|---|---|---|
| GET | `/health/live`, `/health/ready` | — | 프로세스 생존 / 모델 준비 여부 (준비 전 503) |
| GET | `/api/info` | — | 모델·어휘 크기·기본값 |
| POST | `/api/video` | form `url` | YouTube/TikTok HTTPS URL |
| POST | `/api/video/upload` | multipart `file` | 영상 또는 이미지 1개 |
| POST | `/api/video/upload/sequence` | multipart `files`(여러 장) 또는 `archive`(zip), `sequence_id` | 이어진 스크린샷을 한 건으로 |
| POST | `/api/video/csv` | multipart `file` (`label`, `link`/`url` 컬럼) | URL 여러 건 |

요청 인자 `sample_sec`(OCR 간격, 초)·`max_frames`(OCR 최대 장수)는 **OCR에만** 적용. 판정에 쓰는 프레임
(대표 4장, 사기증거 15장)은 고정이라 이 값으로 판정이 바뀌지 않음.

응답 형식은 `{"message": "success", "request_id": ..., "result": {...}}`. `result`는 CLI 결과 JSON과 아래 항목만 다름.

| 필드 | 내용 |
|---|---|
| `scam_objects` / `general_objects` / `icon_emoji_objects` | 탐지 객체 3필드 (`scam_objects` = `scam_evidence`, 상위 5개) |
| `ocr_after` | OCR은 전체 텍스트만 제공. `ocr_frames`·`ocr`·`ocr_before`·`ocr_candidates`는 응답에서 제외 |
| `ocr_settings` | 이번 요청의 `sample_sec`, `max_frames`, 실제 OCR 프레임 수 `n_frames` |
| `review_reason` | `final_label`이 검토필요일 때 그 사유 한 문장, 그 외 `null` |
| `risk_assessment` | 정상 판정이면 `null` |
| `label` / `input_label` | 판정이 사기·검토필요면 `abnormal`, 정상이면 `normal` / 요청 쪽 라벨 |

<p align="center"><sub>최종 업데이트: 2026-10-06 (v0.4)</sub></p>
