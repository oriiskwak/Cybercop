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

운영 기준 코드: `cybercop_pipeline_AdotX_v0_4.py` (CLI)

## Main features

- VLM(Gemma 4) 기반 사기/정상 분류 + 36개 사기증거 어휘 closed-set 확인
- BGE-m3 임베딩으로 427개 클래스 중 일반 객체 후보 검색 → VLM으로 등장 확인
- RapidOCR(PP-OCRv5 Korean) 기반 텍스트 추출
- 독립된 위험도 모델 2종(`text_risk`, `guideline` S1~S6) + 규칙 기반 사기유형 분류
- 정상 판정 시 이후 단계(OCR·객체·위험도) 생략 → 비용 절감

---

## 📁 Repository structure

```text
aop_cybercop/
├─ cybercop_pipeline_AdotX_v0_4.py  # 현재 CLI/핵심 파이프라인
├─ text_risk.py                     # 위험도·사기유형 분류 연결
├─ download_models.py               # 위험도 체크포인트 다운로드
├─ agents/                          # 위험도·사기유형 모델 코드 (체크포인트 별도 다운로드)
│   ├─ text_risk_model/             # 텍스트 전용 3-class 위험도
│   ├─ guideline_multitask_model/   # S1~S6 가이드라인 결합 위험도
│   ├─ risk_agent/                  # 사기유형 규칙 분류기, 금액 파서
│   └─ metadata_agent/              # 범죄유형·수법 통제어휘
├─ pipeline/                        # 프레임 샘플링, 분류, 증거 집계, 어휘
├─ rag/allowed_objects.json         # 일반 객체 427종 (v0.4에서도 사용)
├─ data/                            # 예제 입력과 CSV
├─ requirements.txt
├─ docs/                            # 버전별 변경 이력 (CHANGELOG*.md)
└─ legacy/                          # 이전 버전 — 재현·비교용, 실행 대상 아님
    ├─ cybercop_pipeline_AdotX.py, _v0_1.py, _v0_2.py, _v0_3.py
    └─ app/                         # FastAPI 서버 (v0.3 기준, v0.4 미반영)
```

## ⚙️ Installation

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt

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

체크포인트 없어도 파이프라인 동작함. 사기유형 분류는 키워드 기반이라 모델 불필요.
위험도(`text_risk`/`guideline_risk`)만 `available: false` 반환.

---

## 🧩 Components

| 역할 | 모델/엔진 | 위치 |
|---|---|---|
| VLM (분류·증거·객체 확인) | [google/gemma-4-26B-A4B-it](https://huggingface.co/google/gemma-4-26B-A4B-it), 26B(A4B MoE) BF16 | HF 자동 다운로드 |
| 임베딩 (일반객체 후보 검색) | [BAAI/bge-m3](https://huggingface.co/BAAI/bge-m3) | HF 자동 다운로드 |
| OCR | RapidOCR 3.9.2, PP-OCRv5 Korean, ONNX Runtime CPU | 첫 실행 시 자동 다운로드 |
| 위험도 (text_risk / guideline S1~S6) | KoSimCSE-RoBERTa 기반 분류기 2종 | `agents/` + `download_models.py` |
| 사기유형·수법 분류 | 키워드/규칙 기반 매칭 | `agents/risk_agent/risk_api.py` |

## 🔄 Workflow

1. 대표 프레임 선택 — 영상: 장면 전환 기반 최대 4장 / 이미지: 1장 / 이미지 시퀀스: 전 구간 고르게 4장
2. VLM이 관찰 내용·체크리스트 판정 분리 → 사기/정상 출력 (파싱 실패 시 불명확)
3. VLM이 별도 샘플 프레임에서 36개 사기증거 어휘 확인 → `evidence_score` 산출
4. 검토필요 전환 조건
   - 분류 결과 불명확
   - 정상 + `evidence_score >= 0.15`
   - 사기 + `evidence_score < 0.15`
5. 정상 확정 → 이후 단계 생략 (`skipped: true`, 비용 절감)
6. 사기/검토필요 → OCR → 일반객체 확인 → 위험도·사기유형 분류 수행

## 💻 Requirements

- Python 3.11 또는 3.12 / Linux 권장
- CUDA 12.6 지원 NVIDIA 드라이버·GPU (requirements.txt: PyTorch 2.12.0 + cu126 기준)
- VRAM 합계 64GB↑ 권장 (`device_map="auto"`로 다중 GPU 분산)
- 여유 디스크 80GB↑ 권장 (VLM 약 50GB + 임베딩 약 4GB + 위험도 체크포인트 약 0.9GB)
- URL 입력 시 ffmpeg 필요

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
| `--model` | google/gemma-4-26B-A4B-it | VLM ID 또는 로컬 경로 |
| `--scan_sec` | 0.5 | 대표 프레임 탐색 간격 |
| `--max_vlm_frames` | 4 | 분류 대표 프레임 최대 수 |
| `--evidence_sec` | 3.0 | 증거 탐지 샘플링 간격 |
| `--max_evidence_frames` | 15 | 증거 탐지 최대 프레임 |
| `--sample_sec` | 2.0 | OCR 샘플링 간격 |
| `--max_frames` | 1000 | OCR 최대 프레임 |
| `--escalate_thr` | 0.15 | 검토필요 전환 증거 점수 |
| `--victim_count` | 미지정(=0) | 위험도 S2용 피해자 수 수동 입력 |
| `--total_loss_won` | 미지정 | 위험도 S1용 피해금액(원) 수동 입력, 미지정 시 OCR 추출 시도 |

`--dino_sec` / `--max_dino_frames`: 이전 자동화 호환용 별칭 (Grounding DINO는 실행 안 됨)

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
  "ocr_before": ["..."],
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

- `final_label` 정상 → `skipped: true`, 상세 분석 배열 비어 있음, `risk_assessment` 필드 없음
- `text_risk`·`guideline_risk`는 서로 독립된 두 모델 결과. 합쳐진 단일 등급 없음
- `guideline_risk`는 S1(피해금액)·S2(피해자 수)가 전체 가중치의 60% 차지. 영상 단독 분석으로는 값을 알 수 없어 기본 0 처리 → 등급이 보수적(낮게)으로 나오는 경향. 현재 구성에서는 `text_risk`가 더 신뢰 가능한 지표.

<p align="center"><sub>최종 업데이트: 2026-09-09 (v0.4)</sub></p>
