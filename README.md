# CyberCOP v0.4

영상·이미지·이미지 시퀀스에서 사이버 범죄 징후를 분석하는 GPU 추론 파이프라인입니다.
운영 기준 코드는 **`cybercop_pipeline_AdotX_v0_4.py`** (CLI)입니다.

v0.4는 v0.3까지 있던 자체 RAG(범죄유형 매칭)를 제거하고, 그 자리에 별도로 구축된 risk_agent 모델
(위험도·사기유형 분류)을 연결했습니다. 한 번 실행하면 **영상 분석부터 위험도·사기유형 판정까지 하나의
결과 JSON**에 담깁니다.

## 구성 요소

| 역할 | 사용 모델/엔진 | 위치 |
|---|---|---|
| VLM (분류·증거·객체 확인) | [google/gemma-4-26B-A4B-it](https://huggingface.co/google/gemma-4-26B-A4B-it), 26B(A4B MoE) BF16 | HF에서 자동 다운로드 |
| 임베딩 (일반객체 후보 검색) | [BAAI/bge-m3](https://huggingface.co/BAAI/bge-m3) | HF에서 자동 다운로드 |
| OCR | RapidOCR 3.9.2, PP-OCRv5 Korean, ONNX Runtime CPU | 첫 실행 시 자동 다운로드 |
| 위험도 (text_risk / guideline S1~S6) | KoSimCSE-RoBERTa 기반 분류기 2종 | `agents/` + `download_models.py` |
| 사기유형·수법 분류 | 키워드/규칙 기반 매칭 | `agents/risk_agent/risk_api.py` |

## 처리 흐름

1. 영상은 장면 전환 기반으로 대표 프레임 최대 4장 선택. 이미지는 1장, 이미지 시퀀스는 전 구간에서 고르게 4장.
2. VLM이 관찰 내용과 체크리스트 판정을 분리해 **사기 / 정상**을 출력 (파싱 실패 시 불명확).
3. VLM이 별도 샘플 프레임에서 36개 사기증거 어휘를 확인하고 `evidence_score` 산출.
4. 다음의 경우 최종 판정을 **검토필요**로 조정한다.
   - 분류 결과가 불명확
   - 정상인데 `evidence_score >= 0.15`
   - 사기인데 `evidence_score < 0.15`
5. 최종 판정이 **정상이면 이후 단계를 생략**한다 (비용 절감, `skipped: true`).
6. **사기 또는 검토필요**이면 OCR → 일반객체 확인 → 위험도·사기유형 분류를 수행한다.

자동 판정 결과는 수사·법률 판단을 대체하지 않습니다.

## 시스템 요구사항

- Python 3.11 또는 3.12 / Linux 권장
- CUDA 12.6 지원 NVIDIA 드라이버 및 GPU (requirements.txt는 PyTorch 2.12.0 + cu126 기준)
- 사용 가능한 VRAM 합계 64GB 이상 권장 (`device_map="auto"`로 다중 GPU 분산)
- 여유 디스크 80GB 이상 (VLM 약 50GB + 임베딩 약 4GB + 위험도 체크포인트 약 0.9GB)
- URL 입력 시 ffmpeg 필요

## 설치

~~~bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt

# 위험도 모델 체크포인트 내려받기 (약 848MB)
python download_models.py
~~~

requirements.txt 안에 CUDA 12.6 PyTorch index가 포함되어 있으므로 별도 index 옵션은 필요 없습니다.

VLM(Gemma 4)과 BGE-m3, RapidOCR 모델은 **첫 실행 때 자동으로 내려받습니다.** 완전히 캐시된 뒤에만
`HF_HUB_OFFLINE=1`을 사용하십시오.

### 위험도 모델 체크포인트

`agents/` 아래의 코드·토크나이저는 저장소에 포함되어 있으나, 체크포인트 두 개는 각각 400MB를 넘어
GitHub 파일 크기 제한(100MB)을 초과하므로 **GitHub Releases**로 배포합니다. `download_models.py`가
이를 받아 아래 위치에 넣습니다.

~~~text
agents/text_risk_model/checkpoints/rule14b1_final_risk_level_text_classifier_best.pt   (423MB)
agents/guideline_multitask_model/checkpoints/guideline_multitask_best.pt               (425MB)
~~~

체크포인트가 없어도 파이프라인은 동작합니다. **사기유형 분류는 키워드 기반이라 모델이 필요 없고**,
위험도(`text_risk` / `guideline_risk`)만 `available: false`로 반환됩니다.

## 주요 환경변수

| 변수 | 기본값 | 설명 |
|---|---|---|
| `VLM_MODEL` | google/gemma-4-26B-A4B-it | VLM ID 또는 로컬 경로 |
| `VLM_GPU_RESERVE_GIB` | 4 | GPU별로 추론용으로 남겨 둘 메모리(GiB) |
| `EMBED_MODEL` | BAAI/bge-m3 | SentenceTransformer 임베딩 모델 |
| `MODEL_CACHE_DIR` | 프로젝트/hf_models | VLM 로컬 저장 위치 |
| `HF_HOME` | HF 기본 캐시 | BGE-m3 등 Hub 캐시 위치 |
| `HF_HUB_OFFLINE` | 미설정 | 1이면 네트워크 없이 캐시만 사용 |
| `RISK_AGENT_DIR` | 저장소 루트 | 위험도 모델 코드(`agents/`)의 상위 경로 |
| `CUDA_VISIBLE_DEVICES` | 미설정 | 사용할 GPU 선택 |

## 실행

입력 옵션 다섯 개 중 정확히 하나가 필요합니다.

~~~bash
# 이미지 한 장
python cybercop_pipeline_AdotX_v0_4.py --file data/eximg.png

# 로컬 영상 한 개
python cybercop_pipeline_AdotX_v0_4.py --file /path/to/video.mp4

# 폴더 안 파일을 각각 독립 분석
python cybercop_pipeline_AdotX_v0_4.py --dir /path/to/media

# 이어지는 스크린샷을 한 건으로 분석
python cybercop_pipeline_AdotX_v0_4.py --seq_dir data/img/kakao1

# 단일 URL
python cybercop_pipeline_AdotX_v0_4.py --url "https://www.youtube.com/shorts/VIDEO_ID"

# CSV 배치 (label, url 또는 link 컬럼)
python cybercop_pipeline_AdotX_v0_4.py --csv data/labels.csv
~~~

주요 옵션:

| 옵션 | 기본값 | 설명 |
|---|---:|---|
| `--out_dir` | ./output_AdotX_v0.3 | 결과·다운로드 저장 경로 (이름은 레거시, v0.4도 동일 사용) |
| `--model` | google/gemma-4-26B-A4B-it | VLM ID 또는 로컬 경로 |
| `--scan_sec` | 0.5 | 대표 프레임 탐색 간격 |
| `--max_vlm_frames` | 4 | 분류 대표 프레임 최대 수 |
| `--evidence_sec` | 3.0 | 증거 탐지 샘플링 간격 |
| `--max_evidence_frames` | 15 | 증거 탐지 최대 프레임 |
| `--sample_sec` | 2.0 | OCR 샘플링 간격 |
| `--max_frames` | 1000 | OCR 최대 프레임 |
| `--escalate_thr` | 0.15 | 검토필요 전환 증거 점수 |
| `--victim_count` | 미지정(=0) | 위험도 S2용 피해자 수 수동 입력 |
| `--total_loss_won` | 미지정 | 위험도 S1용 피해금액(원) 수동 입력. 미지정 시 OCR에서 추출 시도 |

`--dino_sec` / `--max_dino_frames`는 이전 자동화 호환을 위한 별칭입니다 (Grounding DINO는 실행되지 않음).

결과는 `<out_dir>/results/ocr_results_<ID>.json`에 저장됩니다. 처리 실패가 한 건이라도 있으면 종료 코드 1을 반환합니다.

## 결과 형식

~~~json
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
~~~

- `final_label`이 **정상**이면 `skipped: true`이고 상세 분석 배열은 비어 있으며 `risk_assessment` 필드가 없습니다.
- `text_risk`와 `guideline_risk`는 **서로 독립적인 두 모델**의 결과입니다. 합쳐진 단일 등급은 없습니다.
- `guideline_risk`는 S1(피해금액)·S2(피해자 수)가 전체 가중치의 60%를 차지하는데, 영상 단독 분석으로는
  이 값을 알 수 없어 기본 0으로 둡니다. 따라서 등급이 보수적으로(낮게) 나오는 경향이 있으며,
  현재 구성에서는 `text_risk`가 더 신뢰 가능한 지표입니다.

## 저장소 구조

~~~text
.
├── cybercop_pipeline_AdotX_v0_4.py  # 현재 CLI/핵심 파이프라인
├── text_risk.py                     # 위험도·사기유형 분류 연결
├── download_models.py               # 위험도 체크포인트 다운로드
├── agents/                          # 위험도·사기유형 모델 코드 (체크포인트는 별도 다운로드)
│   ├── text_risk_model/             # 텍스트 전용 3-class 위험도
│   ├── guideline_multitask_model/   # S1~S6 가이드라인 결합 위험도
│   ├── risk_agent/                  # 사기유형 규칙 분류기, 금액 파서
│   └── metadata_agent/              # 범죄유형·수법 통제어휘
├── pipeline/                        # 프레임 샘플링, 분류, 증거 집계, 어휘
├── rag/allowed_objects.json         # 일반 객체 427종 (v0.4에서도 사용)
├── data/                            # 예제 입력과 CSV
├── requirements.txt
├── docs/                            # 버전별 변경 이력 (CHANGELOG*.md)
└── legacy/                          # 이전 버전 — 재현·비교용, 실행 대상 아님
    ├── cybercop_pipeline_AdotX.py, _v0_1.py, _v0_2.py, _v0_3.py
    └── app/                         # FastAPI 서버 (v0.3 기준, v0.4 미반영)
~~~

`legacy/`의 코드는 재현·비교용으로만 보관합니다. 신규 실행에는 v0.4만 사용하십시오.
이 파일들은 저장소 루트를 기준으로 상대 경로를 잡도록 작성되어 있어, 그대로 실행하려면 경로 조정이
필요합니다.

## 라이선스

프로젝트 코드는 LICENSE를 따릅니다. Gemma 4, BGE-m3 등 외부 모델은 각 배포 페이지의 라이선스를
별도로 확인하십시오.

---
최종 업데이트: 2026-09-09 (v0.4)
