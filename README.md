# Video OCR & RAG 기반 사이버 범죄 탐지 파이프라인

YouTube/TikTok 영상 또는 로컬 영상/이미지에서 프레임 추출 → VLM(SKT A.X-4.0-VL-Light)으로 OCR/객체 인식 →
RAG(FAISS + BGE-m3)로 사이버 범죄 유형 매칭 및 위험도 산출.

이 저장소엔 서로 다른 세대의 파이프라인이 같이 있음. 뭐가 뭔지는 아래 [파이프라인 버전 안내](#파이프라인-버전-안내) 먼저 확인.

> ⚠️ **TikTok URL 현재 막힘**: `--url`/`--csv`(TikTok 링크)와 `POST /api/video`(TikTok URL)가 TikTok의
> 봇 차단(챌린지 페이지)에 걸려서 `Unexpected response from webpage request` 에러로 실패함. yt-dlp는 이미
> 최신 버전이라 업데이트로 해결 안 됨 — 우회 방안(쿠키 등) 찾는 중. YouTube URL과 로컬 파일(`--file`/`--dir`/
> `--seq_dir`, 이미지·영상 업로드) 입력은 이 문제와 무관하게 정상 동작.

## 시스템 요구사항

- **Python**: 3.10
- **CUDA**: 12.4 (GPU 필수)
- **VRAM**: 10GB 이상
- **테스트 환경**: torch 2.6.0 + transformers 4.51.3

## 디렉토리 구조

```
├── cybercop_pipeline_AdotX.py       # 원본 파이프라인 (레거시, 참고용 — 실행 X)
├── cybercop_pipeline_AdotX_v0_1.py  # 이전 버전 (CoT 분류 + Grounding DINO, --seq_dir 없음)
├── cybercop_pipeline_AdotX_v0_2.py  # 현재 사용하는 파이프라인 (v0.1 + 이미지 시퀀스 입력)
├── pipeline/                        # AdotX_v0_1.py/v0_2.py 공용 서브모듈 (frame_sampler, vlm_classify, grounding_detector, evidence_aggregator, vocab)
├── CHANGELOG.md                     # AdotX.py → AdotX_v0_1.py 변경 이력
├── CHANGELOG_0.2.md                 # AdotX_v0_1.py → AdotX_v0_2.py 변경 이력
├── requirements.txt                 # 패키지 목록
├── install.bat                      # Windows 설치 스크립트
├── app/
│   ├── main.py                      # 원본 파이프라인 FastAPI 서버 (레거시)
│   ├── main_v0_1.py                 # FastAPI 서버 (이전 버전, AdotX_v0_1.py 기준 — --seq_dir 없음)
│   └── main_v0_2.py                 # FastAPI 서버 (권장, AdotX_v0_2.py 기준 — --seq_dir/이미지 시퀀스 업로드 포함)
├── detection_ocr/                   # 개발용 폴더. AdotX_v0_1.py는 여기서 개발되어 루트로 배포 완료
│   └── cybercop_pipeline_v2.py      # 개발 중간 단계 (참고용)
├── data/
│   ├── labels.csv                   # 테스트용 영상 URL 목록 (label,url) — --csv 기본값
│   ├── thecheat.csv                 # 테스트용 영상 URL 목록 (추가 세트) — 기본값 아님, --csv로 직접 지정해야 씀
│   ├── img/                         # --seq_dir 테스트용 이미지 시퀀스 폴더
│   │   └── kakao1/                  # 카톡 대화 스크린샷 예시 (6장)
│   └── vlm_fallback_object_frequency.csv  # VLM 폴백 원본 출력 빈도 집계 (allowed_objects.json 어휘 보강 검토용, 1회성 진단 산출물)
├── rag/
│   ├── allowed_objects.json         # 허용 객체 클래스 목록 (439종, JSON DB) — VLM 폴백 결과 정규화용 (두 파이프라인 다 사용)
│   └── retrieval_docs.json          # 범죄 유형 문서 (26종 + 객체 키워드 자동 추가분) — 두 파이프라인 다 사용
├── downloaded_videos/                # 다운로드 영상 저장 (자동 생성, git 비공개)
└── hf_models/                        # 모델 캐시 (자동 생성, git 비공개)
```

> `*.mp4` 등 다운로드 영상 원본, `hf_models/`는 `.gitignore`로 제외 (public 저장소라 스크래핑한 영상/증거
> 텍스트는 올리지 않음). 코드 받은 뒤 각자 환경에서 파이프라인 돌려서 산출물은 로컬에서 재생성.
>
> `classification/`, `extract_cyber_objects.py`/`review_and_add_objects.py`, `detection_ocr/eval_metrics.py`,
> `detection_ocr/output_v2`·`output_v3`는 지금 워크플로에서 안 씀. 로컬엔 남아있지만 `.gitignore`로 push 제외.

## 파이프라인 버전 안내

| 파일 | 상태 | 분류 방식 | 특징 |
|---|---|---|---|
| `cybercop_pipeline_AdotX.py` | 레거시 (참고용, 실행 X) | 단일 VLM 호출로 OCR+객체 동시 추출, RAG 유사도 임계값으로 abnormal/normal 자동 판정 | — |
| `cybercop_pipeline_AdotX_v0_1.py` | 이전 버전 (실행은 가능, 신규 기능 없음) | 2단계 CoT(묘사→판단) 분류 + Grounding DINO 증거탐지 + OCR 프레임간 클러스터링(IoU+CER) | 사기 판정 영상만 OCR/상세분석 실행, evidence_score 기반 정상→검토필요 에스컬레이션 |
| `cybercop_pipeline_AdotX_v0_2.py` | **현재 사용** | v0.1과 동일 + 이미지 시퀀스(`--seq_dir`) 입력 지원 | 이어지는 스크린샷 여러 장을 폴더 전체 한 건으로 판정, 이미지 입력엔 OCR 클러스터링 대신 이미지별 개별 교정 사용 |

`AdotX.py` → `AdotX_v0_1.py` 변경 내역은 **[CHANGELOG.md](./CHANGELOG.md)**, `AdotX_v0_1.py` → `AdotX_v0_2.py`
변경 내역은 **[CHANGELOG_0.2.md](./CHANGELOG_0.2.md)** 참고. 새 버전이 나올 때마다 이전 버전 파일은 그대로 두고
다음 버전 파일을 새로 만드는 방식으로 관리함 (버전 파일을 직접 덮어쓰지 않음).
`detection_ocr/`는 `AdotX_v0_1.py` 개발하던 작업 폴더. 실제 배포/실행은 루트의 `cybercop_pipeline_AdotX_v0_2.py` 사용
(`detection_ocr/cybercop_pipeline_v3.py`와 `pipeline/` 서브모듈은 루트 배포 후 중복이라 삭제, `app/main_v0_2.py`도
루트 파일을 직접 import). API로 `--seq_dir`(이미지 시퀀스)까지 쓰려면 `app/main_v0_2.py`를 띄우세요 —
`app/main_v0_1.py`는 아직 `AdotX_v0_1.py` 기준이라 이 기능이 없음.

## 설치

```bash
pip install -r requirements.txt --index-url https://download.pytorch.org/whl/cu124 --extra-index-url https://pypi.org/simple
```

> Linux GPU 서버: `requirements.txt`의 `faiss-cpu` → `faiss-gpu`로 바꿔서 설치

---

## 실행 방법 (`cybercop_pipeline_AdotX_v0_2.py`)

```bash
python cybercop_pipeline_AdotX_v0_2.py --csv "data/labels.csv"
```

단일 URL:
```bash
python cybercop_pipeline_AdotX_v0_2.py --url "https://www.youtube.com/shorts/영상ID"
```

로컬 파일(영상 또는 이미지) 하나:
```bash
python cybercop_pipeline_AdotX_v0_2.py --file "data/eximg.png"
```

로컬 폴더(안의 영상/이미지 파일 전부 각각 독립 판정):
```bash
python cybercop_pipeline_AdotX_v0_2.py --dir "C:\사이버 범죄 데이터\직거래 사기"
```

이어지는 이미지 시퀀스(카톡 대화 스크린샷 등, 폴더 전체를 한 건으로 판정) — **v0.2 신규**:
```bash
python cybercop_pipeline_AdotX_v0_2.py --seq_dir "data/img/kakao1"
```

> `--url` / `--file` / `--dir` / `--seq_dir` / `--csv` 중 하나 (우선순위도 이 순서). 이미지는 시간 축이 없어서
> `--file`은 1~3단계 모두 같은 프레임 1장을 재사용, `--seq_dir`은 폴더 안 이미지들을 파일명 순으로 정렬해
> 하나의 프레임 시퀀스로 묶음 (분류용 대표 프레임은 `--max_vlm_frames`보다 많으면 전체 구간에서 고르게 추출).
> `output_AdotX_v0.2/results/ocr_results_{id}.json`이 이미 있으면 자동 스킵하므로, CSV에 새 행만 추가하고
> 다시 돌리면 새로 추가된 영상만 처리됨 (`--seq_dir`은 폴더명이 `{id}`).

### 옵션
| 인자 | 기본값 | 설명 |
|---|---|---|
| `--csv` | `data/labels.csv` | `label`/`link`(또는 `url`) 컬럼을 가진 CSV |
| `--seq_dir` | — | 이어지는 이미지 시퀀스가 담긴 디렉토리 경로 (폴더 전체를 한 건으로 판정, **v0.2 신규**) |
| `--out_dir` | `./output_AdotX_v0.2` | 결과 저장 경로 |
| `--model` | `skt/A.X-4.0-VL-Light` | 분류/OCR-객체 추출용 VLM |
| `--gdino_model` | `IDEA-Research/grounding-dino-tiny` | 증거탐지용 Grounding DINO |
| `--scan_sec` | `0.5` | 분류용 키프레임 스캔 간격 (장면전환 기반 최대 4프레임 선정, 영상만 해당) |
| `--max_vlm_frames` | `4` | 분류에 사용할 대표 프레임 수 |
| `--dino_sec` | `3.0` | DINO 증거탐지 프레임 간격 (영상만 해당) |
| `--max_dino_frames` | `15` | DINO에 넘길 최대 프레임 수 (영상만 해당) |
| `--sample_sec` | `2.0` | 사기/검토필요 판정 시 OCR용 균등 샘플링 간격 (영상만 해당) |
| `--max_frames` | `1000` | OCR용 최대 프레임 수 (영상만 해당) |
| `--escalate_thr` | `0.15` | evidence_score가 이 값 이상이면 "정상"→"검토필요"로 에스컬레이션 |
| `--top_k` | `3` | RAG 검색 상위 문서 수 |

### 처리 파이프라인

```
1단계 CoT 분류(묘사→판단) → 2단계 Grounding DINO 증거탐지(전체 영상) → 에스컬레이션 판정
   → (사기/검토필요만) OCR 프레임간 클러스터링 + 오브젝트 상세분석 → RAG 검색
```

1. **1단계 분류**: 장면전환 기반 키프레임(최대 4개)을 grid 이미지로 만들어 VLM에 입력 → ①이미지→서술(판단 없이 관찰만) ②서술 텍스트→8개 체크리스트 기준 판단(사기/정상). 이미지 인코딩 없는 2단계는 1단계보다 5~10배 빠름
2. **Grounding DINO 증거탐지**: 모든 영상에서 실행. 3초 간격 최대 15프레임에서 36종 사기 증거 어휘(주식차트, 카카오톡/텔레그램 화면, 계좌번호, QR코드, 검찰청 공문서 등, `SCAM_EVIDENCE_VOCAB`)를 zero-shot 탐지, 영상 단위로 집계해 `evidence_score`/`top_labels` 산출
3. **에스컬레이션**: CoT가 "정상"으로 판단해도 `evidence_score ≥ escalate_thr`이면 최종 라벨을 "검토필요"로 올림 (사기를 놓치는 것보다 과탐지가 낫다는 원칙). `evidence_score`/`top_labels` 둘 다 `avg_score >= 0.55`(`MIN_LABEL_SCORE`, `pipeline/evidence_aggregator.py`)인 탐지만 반영 — 경계선 애매한 탐지가 정상 영상을 검토필요로 잘못 넘기거나 RAG 쿼리를 오염시키는 걸 방지
4. **OCR + 상세분석** (최종 라벨이 사기/검토필요인 영상만): PaddleOCR로 텍스트 추출.
   - **영상**: 프레임 간 **IoU(같은 화면 위치) + CER(비슷한 문자열)** 기반 Union-Find 클러스터링으로 동일 UI 텍스트의 OCR 오탈자를 다수결 병합 → 클러스터링된 전체 타임라인을 VLM에 한 번 더 넣어 오타/할루시네이션 교정. 실시간으로 바뀌는 숫자값(호가 등)은 클러스터링 대상에서 제외해 오탐 병합 방지
   - **이미지 시퀀스(`--seq_dir`, v0.2 신규)**: 이 클러스터링은 안 씀 — 각 이미지가 서로 다른 화면이라 "같은 요소가 여러 프레임에 반복 등장"한다는 전제 자체가 안 맞기 때문. 대신 이미지별 원문(`ocr_before[i]`)을 그대로 유지한 채 VLM으로 이미지 1장씩 개별 오타 교정만 수행 (문장 재배열·타 이미지와 병합 없음)
5. **객체 인식** (최종 라벨이 사기/검토필요인 영상만): Grounding DINO의 `top_labels`(36종 어휘, 위 2번)와 VLM 자유서술 폴백 결과를 **항상 둘 다** 실행해서 합침 — DINO는 정밀하지만 좁은 어휘라 놓치는 게 많고 VLM은 넓지만 부정확할 수 있어서, 사기로 확정된 영상은 증거를 최대한 남기기 위해 서로 보완시킴. VLM 폴백 결과는 `ObjectMapper`(임베딩 유사도, threshold 0.7)로 `rag/allowed_objects.json`(439종)에 정규화해서 합침 (정상 영상은 이 단계 자체를 스킵)
6. **RAG 검색**: 병합/교정된 OCR 텍스트 + 감지 객체를 쿼리로 범죄 유형 문서 검색

### Output
`output_AdotX_v0.2/results/ocr_results_{id}.json`:

```json
{
  "id": "영상ID",
  "url": "원본 URL",
  "title": "영상 제목",
  "label": "abnormal",
  "cls_label": "사기",
  "final_label": "사기",
  "classify_summary": "관찰 서술 + 판단근거 (묘사 부분은 할루시네이션 가능성 있음 — 정확한 텍스트는 ocr/ocr_after 참고)",
  "grounding": {
    "labels": { "카카오톡 화면": { "count": 3, "frame_count": 2, "frame_ratio": 0.4, "segment": "early", "avg_score": 0.62 } },
    "evidence_score": 0.38,
    "n_detected_frames": 5,
    "top_labels": ["카카오톡 화면", "주식 차트"]
  },
  "objects": ["휴대전화", "채팅창"],
  "ocr_before": ["프레임별(또는 이미지별) 원시 OCR 텍스트", "..."],
  "ocr": [{ "start": "00:00", "end": "00:04", "text": "지금 투자하면 300% 수익 보장! @kakao_id" }],
  "ocr_after": "지금 투자하면 300% 수익 보장! @kakao_id | ...",
  "ocr_candidates": [{ "candidates": [{ "text": "...", "count": 3, "avg_score": 0.91 }], "dominant_share": 0.75, "stable": true, "bbox": [10.0, 10.0, 100.0, 26.0] }],
  "rag": [{ "crime_type": "피싱", "similarity": 0.87, "risk_level": 0.22 }],
  "total_inference_time": 32.69,
  "skipped": false
}
```

`final_label`이 정상이면 `skipped: true`, `objects`/`ocr`/`ocr_after`/`rag`는 빈 값.
`--seq_dir`(이미지 시퀀스) 입력은 `ocr_candidates`가 항상 빈 배열 (클러스터링을 안 쓰므로).

### 평가 (accuracy/precision/recall/f1/평균 추론시간)
```bash
python detection_ocr/eval_metrics.py --results_dir output_AdotX_v0.2/results --csv data/labels.csv --out output_AdotX_v0.2/eval_summary.json
```

---

## 이전 버전들

- **`cybercop_pipeline_AdotX_v0_1.py`**: v0.2의 직전 버전. CoT 분류 + Grounding DINO는 동일하고,
  `--seq_dir`(이미지 시퀀스 입력)만 없음. 실행 자체는 가능 (`--url`/`--file`/`--dir`/`--csv`).
  v0.1→v0.2 차이는 [CHANGELOG_0.2.md](./CHANGELOG_0.2.md) 참고.
- **`cybercop_pipeline_AdotX.py`(레거시)**: 원본 파이프라인. 프레임마다 VLM 1번으로 OCR+객체를 동시 추출하고
  RAG 유사도 임계값(0.5)으로 abnormal/normal 자동 판정하는 단순한 구조. 참고용으로 남겨둠, 실행 X.
  **새 작업은 `cybercop_pipeline_AdotX_v0_2.py`로.** 차이는 [CHANGELOG.md](./CHANGELOG.md) 참고.

```bash
python cybercop_pipeline_AdotX.py --url "https://www.youtube.com/shorts/영상ID"
python cybercop_pipeline_AdotX.py --file "data/eximg.png"
python cybercop_pipeline_AdotX.py --dir "C:\사이버 범죄 데이터\직거래 사기"
```

---

## 객체 인식 방식

- **`cybercop_pipeline_AdotX.py`(레거시)**: VLM이 자유롭게 출력한 객체명을 **임베딩 유사도(BGE-m3)**로
  `rag/allowed_objects.json`의 허용 클래스(439종)에 매핑. 프롬프트에 전체 클래스 목록을 안 넣어서 inference
  빠르고, JSON 파일만 편집하면 클래스 목록 관리 가능.
- **`cybercop_pipeline_AdotX_v0_1.py`/`v0_2.py`(현재)**: v0.1과 동일한 방식을 v0.2도 그대로 씀 (변경 없음).
  두 소스를 합쳐서 사용.
  1. Grounding DINO가 탐지한 사기 증거 라벨 중 신뢰도(`avg_score >= 0.55`) 높은 것만 (`SCAM_EVIDENCE_VOCAB`,
     `pipeline/vocab.py`의 36개 고정 어휘 — 좁고 정밀한 사기 판별용 어휘, `allowed_objects.json`과는 별개 목록)
  2. VLM 자유서술 폴백 결과를 레거시와 동일한 `ObjectMapper`(임베딩 유사도, threshold 0.7)로 `allowed_objects.json`
     (439종)에 정규화한 것
  둘을 합집합으로 합쳐서 최종 `objects`로 사용 (사기/검토필요 확정된 영상만 실행).

---

## Input 공통 정의

| 항목 | 형식 | 설명 |
|---|---|---|
| `--url` | URL 문자열 | YouTube / TikTok 단일 영상 |
| `--file` | 파일 경로 | 로컬 영상 또는 이미지 단일 파일 (모든 파이프라인 지원) |
| `--csv` | CSV 파일 경로 | `label`, `link`(또는 `url`) 컬럼 포함 |
| `--dir` | 디렉토리 경로 | 영상/이미지 파일이 담긴 로컬 폴더, 파일마다 독립 판정 (모든 파이프라인 지원) |
| `--seq_dir` | 디렉토리 경로 | 이어지는 이미지 시퀀스가 담긴 폴더, 폴더 전체를 한 건으로 판정 (`AdotX_v0_2.py`만) |

CSV 형식:
```csv
label,link
abnormal,https://www.youtube.com/shorts/xxxxx
normal,https://www.tiktok.com/@user/video/xxxxx
```

### 지원 범죄 유형 (26종)
직거래 사기, 쇼핑몰 사기, 게임 사기, 이메일 무역사기, 기타 사이버 사기,
피싱, 파밍, 스미싱, 메모리 해킹, 몸캠피싱, 메신저 이용사기, 기타 사이버 금융범죄,
개인·위치정보 침해, 사이버저작권 침해, 기타 정보통신망 이용범죄,
아동성 착취물, 불법 촬영물, 허위 영상물, 불법성 영상물, 기타 불법콘텐츠 범죄,
스포츠 토토, 경마·경륜·경정, 카지노, 기타 사이버 도박, 명예훼손, 모욕

---

## API 서버

CLI 파이프라인을 FastAPI로 래핑한 서버 세 개. **신규 연동은 `app/main_v0_2.py`.**

| 파일 | 기반 파이프라인 | 상태 |
|---|---|---|
| `app/main.py` | `cybercop_pipeline_AdotX.py` (레거시) | 레거시 |
| `app/main_v0_1.py` | `cybercop_pipeline_AdotX_v0_1.py` | 이전 버전 (이미지 시퀀스 엔드포인트 없음) |
| `app/main_v0_2.py` | `cybercop_pipeline_AdotX_v0_2.py` | **권장** (이미지 시퀀스 업로드 엔드포인트 포함) |

### 서버 실행

```bash
# 현재
uvicorn app.main_v0_2:app --host 0.0.0.0 --port 8000

# 이전 버전 / 레거시
uvicorn app.main_v0_1:app --host 0.0.0.0 --port 8000
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

> 기본 GPU는 0번. 다른 GPU를 쓰려면 실행 전 `export CUDA_VISIBLE_DEVICES=1` 등으로 지정 (0번이 다른 프로세스로
> 차 있을 때 우회용).

서버 시작 시 VLM 모델, Grounding DINO(레거시 제외), PaddleOCR, RAG 인덱스 자동 로드.

### 엔드포인트 (v0.1/v0.2 공통 + v0.2 신규)

| 엔드포인트 | 설명 |
|---|---|
| `POST /api/video` | YouTube / TikTok URL 분석 (`url`, `top_k` Form 파라미터) |
| `POST /api/video/upload` | 파일 업로드 분석 (`file`, `top_k`; 영상/이미지 둘 다 지원) |
| `POST /api/video/upload/sequence` | **v0.2 신규.** 이어지는 이미지 여러 장 업로드 → 한 건으로 판정. `files`(이미지 리스트) 또는 `archive`(폴더를 압축한 .zip, 파일명 순 정렬) 중 하나, `sequence_id`(결과 id, 안 주면 자동 생성), `top_k`. CLI `--seq_dir`과 동일 로직 |
| `POST /api/video/csv` | CSV 배치 분석 (`file`, `top_k`; CSV는 `link` 컬럼 필수) |

> **`/api/video/upload/sequence`를 Swagger UI(`/docs`)에서 테스트할 땐 `archive`(zip) 필드를 쓰세요.**
> `files`(배열)는 FastAPI가 생성하는 스펙을 이 Swagger UI 버전이 제대로 못 그려서 "파일 선택" 버튼 대신
> 이상한 문자열 입력칸으로 뜸 (API 자체는 정상 — Swagger UI 렌더링 한계). `files`로 여러 장을 올려야 하면
> Swagger UI 대신 curl을 쓰세요.

```bash
curl -X POST http://localhost:8000/api/video -F "url=https://www.youtube.com/shorts/영상ID"

# 이미지 시퀀스 (v0.2) — 낱장 여러 개 (curl 전용, Swagger UI에서는 안 됨 — 위 안내 참고)
curl -X POST http://localhost:8000/api/video/upload/sequence \
  -F "files=@data/img/kakao1/카톡_팀미션사기1.png" \
  -F "files=@data/img/kakao1/카톡_팀미션사기2.png" \
  -F "sequence_id=kakao1"

# 이미지 시퀀스 (v0.2) — 폴더를 zip으로 묶어서 한 번에 (Swagger UI/curl 둘 다 가능)
python3 -c "
import zipfile
from pathlib import Path
src = Path('data/img/kakao1')
with zipfile.ZipFile('/tmp/kakao1.zip', 'w', zipfile.ZIP_DEFLATED) as zf:
    for f in sorted(src.iterdir()):
        if f.is_file():
            zf.write(f, arcname=f.name)
"
curl -X POST http://localhost:8000/api/video/upload/sequence \
  -F "archive=@/tmp/kakao1.zip" \
  -F "sequence_id=kakao1"
```

> `zip` 명령이 설치 안 된 환경(예: 이 서버)이 있어서 위 예시는 파이썬 `zipfile` 모듈로 압축함. `zip -j` 명령이
> 설치돼 있으면 `zip -j /tmp/kakao1.zip data/img/kakao1/*`로도 동일하게 가능.

### 현재 버전(v0.2) API 응답 형식

**단건 (`/api/video`, `/api/video/upload`, `/api/video/upload/sequence`)**
```json
{
  "message": "success",
  "result": {
    "id": "영상ID",
    "title": "영상 제목",
    "label": "abnormal",
    "cls_label": "사기",
    "final_label": "사기",
    "classify_summary": "...",
    "grounding": { "evidence_score": 0.38, "top_labels": ["카카오톡 화면"] },
    "objects": ["휴대전화", "채팅창"],
    "ocr": [{ "start": "00:01", "end": "00:03", "text": "지금 투자하면 300% 수익 보장!" }],
    "ocr_after": "지금 투자하면 300% 수익 보장! | ...",
    "ocr_candidates": [],
    "rag": [{ "crime_type": "피싱", "similarity": 0.87 }]
  }
}
```

- `label`: 외부 연동 편의용 2단계 요약 (`final_label`이 "사기"/"검토필요"면 `abnormal`, "정상"이면 `normal`)
- `final_label`: 실제 3단계 판정 (`사기` / `정상` / `검토필요`) — evidence_score 기반 에스컬레이션 포함
- `final_label`이 정상이면 `objects`/`ocr`/`ocr_candidates`/`rag`는 빈 배열
- `/api/video/upload/sequence` 응답엔 `n_images`(업로드한 이미지 수)가 추가되고, `ocr_candidates`는 항상 빈 배열 (클러스터링 안 씀)

**배치 (`/api/video/csv`)**는 위 필드에 `gt_label`(CSV 입력 레이블), `error` 추가된 리스트 반환.

v0.1 API 응답은 필드명이 `ocr_text`(= v0.2의 `ocr_after`)로 다르고 `/api/video/upload/sequence`가 없음.
레거시 API 응답 형식은 `app/main.py` 코드 참고 (`ocr_frames`/`ocr_merged` 필드 구조가 다름).
