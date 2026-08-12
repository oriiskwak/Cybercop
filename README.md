# Video OCR & RAG 기반 사이버 범죄 탐지 파이프라인

YouTube/TikTok 영상, 로컬 영상/이미지, 이어지는 이미지 시퀀스를 입력받아 VLM(SKT A.X-4.0-VL-Light)으로
사기/정상 여부를 1차 판단(CoT)하고, Grounding DINO로 사기 특화 객체가 탐지될 시 재판정 실시.
사기(또는 검토필요)로 판정된 경우에만 PaddleOCR로 텍스트를 추출하고, RAG(FAISS + BGE-m3)로
범죄 유형을 매칭해 위험도를 산출한다.


## 주의사항
TikTok 링크를 넣었을 때 다운로드 단계에서 문제가 발생 → 최근 틱톡에서 크롤링 프로그램을 이전보다 강하게 차단하고 있어서, TikTok 영상 페이지에 접근하려 하면 사람이 아니라 로봇으로 판단해 접근을 막고 있는 오류 발생. 
YouTube 영상이나 로컬 파일은 현재 문제와 무관하게 동작,
현재 우회 방법은 검토 중.

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
│   ├── main.py                      # 원본 파이프라인 FastAPI 서버 
│   ├── main_v0_1.py                 # FastAPI 서버 (이전 버전, AdotX_v0_1.py 기준 — --seq_dir 없음)
│   └── main_v0_2.py                 # FastAPI 서버 (권장, AdotX_v0_2.py 기준 — --seq_dir/이미지 시퀀스 업로드 포함)
├── data/
│   ├── labels.csv                   # 테스트용 영상 URL 목록 (label,url) — --csv 기본값
│   ├── thecheat.csv                 # 테스트용 영상 URL 목록 (추가 세트) — 기본값 아님, --csv로 직접 지정해야 씀
│   ├── img/                         # --seq_dir 테스트용 이미지 시퀀스 폴더
│   │   └── kakao1/                  # 카톡 대화 스크린샷 예시 (6장)
├── rag/
│   ├── allowed_objects.json         # 허용 객체 클래스 목록 (439종, JSON DB) 
│   └── retrieval_docs_v2.json          # 범죄 유형 (15종 + 객체 키워드) 


## 파이프라인 버전 안내

| 파일 | 상태 | 분류 방식
|---|---|---|---|
| `cybercop_pipeline_AdotX.py` | 이전 버전 | 단일 VLM 호출로 OCR+객체 동시 추출, RAG 유사도 임계값으로 abnormal/normal 자동 판정 
| `cybercop_pipeline_AdotX_v0_1.py` | 이전 버전 | 사기/정상 CoT 분류 + Grounding DINO 증거탐지 + OCR 프레임간 클러스터링(IoU+CER) 
| `cybercop_pipeline_AdotX_v0_2.py` | **현재 사용** | v0.1과 동일 + 이미지 시퀀스 입력 지원 

`AdotX.py` → `AdotX_v0_1.py` 변경 내역은 **[CHANGELOG.md](./CHANGELOG.md)**, `AdotX_v0_1.py` → `AdotX_v0_2.py`
변경 내역은 **[CHANGELOG_0.2.md](./CHANGELOG_0.2.md)** 참고. 

## 설치

```bash
pip install -r requirements.txt --index-url https://download.pytorch.org/whl/cu124 --extra-index-url https://pypi.org/simple
```

> Linux GPU 서버: `requirements.txt`의 `faiss-cpu` → `faiss-gpu`로 바꿔서 설치

---

## 실행 방법 (`cybercop_pipeline_AdotX_v0_2.py`)

```bash
python cybercop_pipeline_AdotX_v0_2.py --csv "data/thecheat.csv"
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

이어지는 이미지 시퀀스(카톡 대화 스크린샷 등, 폴더 전체를 한 건으로 판정) — 
```bash
python cybercop_pipeline_AdotX_v0_2.py --seq_dir "data/img/kakao1"
```

> 입력값은 `--url` / `--file` / `--dir` / `--seq_dir` / `--csv` 

### 옵션
| 인자 | 기본값 | 설명 |
|---|---|---|
| `--csv` | `data/labels.csv` | `label`/`link`(또는 `url`) 컬럼을 가진 CSV |
| `--seq_dir` | — | 연속적인 이미지 시퀀스가 담긴 디렉토리 경로 |
| `--out_dir` | `./output_AdotX_v0.2` | 결과 저장 경로 |
| `--model` | `skt/A.X-4.0-VL-Light` | 초기 사기/정상 분류/ OCR-객체 추출용 VLM |
| `--gdino_model` | `IDEA-Research/grounding-dino-tiny` | 범죄 특화 객체 추출용 Grounding DINO |
| `--scan_sec` | `0.5` | 분류용 키프레임 스캔 간격 (장면전환 기반 최대 4프레임 선정, 영상만 해당) |
| `--max_vlm_frames` | `4` | 분류에 사용할 대표 프레임 수 (초기 사기/정상 분류) |
| `--dino_sec` | `3.0` | DINO 프레임 간격 (영상만 해당) |
| `--max_dino_frames` | `15` | DINO에 넘길 최대 프레임 수 (영상만 해당) |
| `--sample_sec` | `2.0` | 사기/검토필요 판정 시 OCR용 균등 샘플링 간격 (영상만 해당) |
| `--max_frames` | `1000` | OCR용 최대 프레임 수 (영상만 해당) |
| `--escalate_thr` | `0.15` | evidence_score가 이 값 이상이면 정상→검토필요로 에스컬레이션 |
| `--top_k` | `3` | RAG 검색 상위 범죄 유형 |

### 처리 파이프라인

```
1단계 초기 사기 정상 CoT 분류 → 2단계 Grounding DINO  범죄 특화 객체 추출  → 에스컬레이션 판정
   → ( 1단계 결과 사기/검토필요 시) OCR 프레임간 클러스터링 + 객체 리스트 → RAG 검색(범죄 유형 추출)
```
1단계 (초기 사기/정상 CoT 분류): 영상/이미지에서 대표 프레임 최대 4장을 뽑아 VLM에 입력.
①화면에 보이는 내용을 판단 없이 서술 → ②그 서술을 체크리스트(8개 사기 패턴) 기준으로
사기/정상 판단. 

2단계 (Grounding DINO 범죄 특화 객체 추출 → 에스컬레이션 판정): 모든 영상에서 실행.
계좌 화면, 카카오톡/텔레그램 화면, 신분증, QR코드 등 36종 사기 증거 어휘를 화면에서
직접 탐지(zero-shot). 1단계가 "정상"으로 판단했어도 evidence_score가
기준치 이상(현재 0.15)이면 "검토필요"로 재분류 

(1단계 결과 사기/검토필요 시) OCR 프레임간 클러스터링 + 객체 리스트: "정상" 확정 영상은
OCR과 객체 탐지생략. PaddleOCR로 화면 텍스트 추출 후, 여러 프레임에 걸쳐 반복
등장하는 같은 텍스트를 위치+문자열 유사도로 묶어서 OCR 오탈자를 다수결로 정리.
객체는 DINO 탐지 결과 + VLM이 자유롭게 나열한 객체 목록을 합쳐서 최대한 많은 객체리스트를 추출.

RAG 검색 (범죄 유형 추출): 정리된 OCR 텍스트 + 객체 목록을 하나의 쿼리로 만들어
범죄 유형 문서 DB(15개 대분류)에서 임베딩 유사도로 가장 가까운 유형을 검색·매칭.


## Input 

| 항목 | 형식 | 설명 |
|---|---|---|
| `--url` | URL 문자열 | YouTube / TikTok 단일 영상 |
| `--file` | 파일 경로 | 로컬 영상 또는 이미지 단일 파일 (모든 파이프라인 지원) |
| `--csv` | CSV 파일 경로 | `label`, `link`(또는 `url`) 컬럼 포함 |
| `--dir` | 디렉토리 경로 | 영상/이미지 파일이 담긴 로컬 폴더, 파일마다 독립 판정 (모든 파이프라인 지원) |
| `--seq_dir` | 디렉토리 경로 | 이어지는 이미지 시퀀스가 담긴 폴더, 폴더 전체를 한 건으로 판정 |

CSV 형식:
```csv
label,link
abnormal,https://www.youtube.com/shorts/xxxxx
normal,https://www.tiktok.com/@user/video/xxxxx
```

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
  "rag": [{ "crime_type": "사이버금융범죄", "similarity": 0.87, "risk_level": 0.22 }],
  "total_inference_time": 32.69,
}
```

`final_label`이 정상이면 `skipped: true`, `objects`/`ocr`/`ocr_after`/`rag`는 빈 값.
`--seq_dir`(이미지 시퀀스) 입력은 `ocr_candidates`가 항상 빈 배열 (클러스터링을 안 쓰므로).




### 지원 범죄 유형 (15종)
사이버사기, 사이버 금융범죄, 개인·위치정보 침해, 사이버 저작권 침해, 사이버스팸메일,
기타 정보통신망 이용 범죄, 사이버성폭력, 사이버도박, 사이버 명예훼손·모욕, 사이버스토킹,
기타 불법 콘텐츠 범죄, 해킹, 서비스거부공격(DDoS), 악성프로그램, 기타 정보통신망 침해형 범죄

---

## API 서버

CLI 파이프라인을 FastAPI로 래핑한 서버 세 개. **신규 연동은 `app/main_v0_2.py`.**

| 파일 | 기반 파이프라인 | 상태 |
|---|---|---|
| `app/main.py` | `cybercop_pipeline_AdotX.py`  | 레거시 |
| `app/main_v0_1.py` | `cybercop_pipeline_AdotX_v0_1.py` | 이전 버전  |
| `app/main_v0_2.py` | `cybercop_pipeline_AdotX_v0_2.py` | **최신 버전**  |

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
