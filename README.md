# CyberCOP v0.4

영상·이미지·이미지 시퀀스에서 사이버 범죄 징후를 분석하는 GPU 추론 파이프라인입니다. 현재 운영 기준 코드는 **cybercop_pipeline_AdotX_v0_4.py**입니다. API 진입점(**app/main_v0_3.py**)은 아직 v0.3 기준이며 v0.4로 갱신되지 않았습니다 — 자세한 내용은 "알려진 제한" 참고.

v0.4는 자체 RAG(BGE-m3 임베딩 + rag/retrieval_docs_v2.json 기반 범죄유형 매칭)를 제거하고, 그 자리에 별도로 구축된 **risk_agent_package**의 위험도(text_risk, S1~S6 가이드라인)·사기유형 분류를 인라인으로 연결했습니다(`text_risk.py`). Grounding DINO와 PaddleOCR는 v0.3부터 이미 사용하지 않습니다. 기본 VLM인 Gemma 4가 분류·증거 확인·객체 확인을 담당하고, RapidOCR가 OCR을, risk_agent_package가 위험도·사기유형 분류를 담당합니다.

## 현재 모델과 구성

| 역할 | 기본 모델/엔진 | 코드 설정 |
|---|---|---|
| VLM | [google/gemma-4-26B-A4B-it](https://huggingface.co/google/gemma-4-26B-A4B-it), 26B(A4B MoE) BF16 | VLM_MODEL |
| 임베딩(객체 탐지용) | [BAAI/bge-m3](https://huggingface.co/BAAI/bge-m3) | EMBED_MODEL |
| OCR | RapidOCR 3.9.2, PP-OCRv5 Korean recognition, ONNX Runtime CPU | load_ocr() |
| 위험도·사기유형 분류 | risk_agent_package (text_risk_model, guideline_multitask_model, 규칙 기반 사기유형 분류기) | text_risk.assess_risk() |

기본 VLM은 다음 작업에 공통으로 사용됩니다.

1. 대표 프레임 묘사와 8개 사기 패턴 기반 사기/정상 분류
2. 36개 사기 증거 어휘의 closed-set 확인
3. 일반 객체 후보의 closed-set 확인
4. 이미지 시퀀스 OCR 문맥 교정

일반 객체는 **rag/allowed_objects.json**의 427개 클래스 중 BGE-m3로 상위 40개 후보를 찾은 뒤 VLM으로 실제 등장을 확인합니다(RAG 문서검색과는 무관 — v0.4에서도 이 임베딩 모델 자체는 계속 씁니다). 위험도·사기유형 분류는 classify_summary/objects/scam_evidence/ocr_after를 조합한 텍스트를 `text_risk.assess_risk()`를 통해 risk_agent_package(별도 경로, 기본값 `~/risk_agent_package(20260831)`, `RISK_AGENT_DIR` 환경변수로 변경 가능)로 넘겨 얻습니다.

## 처리 흐름

1. 영상은 장면 전환 기반 대표 프레임을 최대 4장 선택합니다. 이미지는 한 장을 그대로 사용하고, 이미지 시퀀스는 전체 구간에서 최대 4장을 고르게 선택합니다.
2. Gemma 4가 관찰 내용과 체크리스트 판정을 분리해 사기 또는 정상을 출력합니다. 형식 파싱에 실패하면 불명확으로 기록합니다.
3. Gemma 4가 별도 샘플 프레임에서 36개 사기 증거 어휘를 확인하고 evidence_score를 계산합니다.
4. 다음 경우 최종 판정을 검토필요로 바꿉니다.
   - 분류 결과가 불명확
   - 정상인데 evidence_score가 0.15 이상
   - 사기인데 evidence_score가 0.15 미만
5. 최종 판정이 정상이면 비용 절감을 위해 OCR·일반 객체·위험도/사기유형 분류를 건너뜁니다.
6. 사기 또는 검토필요이면 RapidOCR, 일반 객체 확인, risk_agent_package 기반 위험도·사기유형 분류(`text_risk.assess_risk()`)를 실행합니다.

risk_level은 코드에 정의된 휴리스틱 값이며 법률적 위험도나 형량 예측이 아닙니다. 자동 판정 결과 역시 수사·법률 판단을 대체하지 않습니다.

## 시스템 요구사항

- Python 3.11 또는 3.12
- Linux 권장. Windows는 NVIDIA CUDA 환경과 install.bat 사용 가능
- requirements.txt는 PyTorch 2.12.0 + CUDA 12.6 wheel 기준
- CUDA 12.6을 지원하는 NVIDIA 드라이버와 NVIDIA GPU
- Docker 배포 시 Docker Engine과 NVIDIA Container Toolkit 구성 필요
- 사용 가능한 VRAM 합계 64GB 이상 권장. `device_map="auto"`로 여러 GPU에 분산 가능
- 모델·Python 패키지를 위한 여유 디스크 80GB 이상 권장
- URL 입력 시 ffmpeg 필요

CPU 경로도 존재하지만 기본 26B(A4B MoE) BF16 VLM의 운영 추론용으로는 권장하지 않습니다. API 프로세스는 GPU 모델을 한 번만 적재하도록 worker 1개로 실행해야 합니다.

## 저장소 구조

~~~text
.
├── cybercop_pipeline_AdotX_v0_4.py  # 현재 CLI/핵심 파이프라인
├── text_risk.py                     # risk_agent_package 연결 (위험도·사기유형 분류)
├── app/
│   ├── main_v0_3.py                 # API 서버 (아직 v0.3 기준 — v0.4 미반영, "알려진 제한" 참고)
│   ├── main_v0_2.py                 # 이전 API
│   ├── main_v0_1.py                 # 이전 API
│   └── main.py                      # 레거시 API
├── pipeline/                        # 프레임 샘플링, 분류, 증거 집계, 어휘
├── rag/
│   ├── allowed_objects.json         # 일반 객체 427개 (v0.4에서도 사용)
│   └── retrieval_docs_v2.json       # RAG 문서 203개 (v0.3 이하 레거시용 — v0.4는 사용 안 함)
├── data/                            # 예제 입력과 CSV
├── requirements.txt                # 운영 의존성 고정
├── Dockerfile                       # NVIDIA GPU 배포 이미지 (아직 v0.3 기준)
~~~

v0.1~v0.3 코드와 changelog는 재현 및 비교용입니다. 신규 실행과 연동에는 v0.4만 사용하십시오.

## 설치

Linux:

~~~bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
~~~

requirements.txt 안에 CUDA 12.6 PyTorch index가 포함되어 있으므로 별도 index 옵션은 필요 없습니다.

Windows:

~~~bat
install.bat
~~~

첫 실행은 Gemma 4, BGE-m3와 RapidOCR ONNX 모델을 다운로드하므로 인터넷 연결과 충분한 디스크가 필요합니다. 완전히 다운로드한 후에만 HF_HUB_OFFLINE=1을 사용하십시오. 새 환경에서 HF_HUB_OFFLINE=1로 시작하면 모델 로드가 실패합니다.

환경변수 예시:

~~~bash
cp .env.example .env
set -a
source .env
set +a
~~~

주요 환경변수:

| 변수 | 기본값 | 설명 |
|---|---|---|
| VLM_MODEL | google/gemma-4-26B-A4B-it | AutoModelForMultimodalLM 호환 VLM ID 또는 로컬 경로 |
| VLM_GPU_RESERVE_GIB | 4 | GPU별로 추론용으로 남겨 둘 메모리(GiB) |
| EMBED_MODEL | BAAI/bge-m3 | SentenceTransformer 임베딩 모델 |
| MODEL_CACHE_DIR | 프로젝트/hf_models | VLM 로컬 저장 위치 |
| HF_HOME | Hugging Face 기본 캐시 | BGE-m3 등 Hub 캐시 위치 |
| HF_HUB_OFFLINE | 미설정/0 | 1이면 네트워크 없이 캐시만 사용 |
| HF_TRUST_REMOTE_CODE | 0 | 신뢰한 커스텀 모델이 필요할 때만 1 |
| CUDA_VISIBLE_DEVICES | 미설정 | 사용할 GPU 선택 |
| RISK_AGENT_DIR | ~/risk_agent_package(20260831) | risk_agent_package 설치 경로 (text_risk.py가 이 경로 아래 agents/를 참조) |

## CLI 실행

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

# CSV 배치
python cybercop_pipeline_AdotX_v0_4.py --csv data/labels.csv
~~~

CSV는 label과 link 또는 url 컬럼을 사용합니다.

~~~csv
label,url
abnormal,https://www.youtube.com/shorts/VIDEO_ID
normal,https://www.youtube.com/watch?v=VIDEO_ID
~~~

주요 CLI 옵션:

| 옵션 | 기본값 | 설명 |
|---|---:|---|
| --out_dir | ./output_AdotX_v0.3 | 결과와 다운로드 파일 저장 경로 (경로 이름은 레거시, v0.4에서도 그대로 사용) |
| --model | google/gemma-4-26B-A4B-it | VLM ID 또는 로컬 경로 |
| --scan_sec | 0.5 | 대표 프레임 탐색 간격 |
| --max_vlm_frames | 4 | 분류 대표 프레임 최대 수 |
| --evidence_sec | 3.0 | VLM 증거 탐지 샘플링 간격 |
| --max_evidence_frames | 15 | VLM 증거 탐지 최대 프레임 |
| --sample_sec | 2.0 | OCR 샘플링 간격 |
| --max_frames | 1000 | OCR 최대 프레임 |
| --escalate_thr | 0.15 | 검토필요 전환 증거 점수 |
| --victim_count | (미지정=0) | risk_assessment용 피해자 수 수동 override |
| --total_loss_won | (미지정 시 OCR에서 추출 시도) | risk_assessment용 총 피해금액(원) 수동 override |

이전 자동화와의 호환을 위해 --dino_sec와 --max_dino_frames는 각각 새 VLM 증거 옵션의 별칭으로만 남아 있습니다. Grounding DINO가 실행되는 것은 아닙니다. v0.3에 있던 --top_k(RAG 검색 결과 수)는 RAG 제거와 함께 삭제되었습니다.

결과는 output 디렉토리 아래 results/ocr_results_ID.json에 저장됩니다. 처리 실패가 한 건이라도 있으면 CLI는 종료 코드 1을 반환합니다.

## 결과 형식

정상과 비정상 모두 동일한 필드를 반환합니다. 정상은 skipped가 true이고 상세 분석 배열이 비어 있으며 risk_assessment 필드 자체가 없습니다(비용 절감을 위해 계산을 생략).

사기 또는 검토필요로 확정된 건은 v0.3의 `rag` 필드 대신 `risk_assessment` 필드가 들어갑니다 — 아래는 `--file data/eximg.png` 실제 실행 결과(2026-09-08)를 축약한 예시입니다.

~~~json
{
  "id": "eximg",
  "url": "data/eximg.png",
  "title": "eximg.png",
  "label": "manual",
  "cls_label": "정상",
  "final_label": "검토필요",
  "classify_summary": "관찰 내용 / 판단근거: ...",
  "grounding": {
    "labels": {},
    "evidence_score": 0.5,
    "n_detected_frames": 1,
    "top_labels": ["비트코인 화면"]
  },
  "objects": ["비트코인 화면", "이미지", "안내 문구", "텍스트"],
  "scam_evidence": ["비트코인 화면"],
  "ocr_before": ["..."],
  "ocr": [{"start": "00:00", "end": "00:00", "text": "..."}],
  "ocr_after": "FLIHANBIT UNION | 나의 숨겨진 채굴코인 찾기 | ...",
  "ocr_candidates": [],
  "skipped": false,
  "total_inference_time": 12.3,
  "risk_assessment": {
    "available": true,
    "statement_text": "...",
    "victim_count_used": 0,
    "total_loss_won_used": 0,
    "amount_source": "none",
    "text_risk": {"pred_label": "상", "probabilities": {"하": 0.0, "중": 0.0, "상": 1.0}},
    "guideline_risk": {"signal_based_level": "중", "score": 45.2},
    "crime_classification": {
      "top_crime_type": "사이버금융범죄",
      "top_crime_method": "사이버투자사기"
    }
  }
}
~~~

CLI의 label은 CSV ground truth 또는 manual입니다. API는 이를 input_label로 옮기고, label에 final_label 기반 normal 또는 abnormal 값을 제공합니다.

## API 실행

~~~bash
uvicorn app.main_v0_3:app --host 0.0.0.0 --port 8000 --workers 1
~~~

서버는 시작할 때 VLM, RapidOCR, BGE-m3와 FAISS 인덱스를 로드합니다. 모델 초기화가 실패하면 서버 시작도 실패합니다.

| 메서드/경로 | 설명 |
|---|---|
| GET /health/live | 프로세스 생존 확인 |
| GET /health/ready | 모델과 RAG 준비 확인. 미준비 시 503 |
| GET /api/info | 실제 모델, OCR, device 정보 |
| POST /api/video | 허용된 YouTube/TikTok HTTPS URL 분석 |
| POST /api/video/upload | 영상 또는 이미지 파일 분석 |
| POST /api/video/upload/sequence | 여러 이미지 또는 zip을 한 건으로 분석 |
| POST /api/video/csv | CSV 동기 배치 분석 |

예시:

~~~bash
curl http://127.0.0.1:8000/health/ready

curl -X POST http://127.0.0.1:8000/api/video/upload \
  -F "file=@data/eximg.png" \
  -F "top_k=3"

curl -X POST http://127.0.0.1:8000/api/video/upload/sequence \
  -F "files=@data/img/kakao1/카톡_팀미션사기1.png" \
  -F "files=@data/img/kakao1/카톡_팀미션사기2.png" \
  -F "sequence_id=kakao1"
~~~

CYBERCOP_API_KEY를 설정하면 모든 /api 엔드포인트에 X-API-Key 헤더가 필요합니다. health 엔드포인트는 인증 대상이 아닙니다.

API는 기본적으로 다음 운영 제한을 적용합니다.

- GPU 분석 동시성 1
- 단일 업로드 250MiB
- zip 업로드 100MiB, 압축 해제 이미지 전체 200MiB
- 이미지 시퀀스 최대 50장
- CSV 최대 100행
- URL은 HTTPS 및 youtube.com, youtu.be, tiktok.com 계열만 허용
- 임시 업로드와 다운로드는 요청 종료 후 삭제
- 내부 예외는 서버 로그에만 남고 응답에는 request_id만 노출

제한값은 .env.example의 환경변수로 조정할 수 있습니다. 긴 CSV 작업은 동기 요청이므로 대규모 운영에서는 별도 작업 큐와 결과 저장소를 앞단에 두는 것을 권장합니다.

## Docker GPU 배포

호스트에 NVIDIA Container Toolkit을 설치한 뒤 Docker runtime을 구성해야 합니다. 설치 방법은 배포 OS에 맞는 NVIDIA 공식 문서를 따르십시오.

~~~bash
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker
~~~

~~~bash
docker build -t cybercop:0.3 .

docker volume create cybercop-models

docker run --rm --gpus all \
  -p 8000:8000 \
  -v cybercop-models:/models/huggingface \
  -e CYBERCOP_API_KEY="change-me" \
  cybercop:0.3
~~~

Gemma 4와 BGE-m3 캐시를 volume에 유지해야 재시작 때 다시 다운로드하지 않습니다. RapidOCR ONNX 모델은 이미지 빌드 중 포함됩니다. 한 GPU에 여러 worker를 띄우면 각 프로세스가 모델을 중복 적재하므로 CMD의 workers 1 설정을 유지하십시오. 수평 확장은 GPU별 컨테이너 복제로 구성합니다.

## 검증

~~~bash
python -m compileall -q cybercop_pipeline_AdotX_v0_4.py text_risk.py pipeline
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
python cybercop_pipeline_AdotX_v0_4.py --file data/eximg.png --out_dir /tmp/cybercop-smoke
~~~

자동 테스트(tests/, pytest)는 아직 v0.4/text_risk.py용으로 작성되지 않아 저장소에서 제외했습니다.

모델을 적재하지 않고 API 라우팅과 상태 응답만 확인하려면:

~~~bash
LOAD_MODELS_ON_STARTUP=0 uvicorn app.main_v0_3:app --host 127.0.0.1 --port 8000
~~~

이 모드에서 /health/live는 200, /health/ready와 분석 엔드포인트는 503이 정상입니다.

## 배포 전 확인사항

- CYBERCOP_API_KEY 또는 외부 API Gateway 인증 설정
- TLS는 reverse proxy/load balancer에서 종료
- hf_models volume의 용량·권한·백업 정책 확인
- workers 1 및 MAX_CONCURRENT_ANALYSES 1 유지
- startup/readiness probe는 /health/ready, liveness probe는 /health/live 사용
- 업로드 크기 제한을 reverse proxy에도 동일하게 설정
- 로그에서 request_id로 오류 추적
- 실제 운영 데이터로 오탐·미탐 임계값 재평가

## 알려진 제한

- TikTok은 봇 차단 정책에 따라 yt-dlp 다운로드가 실패할 수 있습니다.
- 현재 분류 기준은 8개 사기 패턴 중심이므로 모든 사이버 범죄를 완전하게 판별하지 않습니다.
- VLM 출력 파싱 실패는 안전 측으로 검토필요 처리됩니다.
- OCR은 ONNX Runtime CPU로 실행됩니다.
- 코드에 포함된 휴리스틱 risk_level은 법률 판단에 사용할 수 없습니다.
- **API 서버(app/main_v0_3.py)와 Dockerfile은 아직 v0.4로 갱신되지 않았습니다.** 지금은 CLI(cybercop_pipeline_AdotX_v0_4.py)만 risk_assessment를 포함합니다.
- **risk_agent의 사기유형 분류는 규칙(키워드/별칭) 기반이며 커버리지가 제한적입니다.** 현재 사기유형×수법 조합 9개만 규칙이 등록되어 있고(risk_agent_package의 통제 어휘는 15종×25종), 해당하지 않는 유형은 "미분류"로 나올 수 있습니다.
- risk_assessment의 victim_count/total_loss_won은 Cybercop 출력만으로는 알 수 없어 기본값 0(정보없음)이며, ocr_after에서 금액을 best-effort로 추출하는 것 외에는 --victim_count/--total_loss_won 수동 입력에 의존합니다.

## 라이선스

프로젝트 코드는 LICENSE를 따릅니다. Gemma 4와 BGE-m3 등 모델은 각 모델 배포 페이지의 라이선스를 별도로 확인하십시오.

---
최종 업데이트: 2026-09-08 (v0.4)
