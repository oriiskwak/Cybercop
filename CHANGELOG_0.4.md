# CHANGELOG — cybercop_pipeline_AdotX_v0_3.py → cybercop_pipeline_AdotX_v0_4.py

`cybercop_pipeline_AdotX_v0_3.py`는 그대로 두고, 아래 변경사항을 반영한 `cybercop_pipeline_AdotX_v0_4.py`를
새 파일로 만듦 (기존 버전 파일은 건드리지 않고 다음 버전 파일을 새로 만드는 방식 유지).

v0.4의 핵심은 **자체 RAG(범죄유형 매칭)를 제거하고, 그 역할을 이미 구축된 별도 파이프라인
`risk_agent_package`의 위험도·사기유형 분류로 대체**한 것 하나임. 나머지는 그 여파로 따라온 정리 작업.

## 1. 자체 RAG(BGE-m3 + retrieval_docs_v2.json 범죄유형 매칭) 제거

- **문제**: v0.3까지는 OCR 텍스트와 객체 목록을 쿼리로 만들어 `rag/retrieval_docs_v2.json`(203개 문서,
  15개 범죄유형)에서 임베딩 유사도 검색을 하고 그 결과를 `rag` 필드에 담았음. 그런데 이 "범죄유형 분류"와
  "위험도 산출"은 이미 `risk_agent_package`에 별도로 구축·평가까지 끝난 파이프라인이 있어서, 같은 기능을
  두 곳에서 따로 유지할 이유가 없었음.
- **변경**: `CrimeRAG` 클래스와 `CRIME_RISK_MAP` 상수를 삭제하고, `rag.search()` 호출 및 결과 JSON의
  `rag` 필드를 제거. `DEFAULT_DOCS_PATH`(`RAG_DOCS_PATH` env var)도 함께 삭제.
  - `rag/retrieval_docs_v2.json` 파일 자체는 남겨둠 — v0.3 이하가 계속 참조하므로.
  - `faiss` import 제거 (RAG 검색에만 쓰였음).
  - **`ObjectMapper`는 그대로 유지**. 일반객체 427종 closed-set 탐지에 BGE-m3 임베딩이 계속 필요한데,
    v0.3에서는 `CrimeRAG`가 로드한 SentenceTransformer 인스턴스를 재사용하는 구조였음. v0.4에서는
    `CrimeRAG` 없이 `SentenceTransformer(DEFAULT_EMBED_MODEL, device=DEVICE)`를 직접 로드해 넘김.
  - `--top_k`(RAG 검색 결과 수) CLI 인자 삭제.
  - 로그 메시지 `[2/3] OCR + RAG 로드...` → `[2/3] OCR + 임베딩 로드...`

## 2. risk_agent_package 연결 — `text_risk.py` 신규 추가

- 예전 `rag.search()`가 있던 자리에 `text_risk.assess_risk()` 호출을 인라인으로 넣어, **한 번 실행으로
  영상분석 → 위험도 → 사기유형까지 한 결과 JSON에 담기도록** 함. 별도 후처리 스크립트를 돌릴 필요 없음.
- `text_risk.py`가 하는 일:
  1. Cybercop 결과(`classify_summary` / `objects` / `scam_evidence` / `ocr_after`)를 조합해
     risk_agent 모델 입력용 진술 텍스트를 구성
  2. `predict_text_risk()` — 텍스트 전용 3-class 위험도(하/중/상)
  3. `predict_guideline_multitask_risk()` — S1~S6 가이드라인 결합 위험도
  4. `classify_crime_from_metagraph()` — 사기유형·수법 분류 (규칙 기반 키워드 매칭)
- **risk_agent 모듈은 지연 로딩(lazy import)** — `assess_risk()` 최초 호출 시점에만 sys.path를 조작해
  import하므로, `from text_risk import assess_risk`만으로는 무거운 모델 로드가 트리거되지 않음.
- risk_agent 예측 함수는 내부에서 `CUDA_VISIBLE_DEVICES=""`를 강제하는 CPU 전용이라 VLM의 GPU 사용과
  간섭하지 않음. 모델은 lazy singleton으로 캐싱되어 배치 처리 시 최초 1건에서만 로드됨.

## 3. 결과 JSON: `rag` → `risk_assessment` 필드 교체

- v0.3의 `rag`(범죄유형 검색 결과 리스트) 필드가 사라지고, 대신 `risk_assessment` 객체가 들어감:
  ```
  risk_assessment: {
    available, statement_text, victim_count_used, total_loss_won_used, amount_source,
    text_risk:            { pred_label(하/중/상), probabilities, ... },
    guideline_risk:       { signal_based_level(하/중/상), score, indicator_scores(S1~S6), ... },
    crime_classification: { top_crime_type, top_crime_method, crime_types, crime_methods }
  }
  ```
- **실행 조건은 v0.3의 RAG와 동일** — `final_label`이 "사기" 또는 "검토필요"일 때만 계산하고,
  "정상"이면 필드 자체가 없음(비용 절감).
- `print_report()`의 콘솔 출력에서 "범죄 유형"을 RAG 결과가 아니라
  `risk_assessment.crime_classification.top_crime_type`에서 가져오도록 변경.

## 4. 피해자 수·피해금액(S1·S2) 처리 방침

- risk_agent의 `guideline_risk`는 S1(피해금액)·S2(피해자 수)를 원시 숫자로 받는데, **영상 단독 분석으로는
  알 수 없는 정보**임(진정서에서 나오는 값). 억지로 추정해 채우지 않고 기본값 0으로 두는 방침을 택함.
- `--victim_count`, `--total_loss_won` CLI 인자를 추가해 필요 시 수동 입력 가능.
- 금액은 미지정 시 `ocr_after`에서 금액 패턴을 정규식으로 찾아 best-effort 추출(`amount_source` 필드에
  `manual` / `ocr_extracted` / `none` 중 하나로 출처 기록). 실패해도 0으로 두고 파이프라인은 정상 진행.

## 5. 진술 텍스트 구성에서 제외/정리한 것

- **`cls_label`/`final_label` 제외**: `assess_risk()`는 애초에 판정이 "사기"/"검토필요"일 때만 호출되므로
  이 값은 항상 같은 신호("의심됨")만 반복해 정보량이 없고, S3~S6 예측이 실제 증거 대신 판정 문구 자체에
  끌려 편향될 위험이 있어 제외.
- **`objects` ∩ `scam_evidence` 중복 제거**: Cybercop의 `objects` 필드는 파이프라인 내부에서 이미
  `사기증거 top_labels ∪ 일반객체`로 합쳐진 값이라, `scam_evidence`를 별도로 넘기면 같은 단어가
  진술 텍스트/hints에 두 번 들어가 가중치가 실질적으로 두 배가 됨. `_general_objects()`로 겹치는 항목을
  빼고 순수 일반객체만 넘기도록 정리.

## 6. 의존성 추가

- `sse-starlette==3.4.11` — `text_risk.py`가 risk_agent의 `risk_api.py`에서
  `classify_crime_from_metagraph`를 import하는 데 필요(해당 파일이 최상단에서 sse_starlette를 사용).
  `risk_api.py`는 import만 하고 서버는 띄우지 않음(`uvicorn.run()`은 `__main__` 가드 안에 있음).

## 실측 결과 (로컬 영상·이미지 31건)

| 항목 | 결과 |
|---|---|
| 사기/정상 1차 판정 | Recall 100%, Precision 87.5%, F1 93.3% (GT 25건 기준) |
| 사기유형 분류 성공률 | 19/22건 (86.4%), 미분류 3건 |
| `text_risk` 등급 분포 | 하 9 / 중 9 / 상 4 |
| `guideline_risk` 등급 분포 | 하 10 / 중 12 / **상 0** |

## 남은 정리 대상 / 확인 필요

- **`guideline_risk`가 구조적으로 "상"에 도달할 수 없음.** 가중치의 60%가 S1·S2에 걸려 있는데 영상
  단독 분석에서는 두 값이 0이라, S5/S6이 "확인"(100)으로 잡히는 예외 경로 외에는 "중"을 넘지 못함.
  실측 22건에서 "상" 0건. 현재로서는 `text_risk`가 더 신뢰 가능한 지표.
- **risk_agent의 유형-수법 짝이 깨질 수 있음.** `risk_api.py`의 `_aggregate()`가 범죄유형과 범죄수법을
  독립적으로 합산해 각각 1위를 뽑기 때문에, "사이버금융범죄 / 쇼핑몰 사기"처럼 통제어휘상 존재하지 않는
  조합이 출력됨(실측 19건 중 11건). 유형은 최대 5개 규칙이 합산되어 유리하고 수법은 규칙 단위로 경쟁하는
  구조 차이 때문.
- **사기유형 분류의 커버리지 한계.** 규칙(`RISK_CRIME_RULES`)이 3개 대분류 × 8개 수법(9개 조합, 키워드
  110개)만 등록되어 있어, 통제어휘 전체(15개 대분류 × 26개 수법) 대비 커버리지가 좁음. 학습 기반이 아닌
  키워드 매칭이라 데이터를 늘려도 개선되지 않고 규칙을 직접 추가해야 함.
- **다의어 오매칭.** "선물"(금융상품 선물 ↔ 선물세트), "주문"(주식 주문 ↔ 쇼핑 주문) 등이 다른 수법
  규칙에 걸려 오분류를 유발함.
- **`RISK_CRIME_RULES`와 통제어휘의 매핑 불일치.** 사이버투자사기가 `RISK_CRIME_RULES`에서는
  "사이버금융범죄", `provo_metadata.CRIME_METHOD_TO_TYPE`에서는 "사이버 사기"로 등록되어 있음.
- **`--out_dir` 기본값이 `./output_AdotX_v0.3`으로 남아 있음.** v0.4에서도 그대로 쓰이나 이름이 레거시.
- **API 서버(`app/main_v0_3.py`)와 Dockerfile은 v0.4 미반영.** 현재 CLI만 `risk_assessment`를 포함함.
- **OCR 인식 정확도.** RapidOCR의 한국어 인식 모델은 mobile 버전만 존재하고(server 버전은 중국어만 제공)
  검출 모델만 server로 교체 가능. 프레임 반복으로 인한 유사 텍스트 중복도 남아 있음.
