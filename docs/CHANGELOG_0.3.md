# CHANGELOG — cybercop_pipeline_AdotX_v0_2.py → cybercop_pipeline_AdotX_v0_3.py

`cybercop_pipeline_AdotX_v0_2.py`는 그대로 두고, 아래 변경사항을 반영한 `cybercop_pipeline_AdotX_v0_3.py`를
새 파일로 만듦 (앞으로도 뭔가 바뀌면 기존 버전 파일은 건드리지 않고 다음 버전 파일을 새로 만드는 방식으로 진행).

v0.3의 핵심은 **별도 탐지 모델(Grounding DINO)을 걷어내고 증거·객체 탐지를 전부 VLM closed-set으로 통일**한 것,
그리고 **OCR 엔진을 PaddleOCR에서 RapidOCR로 교체**한 것 두 가지임. 나머지는 그 여파로 따라온 정리 작업.

## 1. 사기증거 탐지를 Grounding DINO → VLM closed-set으로 교체

- **문제**: Grounding DINO는 영어 vocab 구문으로 open-vocabulary 탐지를 하는데, `SCAM_EVIDENCE_VOCAB`의
  대상이 "주식 차트", "카카오톡 화면"처럼 **물리적 사물이 아니라 화면/UI 유형**이라 DINO의 강점(사물 박스 탐지)과
  안 맞았음. 평가 결과 **F1 0%** — 사실상 아무것도 못 맞춤. 라벨 한국어 역매핑(v0.2 #9)까지 붙여가며 보정했지만
  근본적으로 태스크가 안 맞는 문제였음. 게다가 DINO 모델을 따로 로딩하느라 GPU 메모리도 계속 점유했음.
- **변경**: `pipeline.grounding_detector.GroundingDetector` / `evidence_aggregator.aggregate()` 사용을 제거하고,
  신규 함수 `vlm_scam_evidence_batch()`로 대체. 프레임마다 VLM에게 **어휘 목록 전체를 프롬프트에 그대로 보여주고
  "이 중에 실제로 보이는 것만 골라라"** 는 closed-set 방식으로 확인함.
  - `SCAM_EVIDENCE_VOCAB`은 `{한국어: 영어}` dict인데, v0.3에서는 **한국어 키만** `SCAM_EVIDENCE_LABELS`로 뽑아
    목록으로 사용. 영어 값은 DINO 전용이었으므로 더 이상 안 씀 (vocab 파일 자체는 그대로 둠).
  - 어휘 수가 적어서 **청크 분할이나 retrieval 없이 한 번에 다 넣어도 할루시네이션이 안 남**. 평가로 확인.
  - 파싱은 `_parse_bracket_list()` 공용 함수로 처리 — `[EVIDENCE] a, b` 형식에서 목록을 뽑되,
    **원본 목록에 없는 단어는 버림**(closed-set 강제). 프롬프트로만 막지 않고 코드에서 한 번 더 거름.
  - **평가 결과: F1 0% → 58.8%**
  - `GDINO_MODEL_ID` 상수와 `--gdino_model` 인자 삭제, `main()`의 로딩 단계도 `[1/4]~[4/4]` → `[1/3]~[3/3]`으로 축소.

## 2. 일반객체 탐지를 "자유서술 → 임베딩 매핑" → "묘사 → RAG retrieval → closed-set 검증" 3단계로 교체

- **문제**: v0.2의 `vlm_object_fallback()`은 VLM에게 자유롭게 사물 이름을 나열시킨 뒤(`[OBJECTS] ...`),
  그 결과를 `ObjectMapper`(BGE-m3 임베딩 유사도)로 `allowed_objects.json`(427종)에 사후 매핑하는 방식이었음.
  VLM이 뱉은 임의의 단어를 억지로 허용 목록에 끼워맞추는 구조라 정확도가 낮았음 (**F1 23.6%**).
- **변경**: 427종 목록을 VLM에게 직접 주되, **후보를 미리 줄여서** 주는 방식으로 바꿈.
  1. `DESCRIBE_PROMPT`로 프레임을 한두 문장으로 **묘사**시킴 (객체 나열이 아니라 장면 서술)
  2. 그 묘사문을 BGE-m3로 임베딩해 427종과 유사도를 계산, **top-K(기본 40개) 후보만 retrieval**
  3. 그 40개만 프롬프트에 넣어 closed-set으로 확인 (`_general_object_verify_prompt()`)
  - **427개를 한 번에/청크로 다 주는 방식은 실패했음** — 모델이 목록을 그대로 반복 생성하는 할루시네이션이
    발생 (평가 데이터 25개 중 다수에서 확인, F1 29.5%에 그침). 그래서 retrieval로 후보를 줄이는 방식을 채택.
  - **평가 결과: F1 23.6% → 36.5%.** 프레임당 VLM 호출이 2회(묘사 + 검증)로 늘지만, 청크 방식 대비
    호출 수는 오히려 적어서 자유서술 방식과 비슷한 수준으로 유지됨.
  - `OBJECT_RAG_TOP_K = 40` — 40과 60의 F1이 거의 동일(36.5% vs 36.0%)해서 프롬프트가 더 짧은 40 채택.
  - **새 모델 로딩 없음**: 이미 `ObjectMapper`가 들고 있는 BGE-m3와 427종 임베딩(`obj_mapper.model`,
    `obj_mapper.embeddings`, `obj_mapper.allowed`)을 그대로 재사용.
  - 함수명 `vlm_object_fallback()`은 호출부 호환을 위해 유지했지만 **더 이상 "폴백"이 아니라 상시 실행되는
    메인 경로**임. 인자에 `obj_mapper`가 추가됨.
  - `ObjectMapper.map()`(사후 임베딩 매핑)과 `_analyze_one_frame_vlm()`, `DEFAULT_PROMPT`는 호출부가 없어져
    사실상 미사용 상태가 됨 (`ObjectMapper` 클래스 자체는 임베딩 보관용으로 계속 필요).

## 3. OCR 엔진 교체: PaddleOCR → RapidOCR (ONNXRuntime)

- **문제**: PaddleOCR은 (1) GPU 사용 시 이 환경의 cuDNN을 못 찾아 `PreconditionNotMet: cudnn_dso_handle`로
  매 요청 500 에러가 나서 결국 CPU로 되돌려 쓰고 있었고, (2) PaddleX가 이미 캐시된 모델도 매번 온라인으로
  재확인하려다 네트워크가 죽으면 수십 분씩 멎는 문제가 있어 `HF_HUB_OFFLINE=1`을 강제해야 했음.
  paddlepaddle/paddlex 의존성 스택 자체가 무거웠음.
- **변경**: `load_paddleocr()` → `load_ocr()`로 교체, RapidOCR(ONNXRuntime 백엔드) 사용.
  - 검출은 `PP-OCRv5` mobile det(언어 무관 단일 모델), 인식은 `korean` PP-OCRv5 mobile rec.
  - 스레드 수 명시(`intra_op=4`, `inter_op=1`), `use_cuda: False`로 CPU 고정.
  - `run_paddleocr_on_frame()` → `run_ocr_on_frame()`으로 이름/구현 교체. 반환 스키마를 `[[poly, (text, score)], ...]`
    (Paddle 원본 형식)에서 **`{text, score, bbox}` 딕셔너리 리스트**로 통일.
  - 그에 맞춰 `_merge_same_line()`도 딕셔너리 입출력으로 재작성. **`_poly_bounds()` 삭제** — bbox를
    `run_ocr_on_frame()` 안에서 폴리곤 → 축정렬 사각형으로 바로 변환하므로 별도 헬퍼가 불필요해짐.
  - 타입 힌트 `PaddleOCR` → `RapidOCR`로 전부 교체 (`run_ocr`, `_ocr_worker`, `_analyze_and_save`,
    `process_single_video`, `process_local_file`, `process_image_sequence`).

## 4. 영상 OCR의 VLM 후처리(`postprocess_ocr_with_vlm`) 제거

- **문제**: 영상 전체의 OCR 스팬을 한 번에 VLM에 넣어 교정/병합시키는 단계에서 **건당 30~40분**이 소요됨.
  v0.2 #3에서 이미 이미지 시퀀스에 대해서는 이 함수가 타임스탬프를 망가뜨리는 문제로 우회 경로를 만들었는데,
  영상 경로에서도 비용 대비 효용이 맞지 않았음.
- **변경**: `postprocess_ocr_with_vlm()` 함수 자체를 삭제. 영상 경로는
  `corrected_ocr = detail["ocr_spans"]` — **클러스터링 결과를 그대로 사용**함.
  - 이미지 시퀀스 경로(`is_sequence=True`)의 `correct_ocr_text_with_vlm()`(이미지별 개별 교정)은 **그대로 유지**.
    실사용 리포트에 OCR 텍스트가 필요하고, 이쪽은 이미지 수가 적어 비용이 감당 가능하므로.
  - 결과 JSON의 `ocr` / `ocr_after` 필드는 그대로 있으나, **영상 입력에서는 이제 "교정 후"가 아니라
    "클러스터링 후 다수결 원문"** 이라는 점에 유의 (`ocr_before`와의 대칭은 이름만 남음).

## 5. OCR 클러스터링 IoU 임계값 상향: 0.3 → 0.7

- **문제**: `IOU_THRESHOLD = 0.3`은 화면상 위치가 조금만 겹쳐도 서로 다른 텍스트를 같은 클러스터로 묶었음.
- **변경**: `0.7`로 상향해서 **거의 같은 자리에 있는 검출만** 묶이도록 조임.
  - 4번에서 VLM 후처리가 빠졌기 때문에 클러스터링 결과가 곧 최종 출력이 됨 → 잘못 묶인 클러스터를
    뒤에서 바로잡아줄 단계가 없어졌으므로, 과소병합(중복이 남음)이 과대병합(내용이 손실됨)보다 안전하다는 판단.

## 6. 양방향 에스컬레이션 — "사기 → 검토필요" 디에스컬레이션 추가

- **문제**: v0.2는 `정상 → 검토필요` 한 방향만 있었음. CoT 분류가 "사기"라고 했는데 증거 탐지가
  아무것도 뒷받침해주지 못하는 경우를 구분할 방법이 없었음.
- **변경**: `cls_label == "사기"`인데 `evidence_score < escalate_thr`이면 `final_label = "검토필요"`로 내림.
  ```
  [디에스컬레이션] 사기 → 검토필요  (evidence_score=0.05 < 0.15, 뒷받침 증거 없음)
  ```
  - **어느 방향이든 "검토필요"로만 이동하고 "정상"으로 자동 확정되는 경로는 없음.** 증거탐지 recall이
    완벽하지 않은 상태에서 사기 판정을 "정상"으로 지워버리면 "사기를 놓치는 것보다 과탐지가 낫다"는
    원래 설계 원칙에 정면으로 위배되므로, 사람이 한 번 더 보도록 "검토필요"까지만 내림.

## 7. 결과 JSON에 `scam_evidence` 필드 추가

- **이전**: `objects` 필드에 사기증거(top_labels)와 일반객체가 섞여서 들어감. 구분할 방법이 없었음.
- **이후**: `objects`는 기존 스키마 호환을 위해 **그대로 합집합 유지**하고,
  `scam_evidence`(= `evidence["top_labels"]`)를 **별도 필드로 추가**. 평가/분석 시 두 종류를 나눠 볼 수 있음.
- `grounding` 필드명은 유지되지만 내용물은 이제 DINO가 아니라 `vlm_scam_evidence_batch()`의 산출물임
  (`labels` / `evidence_score` / `n_detected_frames` / `top_labels` 스키마는 `aggregate()`와 호환되게 맞춰서
  `should_escalate()`가 수정 없이 그대로 동작함).

## 지원 입력 방식 정리 (v0.3 기준 — v0.2에서 변동 없음)

| 옵션 | 대상 | 판정 단위 |
|---|---|---|
| `--url` | YouTube/TikTok 단일 영상 | 영상 1개 |
| `--file` | 로컬 영상 또는 이미지 1개 | 파일 1개 |
| `--dir` | 영상/이미지가 담긴 폴더 | **파일마다 독립 판정** |
| `--seq_dir` | 이어지는 이미지가 담긴 폴더 | **폴더 전체를 한 건으로 판정** |
| `--csv` | `label,link`(또는 `url`) CSV | 행마다 |

## 남은 정리 대상 / 확인 필요

아직 손대지 않았거나 확인이 필요한 항목 (다음 버전 또는 이번 버전 마무리 시 반영):

- **`evidence_score` 산식이 완전히 바뀌었는데 `ESCALATE_THR = 0.15`는 그대로임.** DINO는 confidence
  평균 기반이었고, VLM closed-set은 연속 score가 없어서 `min(1.0, Σ frame_ratio / 2)`라는 근사치를 씀.
  스케일이 다른 값에 옛 임계값을 그대로 쓰고 있으므로 재튜닝 필요. 6번의 디에스컬레이션도 같은 임계값을
  공유하므로 영향을 받음.
- **`SCAM_EVIDENCE_VOCAB`의 실제 항목 수는 36개**인데 코드 주석·docstring은 34종으로 적혀 있음. 숫자 정정 필요.
- **`dino_frames` / `--dino_sec` / `--max_dino_frames` 네이밍이 그대로 남아 있음.** DINO는 제거됐고
  이 프레임들은 이제 `vlm_scam_evidence_batch()`로 들어가므로 `evidence_frames` / `--evidence_sec` 등으로
  개명하는 게 맞음 (인자명 변경은 기존 실행 스크립트에 영향이 있어 별도 판단 필요).
- **`import onnxruntime as ort`가 실제로는 사용되지 않음.** RapidOCR 내부에서 쓰므로 명시적 import는 불필요.
- **`run_ocr_on_frame()` 안의 `correct_orientation(bgr)` 호출이 주석 처리된 채 남아 있음.** 함수 정의도 없으므로
  주석을 풀면 `NameError`. 사용할 거면 구현을, 아니면 주석을 지울 것.
- **`ObjectMapper.map()` / `_analyze_one_frame_vlm()` / `DEFAULT_PROMPT`가 미사용 상태로 남아 있음.**
- **PaddleOCR 대비 score 필터가 없어짐.** v0.2 `run_paddleocr_on_frame()`은 `score > 0.5`로 걸렀는데
  `run_ocr_on_frame()`은 빈 문자열만 거름. 저신뢰 검출이 그대로 클러스터링에 들어가므로 필터 재도입 검토 필요.
- **`app/main_v0_3.py` 미작성.** API 서버는 아직 `app/main_v0_2.py`(Grounding DINO + PaddleOCR 기반)에 머물러
  있어서 CLI와 동작이 갈라져 있음.
