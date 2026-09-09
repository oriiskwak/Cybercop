# CHANGELOG — cybercop_pipeline_AdotX_v0_1.py → cybercop_pipeline_AdotX_v0_2.py

`cybercop_pipeline_AdotX_v0_1.py`는 그대로 두고, 아래 변경사항을 반영한 `cybercop_pipeline_AdotX_v0_2.py`를
새 파일로 만듦 (앞으로도 뭔가 바뀌면 기존 버전 파일은 건드리지 않고 다음 버전 파일을 새로 만드는 방식으로 진행).

## 1. 이미지 시퀀스(이어지는 스크린샷 여러 장) 입력 지원 — `--seq_dir` 추가

- **문제**: 카카오톡 대화를 이어 찍은 스크린샷처럼, 여러 장이 한 세트로 이어지는 이미지를 어떻게 넣을지가 없었음.
  기존 `--dir`은 폴더 안 파일들을 **각각 독립된 건으로 따로따로** 판정하기 때문에, 대화 스크린샷 6장을 `--dir`에
  넣으면 한 장씩 뚝뚝 끊어서 보고 6개의 별개 결과가 나옴 — 대화 흐름 전체를 하나로 보고 판단해야 하는 케이스에 안 맞음.
- **변경**: `--seq_dir <폴더경로>` 신규 옵션 추가. 폴더 안 이미지들을 파일명 순으로 정렬해 `[(0.0, img1), (1.0, img2), ...]`
  형태의 **하나의 프레임 시퀀스**로 묶어서, 폴더 전체를 한 건으로 `_analyze_and_save()`(CoT 분류 → DINO → 에스컬레이션
  → OCR/객체/RAG) 공통 로직에 넘김. 결과 JSON도 폴더당 1개(`ocr_results_{폴더명}.json`).
  - `process_image_sequence()` 함수로 구현. `_get_image_frames()`(이미지 1장 → 프레임 1개짜리 리스트)와 같은 방식으로,
    프레임 리스트 포맷만 맞추면 나머지 파이프라인은 코드 수정 없이 그대로 재사용된다는 걸 활용.
  - 기존 `--dir`(파일별 독립 판정)은 그대로 유지, `--seq_dir`은 별개 옵션으로 추가 (우선순위: `--url` > `--file` >
    `--dir` > `--seq_dir` > `--csv`).

## 2. 분류용 대표 프레임을 이미지 개수에 따라 고르게 추출

- **문제**: `classify_scam()`은 넘어온 프레임 중 앞에서부터 `max_vlm_frames`(기본 4)장만 잘라서 grid 이미지를 만듦.
  영상은 `sample_keyframe()`이 미리 장면전환 기준으로 대표 프레임을 골라주지만, `--seq_dir`처럼 프레임 리스트를
  직접 만들어 넘기면 이 사전 선별이 없어서 **그냥 앞 4장만 보고 뒤쪽은 아예 안 봄**. 스크린샷 6장짜리 대화에서
  결정적 증거(계좌번호 요구, 송금 완료 화면 등)가 뒤쪽에 있으면 놓칠 수 있음.
- **변경**: `process_image_sequence()` 안에서, 이미지 개수가 `max_vlm_frames`보다 많으면 전체 구간에서 균등한
  인덱스로 미리 뽑아서 `vlm_frames`로 넘김 (예: 6장 중 4장 필요하면 인덱스 `[0, 2, 3, 5]` — 처음과 끝을 포함해
  고르게 분포). `max_vlm_frames <= 1`인 극단적 설정도 0-division 없이 첫 프레임만 쓰도록 처리.
  OCR/DINO용 프레임(`ocr_frames`/`dino_frames`)은 이미지 시퀀스 전체를 그대로 다 씀 (개수 제한 없음, 증거는
  많이 볼수록 좋으므로).

## 3. 이미지 시퀀스는 OCR 프레임간 클러스터링을 건너뛰고 이미지별로 개별 교정

- **문제**: `--seq_dir`로 카톡 대화 스크린샷 6장을 실제로 돌려보니 OCR 결과가 엉망으로 나옴. 원인은
  `build_ocr_outputs`(IoU+CER 프레임간 클러스터링)이 **영상**을 전제로 설계된 로직이기 때문 — "같은 화면
  요소가 여러 프레임에 걸쳐 반복 등장한다"는 가정을 까는데, 스크린샷 6장은 서로 다른 대화 내용이 화면상
  비슷한 위치(상단 상태바, 말풍선 좌표 등)에 우연히 겹칠 뿐임. 그 결과 관련 없는 텍스트끼리 잘못 묶이거나
  대부분 낱줄 단위(약 140개)로 쪼개짐. 그 140개 조각을 한 번에 `postprocess_ocr_with_vlm`에 넣어 "문장
  연결/병합"을 시켰더니, VLM이 무리하게 5덩어리로 뭉치면서 `start`/`end` 타임스탬프까지 전부 엉망(`00:05`로
  통일됨)이 됨.
- **변경**: `_analyze_and_save()`에 `is_sequence` 플래그 추가. `True`일 때는 클러스터링+`postprocess_ocr_with_vlm`을
  건너뛰고, 이미지별 원문(`ocr_before[i]`, 이미 `_merge_same_line`으로 줄 단위는 정리된 상태)을 그대로 유지한 채
  신규 함수 `correct_ocr_text_with_vlm()`으로 **이미지 1장씩 개별** VLM 교정(오타/오인식만, 문장 재배열·타
  이미지와 병합 없음)만 수행. `process_image_sequence()`가 `is_sequence=True`로 호출.
  - 영상 입력(`--url`/`--file`/`--dir`의 영상)은 기존 클러스터링 경로 그대로 유지 (`is_sequence` 기본값 `False`).

## 4. `classify_summary`는 할루시네이션에 취약함 — 콘솔 로그는 판단근거만 출력

- **문제**: `classify_scam()` 1단계(묘사)는 "정확한 글자 판독"이 아니라 VLM이 화면을 보고 받은 인상을
  서술하는 단계라, 작은 글자(카톡 대화방 이름 등)를 잘못 읽는 할루시네이션이 실제로 발생함 (예: 실제
  대화방 이름은 "정회원상담"인데 "정치상담"으로 잘못 서술 — OCR 쪽은 "정회원상담"으로 정확히 잡음).
  기존 콘솔 로그(`print(f"  [분류]  {cls_label}  | {cls_summary[:80]}")`)는 이 할루시네이션되기 쉬운 묘사
  앞부분만 잘라서 보여줘서, 정작 판정에 실제로 쓰인 판단근거를 보려면 결과 JSON을 열어봐야 했음.
- **변경**: 정규식으로 `cls_summary`에서 `판단근거:` 이후 부분만 뽑아 콘솔에 출력하도록 변경.
  `classify_summary` 필드 자체(묘사+판단근거 통짜 문자열)와 그 안의 할루시네이션 가능성은 그대로 남아있음 —
  이건 이름을 바꾸거나 로그를 바꾼다고 없어지는 문제가 아니라, 정확한 텍스트가 필요하면 `ocr`/`ocr_after`
  필드(OCR 기반, 정확도 더 높음)를 봐야 함.

## 5. 결과 JSON 필드명 변경: `ocr_text` → `ocr_after`

- **이전**: `ocr_text` (교정된 OCR 스팬을 `" | "`로 합친 문자열 하나)
- **이후**: `ocr_after` — `ocr_before`(교정 전 원문)와 이름 대칭을 맞춰서 전/후 관계가 필드명만 보고도
  바로 이해되게 함. 값과 계산 방식은 동일, 이름만 바뀜.

## 6. 출력 디렉토리 분리

- **이전**: `--out_dir` 기본값 `./output_AdotX_v0.1`
- **이후**: `./output_AdotX_v0.2` — 버전별로 결과가 안 섞이게 분리.

## 7. API 서버에 이미지 시퀀스 엔드포인트 추가 — `app/main_v0_2.py` (신규 파일)

- **문제**: `app/main_v0_1.py`는 CLI의 `--seq_dir`에 대응하는 엔드포인트가 없어서, 이미지 시퀀스를 API로는
  분석할 방법이 없었음.
- **변경**: `app/main_v0_1.py`를 복사해 `app/main_v0_2.py`를 만들고 `cybercop_pipeline_AdotX_v0_2`를 사용하도록
  전환. `POST /api/video/upload/sequence` 신규 추가 — CLI `process_image_sequence()`와 동일한 로직
  (`_get_sequence_frames()`로 분류용 대표 프레임 균등 추출, OCR은 클러스터링 없이 이미지별 개별 교정).
  - 이미지 여러 장은 `files`(배열) 또는 폴더를 압축한 `archive`(zip) 중 하나로 받음. **Swagger UI(`/docs`)에서
    `files`(배열 파일 필드)는 파일선택 버튼 대신 문자열 입력칸으로 잘못 렌더링됨** — FastAPI가 생성하는
    OpenAPI 3.1 스펙의 `items.contentMediaType`을 이 Swagger UI 버전이 배열 안에서는 인식 못 하는 게 원인
    (API 자체는 정상, curl로는 잘 동작). 그래서 `archive`(zip, 단일 파일 필드라 정상 렌더링됨)를 추가해서
    브라우저에서도 폴더를 통째로 올릴 수 있게 함.

## 8. 사람이 바로 읽는 콘솔 요약 출력 — `print_report()`

- **문제**: 처리 결과를 사람이 확인하려면 결과 JSON 파일을 열어야 했음. 콘솔에는 판단근거 앞부분만 잘려서 나옴.
- **변경**: `print_report(final_label, reason, evidence, rag_results)` 함수 추가. 결과 JSON 파일/필드는 그대로
  두고, 콘솔(CLI)·서버 로그(API)에 아래 4줄을 추가로 출력:
  ```
  판정: 사기
  근거 텍스트: (판단근거)
  근거 객체(탐지): (grounding.labels 키 전체, 필터 없음)
  범죄 유형: (rag 1순위 결과의 major_category)
  ```
  `cybercop_pipeline_AdotX_v0_2.py`(CLI)와 `app/main_v0_2.py`(API) 둘 다에서 호출.

## 9. Grounding DINO 라벨 한국어 역매핑 개선 (공용 모듈 `pipeline/grounding_detector.py`)

- **문제**: DINO가 vocab 구문의 일부만 잘라서 반환하는 경우(예: `"KakaoTalk chat screen"` 대신 그냥
  `"kakaotalk"`, `"ATM screen"` 대신 `"atm"`) 기존 `_map_label()`은 못 잡고 영어 원문을 그대로 반환했음
  (실사례: `grounding.labels`에 `"kakaotalk"`, `"screen"`이 한글 안 거치고 그대로 노출).
- **변경**: 이 fragment를 포함하는 vocab 항목이 **정확히 하나뿐일 때만** 그 한국어 라벨로 매핑하는 역방향
  매칭 추가 (`"kakaotalk"` → `"카카오톡 화면"`, `"atm"` → `"ATM 화면"`, `"telegram"` → `"텔레그램 화면"`).
  `"screen"`처럼 여러 vocab 항목에 공통으로 들어있어서 어떤 화면인지 특정할 수 없는 fragment는 잘못
  매핑하지 않고 그대로 둠 (한글로 강제 매핑하면 오히려 부정확).
  `pipeline/`은 v0.1/v0.2 공용이라 이 수정은 두 버전 모두에 적용됨.

## 지원 입력 방식 정리 (v0.2 기준)

| 옵션 | 대상 | 판정 단위 |
|---|---|---|
| `--url` | YouTube/TikTok 단일 영상 | 영상 1개 |
| `--file` | 로컬 영상 또는 이미지 1개 | 파일 1개 |
| `--dir` | 영상/이미지가 담긴 폴더 | **파일마다 독립 판정** |
| `--seq_dir` | 이어지는 이미지가 담긴 폴더 | **폴더 전체를 한 건으로 판정** (신규) |
| `--csv` | `label,link`(또는 `url`) CSV | 행마다 |
