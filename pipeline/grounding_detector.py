"""
Grounding DINO를 이용해 (이미지, 텍스트 후보 리스트) → bbox 탐지.

모델: IDEA-Research/grounding-dino-tiny (속도 우선)
       IDEA-Research/grounding-dino-base (정확도 우선)
설치: pip install transformers torch
"""
import torch
from pathlib import Path
from PIL import Image
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection


def _map_label(raw_label: str, en_to_ko: dict) -> str:
    """
    DINO가 반환하는 레이블을 한국어로 역매핑.
    DINO는 가끔 여러 항목을 붙여서 반환하거나(예: "kakaotalk chat screen 화면"),
    반대로 어휘 구문의 일부만 잘라서 반환하기도 함(예: "kakaotalk chat screen" → "kakaotalk").
    """
    key = raw_label.lower().strip()
    # 1) 완전 일치
    if key in en_to_ko:
        return en_to_ko[key]
    # 2) DINO 출력이 vocab 구문을 포함 (vocab 쪽이 부분 문자열)
    best_ko, best_len = None, 0
    for en, ko in en_to_ko.items():
        if en in key and len(en) > best_len:
            best_ko, best_len = ko, len(en)
    if best_ko:
        return best_ko
    # 3) DINO 출력이 vocab 구문의 일부(반대 방향) — 이 fragment를 포함하는 vocab이 딱 하나일 때만
    # 매핑 (예: "kakaotalk"은 "KakaoTalk chat screen" 하나뿐이라 안전. "screen"처럼 여러 항목에
    # 공통으로 들어있는 fragment는 어떤 화면인지 특정할 수 없어서 매핑 안 하고 원본 유지)
    if len(key) >= 3:
        candidates = {ko for en, ko in en_to_ko.items() if key in en}
        if len(candidates) == 1:
            return next(iter(candidates))
    # 매핑 실패 시 원본(영어) 반환
    return key

DEFAULT_MODEL_ID  = "IDEA-Research/grounding-dino-tiny"
DEFAULT_BOX_THR   = 0.40   # 낮으면 할루시네이션 증가
DEFAULT_TEXT_THR  = 0.35   # 낮으면 관계없는 텍스트 스팬 매칭


class GroundingDetector:
    def __init__(
        self,
        model_id: str = DEFAULT_MODEL_ID,
        device: str | None = None,
        box_threshold: float = DEFAULT_BOX_THR,
        text_threshold: float = DEFAULT_TEXT_THR,
        local_dir: str | None = None,
    ):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.box_threshold  = box_threshold
        self.text_threshold = text_threshold

        # 로컬 캐시 우선 (네트워크 차단 환경 대비)
        src = local_dir or model_id
        print(f"  [GroundingDINO] 로드 중: {src}  (device={self.device})")
        self.processor = AutoProcessor.from_pretrained(src)
        self.model = AutoModelForZeroShotObjectDetection.from_pretrained(src).to(self.device)
        self.model.eval()
        print("  [GroundingDINO] 로드 완료.")

    # 한 번에 넣을 최대 항목 수 — 너무 많으면 DINO가 스팬을 혼동해 할루시네이션 발생
    CHUNK_SIZE = 10

    def detect(self, image: Image.Image, vocab: dict[str, str]) -> list[dict]:
        """
        image: PIL 이미지
        vocab: {한국어 레이블: 영어 DINO 쿼리}

        vocab을 CHUNK_SIZE개씩 나눠 여러 번 추론 → 결과 합산.
        DINO에는 영어 쿼리, 반환 레이블은 한국어.
        """
        if not vocab:
            return []

        en_to_ko = {en.lower(): ko for ko, en in vocab.items()}
        items = list(vocab.items())   # [(ko, en), ...]
        h, w  = image.size[1], image.size[0]
        all_dets: list[dict] = []

        for start in range(0, len(items), self.CHUNK_SIZE):
            chunk = items[start:start + self.CHUNK_SIZE]
            en_queries = [en for _, en in chunk]
            text_prompt = " . ".join(q.lower().strip() for q in en_queries) + " ."

            inputs = self.processor(
                images=image, text=text_prompt, return_tensors="pt"
            ).to(self.device)

            with torch.no_grad():
                outputs = self.model(**inputs)

            results = self.processor.post_process_grounded_object_detection(
                outputs,
                inputs.input_ids,
                threshold=self.box_threshold,
                text_threshold=self.text_threshold,
                target_sizes=[(h, w)],
            )[0]

            for box, score, label in zip(results["boxes"], results["scores"], results["labels"]):
                ko_label = _map_label(label, en_to_ko)
                all_dets.append({
                    "box":   [round(float(x), 1) for x in box.tolist()],
                    "label": ko_label,
                    "score": round(float(score), 4),
                })

        return all_dets

    def detect_batch(self, frames: list, vocab: dict[str, str]) -> list[dict]:
        """
        여러 프레임에 대해 순차적으로 detect 수행.
        Returns: List of {"sec": float, "detections": [...]}
        """
        results = []
        for sec, img in frames:
            dets = self.detect(img, vocab)
            results.append({"sec": sec, "detections": dets})
        return results
