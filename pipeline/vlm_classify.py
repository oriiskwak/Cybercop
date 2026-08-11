"""
2단계 분류:
  1단계 (VLM + 이미지): 묘사만 생성 — 판단 없음
  2단계 (VLM 텍스트만): 묘사 + 체크리스트 → 사기/정상 판단

VLM이 이미지를 보면서 동시에 논리 규칙을 적용하는 게 어렵기 때문에 분리.
텍스트 전용 2단계는 이미지 인코딩 없으므로 1단계보다 5-10배 빠름.
"""
import re
import torch
from PIL import Image
from .frame_sampler import make_grid

CLASSIFY_GRID_FRAMES = 4

# ── 1단계: 이미지 → 묘사 ─────────────────────────────────────────────────────
DESCRIBE_PROMPT_TEMPLATE = """\
이 이미지는 하나의 동영상에서 장면 전환이 큰 {total}개의 대표 프레임을 격자로 이어붙인 것이다.
왼쪽→오른쪽, 위→아래 순서가 시간 순서다.
{title_section}
화면에서 직접 관찰되는 내용을 있는 그대로 서술하라.
(텍스트, 화면 요소, 사람, 제품, 행동, UI 등 — 판단하지 말고 관찰한 것만 기술)

반드시 아래 형식으로만 출력하라:
[묘사] 3-4문장으로 관찰 내용 서술
"""

# ── 2단계: 묘사 텍스트 → 판단 ────────────────────────────────────────────────
JUDGE_PROMPT_TEMPLATE = """\
아래는 동영상의 대표 프레임을 관찰한 내용이다.
{title_section}
[관찰 내용]
{description}

【판단 기준】 아래 항목 중 하나라도 관찰 내용에 해당하면 사기로 판단하라:
(1) 주식·수익률 화면 + 텔레그램/카카오 오픈채팅 링크·ID가 함께 언급
(2) "고수익 보장", "월 OOO% 수익" 등 구체적 수익 약속 텍스트
(3) 검찰·금융감독원 등 기관 사칭 문서나 화면
(4) 재택·부업으로 고수익을 약속하는 광고 텍스트
(5) 금전을 요청하는 채팅 화면
(6) 제품 영상에서 카카오톡·텔레그램·라인 등 개인 연락처가 직접 노출
(7) 앱 설치 유도 후 계좌번호·비밀번호·잔액 등 금융 정보 입력 화면
(8) 주식·투자 화면 + "꼭 사라", "지금 사야 해" 등 긴급 매수 권유 문구

반드시 아래 형식으로만 출력하라:
[해당항목] 해당되는 번호 (없으면 "없음")
[판단] 사기 또는 정상
[이유] 한 문장
"""


def classify_scam(processor, model, frames: list, title: str = "",
                  device: str = "cuda") -> tuple:
    """
    2단계 분류. Returns: (pred_label, summary)
      - pred_label: "사기" | "정상" | "불명확"
      - summary: 묘사 + 판단근거 (로그/저장용)
    """
    if not frames:
        return "불명확", ""

    pil_frames    = [img for _, img in frames[:CLASSIFY_GRID_FRAMES]]
    grid_img      = make_grid(pil_frames)
    title_section = f"\n[영상 제목]\n{title}\n" if title else ""

    # ── 1단계: 이미지 → 묘사 ──
    description = _run_vlm(
        processor, model, device,
        image=grid_img,
        prompt=DESCRIBE_PROMPT_TEMPLATE.format(
            total=len(pil_frames), title_section=title_section
        ),
    )
    desc_m    = re.search(r"\[묘사\]\s*(.+)", description, re.S)
    desc_text = desc_m.group(1).strip() if desc_m else description.strip()

    # ── 2단계: 묘사 텍스트 → 판단 (이미지 없음, 빠름) ──
    judgment = _run_text(
        processor, model, device,
        prompt=JUDGE_PROMPT_TEMPLATE.format(
            title_section=title_section,
            description=desc_text,
        ),
    )

    label_m  = re.search(r"\[판단\]\s*(사기|정상)", judgment)
    reason_m = re.search(r"\[이유\]\s*(.+?)(?:\n|$)", judgment)
    pred     = label_m.group(1).strip() if label_m else "불명확"
    reason   = reason_m.group(1).strip() if reason_m else ""
    summary  = f"{desc_text} / 판단근거: {reason}"

    return pred, summary


# ── 헬퍼 ──────────────────────────────────────────────────────────────────────

def _run_vlm(processor, model, device, image: Image.Image,
             prompt: str, max_new_tokens: int = 300) -> str:
    """이미지 + 텍스트 → 텍스트."""
    messages = [{"role": "user", "content": [
        {"type": "image", "image": image.convert("RGB")},
        {"type": "text",  "text": prompt},
    ]}]
    text_input = processor.apply_chat_template(
        messages, add_generation_prompt=True, tokenize=False
    )
    inputs = processor(
        text=[text_input], images=[image.convert("RGB")], return_tensors="pt"
    ).to(device)
    with torch.inference_mode():
        out = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
    return processor.batch_decode(
        out[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True
    )[0].strip()


def _run_text(processor, model, device,
              prompt: str, max_new_tokens: int = 150) -> str:
    """텍스트만 입력 → 텍스트 (이미지 없음, 빠름)."""
    messages   = [{"role": "user", "content": prompt}]
    text_input = processor.apply_chat_template(
        messages, add_generation_prompt=True, tokenize=False
    )
    inputs = processor(text=[text_input], return_tensors="pt").to(device)
    with torch.inference_mode():
        out = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
    return processor.batch_decode(
        out[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True
    )[0].strip()
