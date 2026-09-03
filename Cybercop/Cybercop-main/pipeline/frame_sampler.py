"""프레임 샘플링 — v2 로직 재사용."""
import math
import cv2
from pathlib import Path
from PIL import Image


def sample_uniform(video_path: Path, every_n: float = 1.0, max_frames: int = 120) -> list:
    """균등 간격 샘플링. (sec, PIL.Image) 리스트 반환."""
    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    step = max(int(round(fps * every_n)), 1)
    frames, idx = [], 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if idx % step == 0:
            img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            frames.append((idx / fps, img))
            if len(frames) >= max_frames:
                break
        idx += 1
    cap.release()
    return frames


def sample_keyframe(video_path: Path, scan_sec: float = 2.0, max_frames: int = 12,
                    min_score: float = 3.0) -> list:
    """장면 변화 기반 키프레임 샘플링. (sec, PIL.Image) 리스트 반환."""
    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total / fps

    candidates, prev_gray = [], None
    sec = 0.0
    while sec <= duration:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(sec * fps))
        ret, frame = cap.read()
        if ret:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            score = float(cv2.absdiff(gray, prev_gray).mean()) if prev_gray is not None else float("inf")
            candidates.append((sec, frame.copy(), score))
            prev_gray = gray
        sec += scan_sec
    cap.release()

    if not candidates:
        return []

    min_gap = duration / max_frames if max_frames > 0 else 0
    filtered = [c for c in candidates if c[2] >= min_score] or [max(candidates, key=lambda x: x[2])]
    sorted_cands = sorted(filtered, key=lambda x: x[2], reverse=True)

    selected_ts, selected = [], []
    for cand in sorted_cands:
        ts = cand[0]
        if all(abs(ts - s) >= min_gap for s in selected_ts):
            selected.append(cand)
            selected_ts.append(ts)
        if len(selected) >= max_frames:
            break

    selected.sort(key=lambda x: x[0])
    return [(ts, Image.fromarray(cv2.cvtColor(f, cv2.COLOR_BGR2RGB))) for ts, f, _ in selected]


def make_grid(frames: list, max_dim: int = 512) -> Image.Image:
    """프레임 리스트 → 격자 이미지."""
    n = len(frames)
    cols = math.ceil(math.sqrt(n))
    rows = math.ceil(n / cols)
    if frames:
        fw, fh = frames[0].size
        if fh > fw:
            cell_w, cell_h = round(fw / fh * max_dim), max_dim
        else:
            cell_w, cell_h = max_dim, round(fh / fw * max_dim)
    else:
        cell_w = cell_h = max_dim
    grid = Image.new("RGB", (cols * cell_w, rows * cell_h), (30, 30, 30))
    for i, img in enumerate(frames):
        r, c = divmod(i, cols)
        grid.paste(img.resize((cell_w, cell_h), Image.LANCZOS), (c * cell_w, r * cell_h))
    return grid
