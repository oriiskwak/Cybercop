"""
download_models.py — 위험도 모델 체크포인트를 내려받는다.

`agents/` 아래의 코드·토크나이저는 저장소에 포함되어 있지만, 체크포인트 두 개는 각각 400MB를 넘어
GitHub 파일 크기 제한(100MB)을 초과하므로 저장소에 넣을 수 없다. 대신 GitHub Releases에 올려두고
이 스크립트로 받아온다 (Release 첨부파일은 파일당 2GB까지 허용되고 저장소 용량에 포함되지 않는다).

사용:
    python download_models.py              # 없는 것만 받음
    python download_models.py --force      # 이미 있어도 다시 받음

체크포인트가 없어도 파이프라인은 동작한다 — 사기유형 분류(키워드 기반)는 체크포인트가 필요 없고,
위험도(text_risk / guideline_risk)만 `available: false`로 반환된다.
"""
from __future__ import annotations

import argparse
import hashlib
import sys
import urllib.error
import urllib.request
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

# GitHub Releases 태그. 새 버전을 올리면 이 값만 바꾸면 된다.
RELEASE_TAG = "models-v0.4"
RELEASE_BASE = f"https://github.com/oriiskwak/Cybercop/releases/download/{RELEASE_TAG}"

MODELS = [
    {
        "name": "text_risk 3-class 분류기",
        "path": BASE_DIR / "agents" / "text_risk_model" / "checkpoints"
                / "rule14b1_final_risk_level_text_classifier_best.pt",
        "url": f"{RELEASE_BASE}/rule14b1_final_risk_level_text_classifier_best.pt",
        "size_mb": 423,
    },
    {
        "name": "guideline S1~S6 결합 모델",
        "path": BASE_DIR / "agents" / "guideline_multitask_model" / "checkpoints"
                / "guideline_multitask_best.pt",
        "url": f"{RELEASE_BASE}/guideline_multitask_best.pt",
        "size_mb": 425,
    },
]


def _progress(done: int, total: int, label: str) -> None:
    if total <= 0:
        sys.stdout.write(f"\r  {label}: {done / 1048576:.0f}MB")
    else:
        pct = done / total * 100
        sys.stdout.write(f"\r  {label}: {pct:5.1f}%  ({done / 1048576:.0f}/{total / 1048576:.0f}MB)")
    sys.stdout.flush()


def download(spec: dict, force: bool = False) -> bool:
    dest = spec["path"]
    if dest.exists() and not force:
        print(f"[skip] {spec['name']} — 이미 있음 ({dest.stat().st_size / 1048576:.0f}MB)")
        return True

    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    print(f"[받는 중] {spec['name']}  (~{spec['size_mb']}MB)")
    print(f"          {spec['url']}")

    try:
        with urllib.request.urlopen(spec["url"]) as resp, open(tmp, "wb") as f:
            total = int(resp.headers.get("Content-Length") or 0)
            done = 0
            while chunk := resp.read(1024 * 256):
                f.write(chunk)
                done += len(chunk)
                _progress(done, total, dest.name)
        print()
    except urllib.error.HTTPError as e:
        tmp.unlink(missing_ok=True)
        print(f"\n[실패] HTTP {e.code} — Release에 파일이 올라가 있는지 확인하세요: {spec['url']}")
        return False
    except Exception as e:
        tmp.unlink(missing_ok=True)
        print(f"\n[실패] {e}")
        return False

    tmp.replace(dest)
    print(f"[완료] {dest.relative_to(BASE_DIR)}")
    return True


def main() -> int:
    parser = argparse.ArgumentParser(description="위험도 모델 체크포인트 다운로드")
    parser.add_argument("--force", action="store_true", help="이미 있어도 다시 받기")
    args = parser.parse_args()

    ok = all(download(spec, force=args.force) for spec in MODELS)

    print()
    if ok:
        print("모든 체크포인트 준비 완료. 이제 파이프라인을 실행할 수 있습니다:")
        print("  python cybercop_pipeline_AdotX_v0_4.py --file data/eximg.png")
        return 0

    print("일부 체크포인트를 받지 못했습니다.")
    print("체크포인트 없이도 파이프라인은 동작하지만, 위험도(text_risk/guideline_risk)는")
    print("available: false 로 반환되고 사기유형 분류만 수행됩니다.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
