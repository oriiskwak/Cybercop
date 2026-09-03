from types import SimpleNamespace

import pytest
from PIL import Image

import cybercop_pipeline_AdotX_v0_3 as core


def test_vlm_evidence_counts_detected_frames(monkeypatch):
    label_a, label_b = core.SCAM_EVIDENCE_LABELS[:2]
    replies = iter(
        [
            f"[EVIDENCE] {label_a}",
            "[EVIDENCE] 없음",
            f"[EVIDENCE] {label_a}, {label_b}",
        ]
    )
    monkeypatch.setattr(core, "_vlm_generate", lambda *args, **kwargs: next(replies))
    frame = Image.new("RGB", (8, 8), "white")

    result = core.vlm_scam_evidence_batch(
        [(0.0, frame), (1.0, frame), (2.0, frame)],
        processor=object(),
        model=object(),
        device="cpu",
        video_duration=3.0,
    )

    assert result["n_detected_frames"] == 2
    assert result["labels"][label_a]["frame_count"] == 2
    assert result["labels"][label_b]["frame_count"] == 1
    assert 0.0 <= result["evidence_score"] <= 1.0


def test_ocr_cluster_uses_majority_text():
    detections = [
        {
            "frame_idx": index,
            "sec": float(index),
            "text": text,
            "score": score,
            "bbox": [0.0, 0.0, 100.0, 20.0],
        }
        for index, text, score in [
            (0, "선입금 요청", 0.90),
            (1, "선입금 요청", 0.95),
            (2, "선입금 요정", 0.80),
        ]
    ]

    spans, candidates = core.build_ocr_outputs(detections)

    assert spans == [{"start": "00:00", "end": "00:02", "text": "선입금 요청"}]
    assert candidates[0]["stable"] is True
    assert candidates[0]["candidates"][0]["count"] == 2


def test_invalid_video_is_rejected(tmp_path):
    invalid = tmp_path / "invalid.mp4"
    invalid.write_bytes(b"not a video")
    args = SimpleNamespace(
        scan_sec=0.5,
        max_vlm_frames=4,
        evidence_sec=3.0,
        max_evidence_frames=15,
        sample_sec=2.0,
        max_frames=100,
    )

    with pytest.raises(ValueError, match="비디오"):
        core._get_video_frames(invalid, args)


def test_load_vlm_uses_multimodal_auto_class(monkeypatch, tmp_path):
    calls = {}

    class FakeProcessor:
        @classmethod
        def from_pretrained(cls, path, **kwargs):
            calls["processor"] = (path, kwargs)
            return object()

    class FakeModelInstance:
        def eval(self):
            calls["eval"] = True

    class FakeModel:
        @classmethod
        def from_pretrained(cls, path, **kwargs):
            calls["model"] = (path, kwargs)
            return FakeModelInstance()

    def fake_snapshot_download(repo_id, local_dir):
        calls["download"] = (repo_id, local_dir)
        Path(local_dir).mkdir(parents=True)
        return local_dir

    from pathlib import Path

    monkeypatch.setattr(core, "MODEL_CACHE_DIR", tmp_path)
    monkeypatch.setattr(core, "AutoProcessor", FakeProcessor)
    monkeypatch.setattr(core, "AutoModelForMultimodalLM", FakeModel)
    monkeypatch.setattr(core, "snapshot_download", fake_snapshot_download)

    processor, model = core.load_vlm("example/multimodal")

    assert processor is not None
    assert model is not None
    assert calls["download"][0] == "example/multimodal"
    assert "dtype" in calls["model"][1]
    assert "torch_dtype" not in calls["model"][1]
    assert calls["eval"] is True
