from fastapi.testclient import TestClient
import pytest

import app.main_v0_3 as api


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setattr(api, "LOAD_MODELS_ON_STARTUP", False)
    api.resources.ready = False
    api.resources.startup_error = None
    with TestClient(api.app) as test_client:
        yield test_client


def test_health_and_info_without_model_load(client):
    live = client.get("/health/live")
    ready = client.get("/health/ready")
    info = client.get("/api/info")

    assert live.status_code == 200
    assert live.json() == {"status": "ok"}
    assert ready.status_code == 503
    assert ready.json()["status"] == "not_ready"
    assert info.status_code == 200
    assert info.json()["vlm_model"] == "google/gemma-4-26B-A4B-it"


def test_analysis_is_503_until_models_are_ready(client):
    response = client.post(
        "/api/video",
        data={"url": "https://www.youtube.com/watch?v=abcdefghijk"},
    )

    assert response.status_code == 503


def test_optional_api_key(client, monkeypatch):
    monkeypatch.setattr(api, "API_KEY", "secret")

    assert client.get("/api/info").status_code == 401
    assert client.get("/api/info", headers={"X-API-Key": "secret"}).status_code == 200


@pytest.mark.parametrize(
    "url",
    [
        "https://www.youtube.com/watch?v=abcdefghijk",
        "https://youtu.be/abcdefghijk",
        "https://subdomain.tiktok.com/@user/video/123",
    ],
)
def test_allowed_urls(url):
    assert api._validate_url(url) == url


@pytest.mark.parametrize(
    "url",
    [
        "http://youtube.com/watch?v=x",
        "https://example.com/video",
        "https://youtube.com:8443/watch?v=x",
        "https://user:password@youtube.com/watch?v=x",
    ],
)
def test_disallowed_urls(url):
    with pytest.raises(api.HTTPException) as error:
        api._validate_url(url)
    assert error.value.status_code == 400


def test_result_normalization_has_stable_schema():
    result = api._normalize_result(
        {
            "id": "x",
            "label": "manual",
            "final_label": "정상",
            "grounding": {},
        }
    )

    assert result["input_label"] == "manual"
    assert result["label"] == "normal"
    assert result["objects"] == []
    assert result["ocr"] == []
    assert result["rag"] == []
