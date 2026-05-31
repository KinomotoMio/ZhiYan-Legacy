import asyncio
from pathlib import Path

from fastapi.testclient import TestClient

from app.core.config import settings
from app.services.sessions.store import SessionStore


def _install_temp_session_store(monkeypatch, tmp_path: Path) -> SessionStore:
    import app.services.sessions as sessions_pkg
    from app.api.v1 import chat as chat_api
    from app.api.v1 import sessions as sessions_api
    from app.api.v1 import workspaces as workspaces_api
    from app.api.v1 import workspace_sources as workspace_sources_api

    store = SessionStore(tmp_path / "zhiyan-test.db", tmp_path / "uploads")
    asyncio.run(store.init())

    monkeypatch.setattr(sessions_pkg, "session_store", store)
    monkeypatch.setattr(sessions_api, "session_store", store)
    monkeypatch.setattr(chat_api, "session_store", store)
    monkeypatch.setattr(workspace_sources_api, "session_store", store)
    monkeypatch.setattr(workspaces_api, "session_store", store)
    monkeypatch.setattr(settings, "project_root", tmp_path)
    return store


def _create_session(client: TestClient, headers: dict[str, str], title: str) -> str:
    response = client.post("/api/v1/sessions", headers=headers, json={"title": title})
    assert response.status_code == 200
    return response.json()["id"]


def _sample_presentation() -> dict:
    return {
        "presentationId": "pres-1",
        "title": "Agent Loop Speaker Notes",
        "slides": [
            {
                "slideId": "slide-1",
                "layoutType": "intro-slide",
                "layoutId": "intro-slide",
                "contentData": {"title": "封面"},
                "speakerNotes": "旧封面注解",
            },
            {
                "slideId": "slide-2",
                "layoutType": "summary-section-title",
                "layoutId": "summary-section-title",
                "contentData": {"title": "关键发现"},
                "speakerNotes": "旧内容页注解",
                "speakerAudio": {
                    "provider": "minimax",
                    "model": "speech-2.8-hd",
                    "voiceId": "male-qn-qingse",
                    "textHash": "stale-hash",
                    "storagePath": "/tmp/stale.mp3",
                    "mimeType": "audio/mpeg",
                    "generatedAt": "2026-03-27T12:00:00Z",
                },
            },
        ],
    }


def _save_latest_presentation(store: SessionStore, session_id: str, payload: dict) -> None:
    asyncio.run(
        store.save_presentation(
            session_id=session_id,
            payload=payload,
            is_snapshot=False,
            snapshot_label=None,
        )
    )


def test_request_minimax_tts_accepts_zero_status_code(monkeypatch):
    from app.services.speaker_audio import _request_minimax_tts

    class _FakeResponse:
        status_code = 200

        def json(self):
            return {
                "base_resp": {"status_code": 0, "status_msg": "success"},
                "data": {"audio": "66616b652d6d7033"},
            }

    class _FakeClient:
        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return None

        async def post(self, *_args, **_kwargs):
            return _FakeResponse()

    original_provider = settings.tts_provider
    original_key = settings.tts_api_key
    original_base_url = settings.tts_base_url
    original_model = settings.tts_model
    original_voice_id = settings.tts_voice_id
    settings.tts_provider = "minimax"
    settings.tts_api_key = "tts-secret-key"
    settings.tts_base_url = "https://api.minimaxi.com"
    settings.tts_model = "speech-2.8-hd"
    settings.tts_voice_id = "male-qn-qingse"
    monkeypatch.setattr("app.services.speaker_audio.get_safe_httpx_client", lambda **_kwargs: _FakeClient())

    try:
        assert asyncio.run(_request_minimax_tts("最小测试")) == b"fake-mp3"
    finally:
        settings.tts_provider = original_provider
        settings.tts_api_key = original_key
        settings.tts_base_url = original_base_url
        settings.tts_model = original_model
        settings.tts_voice_id = original_voice_id
