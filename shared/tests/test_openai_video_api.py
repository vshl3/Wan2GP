import time
from pathlib import Path
from types import SimpleNamespace

from fastapi.testclient import TestClient

from shared.openai_video_api import create_app


class FakeSession:
    def __init__(self, output_dir: Path) -> None:
        self.output_dir = output_dir
        self.submitted_settings = []

    def list_model_metadata(self, **filters):
        return [{"model_type": "test-video", "main_output": ["video"]}]

    def get_model_metadata(self, model_type):
        if model_type == "test-video":
            return {"main_output": ["video"]}
        return None

    def get_default_settings(self, model_type):
        return {"model_type": model_type, "resolution": "640x360", "video_length": 17}

    def submit_task(self, settings, callbacks=None):
        self.submitted_settings.append(settings.copy())
        path = self.output_dir / f"{len(self.submitted_settings)}.mp4"
        path.write_bytes(b"generated video")
        if callbacks is not None:
            callbacks.on_progress(SimpleNamespace(progress=75))
        return SimpleNamespace(
            done=True,
            cancel=lambda: None,
            result=lambda: SimpleNamespace(
                success=True,
                cancelled=False,
                generated_files=[str(path)],
                errors=[],
                artifacts=[],
            ),
        )

    def close(self):
        pass


def test_video_generation_status_content_and_reference_upload(tmp_path):
    session = FakeSession(tmp_path)
    app = create_app(session, api_key="test-key")
    with TestClient(app) as client:
        unauthorized = client.get("/v1/models")
        assert unauthorized.status_code == 401
        assert unauthorized.json()["error"]["type"] == "authentication_error"
        headers = {"Authorization": "Bearer test-key"}
        models = client.get("/v1/models", headers=headers)
        assert models.status_code == 200
        assert models.json()["data"][0]["id"] == "test-video"

        response = client.post(
            "/v1/videos",
            headers=headers,
            data={"model": "test-video", "prompt": "A quiet ocean at sunrise", "seconds": "4", "size": "1280x720"},
            files={"input_reference": ("start.png", b"reference image", "image/png")},
        )
        assert response.status_code == 202
        video_id = response.json()["id"]
        deadline = time.monotonic() + 2
        while time.monotonic() < deadline:
            result = client.get(f"/v1/videos/{video_id}", headers=headers).json()
            if result["status"] != "queued" and result["status"] != "in_progress":
                break
            time.sleep(0.01)
        assert result["status"] == "completed"
        assert result["progress"] == 100
        content = client.get(f"/v1/videos/{video_id}/content", headers=headers)
        assert content.status_code == 200
        assert content.content == b"generated video"
        settings = session.submitted_settings[0]
        assert settings["video_length"] == "4s"
        assert settings["resolution"] == "1280x720"
        assert settings["image_start"]
        assert not Path(settings["image_start"]).exists()


def test_video_generation_rejects_non_video_models(tmp_path):
    app = create_app(FakeSession(tmp_path), api_key="test-key")
    with TestClient(app) as client:
        response = client.post(
            "/v1/videos",
            headers={"Authorization": "Bearer test-key"},
            data={"model": "unknown", "prompt": "A quiet ocean at sunrise"},
        )
    assert response.status_code == 400
    assert response.json()["error"]["type"] == "invalid_request_error"
