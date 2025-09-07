from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)


def test_health():
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"


def test_detect():
    files = {"file": ("test.png", b"fake_image_bytes", "image/png")}
    resp = client.post("/detect", files=files)
    assert resp.status_code == 200
    assert "detections" in resp.json()
