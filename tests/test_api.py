from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def test_home_route():
    response = client.get("/")
    assert response.status_code == 200
    assert "message" in response.json()
