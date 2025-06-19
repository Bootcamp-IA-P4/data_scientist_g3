import pytest
from fastapi.testclient import TestClient

@pytest.mark.integration
@pytest.mark.critical
class TestAPIEndpoints:
    def test_stroke_prediction_complete_workflow(self, test_client, test_data):
        """CRÍTICO: Prueba del flujo completo de predicción de ictus"""
        response = test_client.post("/predict/stroke", json=test_data)
        assert response.status_code == 200
        result = response.json()
        assert "prediction_id" in result
        assert "risk_level" in result

    def test_image_prediction_links_to_stroke(self, test_client, test_image, test_data):
        """CRÍTICO: Prueba de relación entre imagen y predicción de ictus"""
        # Primero crear predicción de ictus
        stroke_response = test_client.post("/predict/stroke", json=test_data)
        assert stroke_response.status_code == 200
        prediction_id = stroke_response.json()["prediction_id"]

        # Luego añadir imagen
        files = {"file": ("test.jpg", test_image, "image/jpeg")}
        response = test_client.post(f"/predict/image/{prediction_id}", files=files)
        assert response.status_code == 200
        result = response.json()
        assert "image_id" in result
        assert "filename" in result
        assert result["filename"] == "test.jpg"