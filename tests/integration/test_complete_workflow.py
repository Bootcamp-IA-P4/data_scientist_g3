import pytest

@pytest.mark.integration
@pytest.mark.critical
class TestCompleteWorkflow:
    def test_full_prediction_workflow(self, test_client, test_data, test_image):
        """CRÍTICO: Prueba del flujo completo de trabajo"""
        # 1. Predicción de ictus
        stroke_response = test_client.post("/predict/stroke", json=test_data)
        assert stroke_response.status_code == 200
        result = stroke_response.json()
        assert "prediction_id" in result
        prediction_id = result["prediction_id"]

        # 2. Añadir imagen
        files = {"file": ("test.jpg", test_image, "image/jpeg")}
        image_response = test_client.post(f"/predict/image/{prediction_id}", files=files)
        assert image_response.status_code == 200
        image_result = image_response.json()
        assert "image_id" in image_result
        assert "filename" in image_result

        # 3. Verificar resultado combinado
        history_response = test_client.get("/predictions/stroke")
        assert history_response.status_code == 200
        predictions = history_response.json()
        assert len(predictions) > 0
        assert "image_url" in predictions[0]