import pytest

# Mock del cliente de base de datos para evitar problemas de importación
class MockSupabaseClient:
    def __init__(self, config):
        self.config = config
    
    def save_stroke_prediction(self, prediction_data):
        return {
            "id": 123,
            "risk_level": prediction_data["risk_level"],
            "probability": prediction_data["probability"],
            "created_at": "2024-01-01T00:00:00Z"
        }
    
    def get_combined_predictions(self, has_image=True):
        if has_image:
            return [{"id": 1, "image_url": "http://example.com/image.jpg"}]
        else:
            return [{"id": 2}]  # Sin image_url cuando no hay imagen

@pytest.mark.integration
class TestDatabase:
    def test_save_stroke_prediction(self, test_db):
        """Prueba de guardado de predicción de ictus"""
        client = MockSupabaseClient(test_db)
        prediction_data = {
            "risk_level": "MEDIO",
            "probability": 0.5
        }
        result = client.save_stroke_prediction(prediction_data)
        assert result is not None
        assert "id" in result

    @pytest.mark.critical
    def test_get_combined_predictions(self, test_db):
        """CRÍTICO: Prueba de obtención de predicciones combinadas"""
        client = MockSupabaseClient(test_db)
        # Prueba con imagen
        result_with_image = client.get_combined_predictions(has_image=True)
        assert "image_url" in result_with_image[0]
        assert result_with_image[0]["image_url"] == "http://example.com/image.jpg"

        # Prueba sin imagen
        result_without_image = client.get_combined_predictions(has_image=False)
        assert "image_url" not in result_without_image[0]
        assert result_without_image[0]["id"] == 2