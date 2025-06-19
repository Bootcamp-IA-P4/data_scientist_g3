import pytest
from fastapi.testclient import TestClient

@pytest.mark.integration
@pytest.mark.critical
class TestAPIEndpointsDetailed:
    
    @pytest.mark.critical
    def test_post_predict_stroke_complete_flow(self, test_client, test_data):
        """CRÍTICO: Test POST /predict/stroke - Flujo completo API → Pipeline → BD"""
        # 1. Enviar solicitud de predicción
        response = test_client.post("/predict/stroke", json=test_data)
        
        # 2. Verificar respuesta exitosa
        assert response.status_code == 200
        result = response.json()
        
        # 3. Verificar estructura de respuesta
        assert "prediction_id" in result
        assert "risk_level" in result
        assert "probability" in result
        assert "status" in result
        
        # 4. Verificar tipos de datos
        assert isinstance(result["prediction_id"], int)
        assert isinstance(result["risk_level"], str)
        assert isinstance(result["probability"], float)
        assert isinstance(result["status"], str)
        
        # 5. Verificar valores válidos
        assert result["risk_level"] in ["BAJO", "MEDIO", "ALTO", "CRÍTICO"]
        assert 0 <= result["probability"] <= 1
        assert result["status"] == "success"
    
    @pytest.mark.critical
    def test_post_predict_image_stroke_link(self, test_client, test_data, test_image):
        """CRÍTICO: Test POST /predict/image/{id} - Vinculación stroke-imagen correcta"""
        # 1. Crear predicción de stroke primero
        stroke_response = test_client.post("/predict/stroke", json=test_data)
        assert stroke_response.status_code == 200
        prediction_id = stroke_response.json()["prediction_id"]
        
        # 2. Subir imagen vinculada a la predicción
        files = {"file": ("test.jpg", test_image, "image/jpeg")}
        image_response = test_client.post(f"/predict/image/{prediction_id}", files=files)
        
        # 3. Verificar respuesta exitosa
        assert image_response.status_code == 200
        image_result = image_response.json()
        
        # 4. Verificar estructura de respuesta
        assert "image_id" in image_result
        assert "prediction" in image_result
        assert "probability" in image_result
        assert "risk_level" in image_result
        assert "filename" in image_result
        
        # 5. Verificar vinculación correcta
        assert image_result["filename"] == "test.jpg"
        assert image_result["risk_level"] in ["BAJO", "MEDIO", "ALTO", "CRÍTICO"]
        assert 0 <= image_result["probability"] <= 1
    
    def test_get_predictions_stroke_data_retrieval(self, test_client, test_data):
        """Test GET /predictions/stroke - Recuperación de datos"""
        # 1. Crear algunas predicciones
        for _ in range(3):
            response = test_client.post("/predict/stroke", json=test_data)
            assert response.status_code == 200
        
        # 2. Obtener historial de predicciones
        history_response = test_client.get("/predictions/stroke")
        assert history_response.status_code == 200
        predictions = history_response.json()
        
        # 3. Verificar estructura de respuesta
        assert isinstance(predictions, list)
        assert len(predictions) > 0
        
        # 4. Verificar estructura de cada predicción
        for prediction in predictions:
            assert "id" in prediction
            assert "risk_level" in prediction
            assert "probability" in prediction
            assert "image_url" in prediction  # Puede ser None si no hay imagen
    
    @pytest.mark.critical
    def test_api_error_handling_invalid_data(self, test_client):
        """CRÍTICO: Test de manejo de errores con datos inválidos"""
        # 1. Test con datos incompletos
        incomplete_data = {
            "gender": "Masculino",
            "age": 65
            # Faltan campos requeridos
        }
        
        response = test_client.post("/predict/stroke", json=incomplete_data)
        assert response.status_code == 422  # Validation Error
        
        # 2. Test con datos fuera de rango
        invalid_data = {
            "gender": "Masculino",
            "age": 150,  # Edad fuera de rango
            "hypertension": "Sí",
            "heart_disease": "No",
            "ever_married": "Sí",
            "work_type": "Privado",
            "residence_type": "Urbano",
            "avg_glucose_level": 600,  # Glucosa fuera de rango
            "bmi": 28.5,
            "smoking_status": "Fumó antes"
        }
        
        response = test_client.post("/predict/stroke", json=invalid_data)
        assert response.status_code == 422  # Validation Error
    
    @pytest.mark.critical
    def test_api_error_handling_invalid_image(self, test_client, test_data, test_image):
        """CRÍTICO: Test de manejo de errores con imagen inválida"""
        # 1. Crear predicción válida
        stroke_response = test_client.post("/predict/stroke", json=test_data)
        prediction_id = stroke_response.json()["prediction_id"]
        
        # 2. Test con archivo no-imagen
        invalid_file = b"this is not an image"
        files = {"file": ("test.txt", invalid_file, "text/plain")}
        
        response = test_client.post(f"/predict/image/{prediction_id}", files=files)
        assert response.status_code == 400  # Bad Request
        
        # 3. Test con ID de predicción inválido
        files = {"file": ("test.jpg", test_image, "image/jpeg")}
        response = test_client.post("/predict/image/999999", files=files)
        assert response.status_code == 404  # Not Found
    
    def test_api_cross_validation_no_image_without_stroke(self, test_client, test_image):
        """Test validación cross-sistema - No imagen sin stroke existente"""
        # Intentar subir imagen sin predicción de stroke previa
        files = {"file": ("test.jpg", test_image, "image/jpeg")}
        response = test_client.post("/predict/image/999999", files=files)
        
        # Debería fallar porque no existe la predicción de stroke
        assert response.status_code == 404  # Not Found
    
    def test_api_response_consistency(self, test_client, test_data):
        """Test de consistencia en respuestas de la API"""
        # Realizar múltiples predicciones con los mismos datos
        responses = []
        for _ in range(3):
            response = test_client.post("/predict/stroke", json=test_data)
            assert response.status_code == 200
            responses.append(response.json())
        
        # Verificar que todas las respuestas tienen la misma estructura
        for i, result in enumerate(responses):
            assert "prediction_id" in result
            assert "risk_level" in result
            assert "probability" in result
            assert "status" in result
            assert result["status"] == "success"
    
    @pytest.mark.critical
    def test_api_performance_basic(self, test_client, test_data):
        """CRÍTICO: Test básico de rendimiento de la API"""
        import time
        
        # Medir tiempo de respuesta para predicción de stroke
        start_time = time.time()
        response = test_client.post("/predict/stroke", json=test_data)
        end_time = time.time()
        
        assert response.status_code == 200
        response_time = end_time - start_time
        
        # Verificar que la respuesta es razonablemente rápida (< 5 segundos)
        assert response_time < 5.0, f"Respuesta muy lenta: {response_time:.2f}s"
        
        # Medir tiempo de respuesta para obtener historial
        start_time = time.time()
        history_response = test_client.get("/predictions/stroke")
        end_time = time.time()
        
        assert history_response.status_code == 200
        history_time = end_time - start_time
        
        # Verificar que la respuesta es razonablemente rápida (< 2 segundos)
        assert history_time < 2.0, f"Respuesta de historial muy lenta: {history_time:.2f}s" 