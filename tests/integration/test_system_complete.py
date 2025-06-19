import pytest
from fastapi.testclient import TestClient
import time

@pytest.mark.integration
@pytest.mark.critical
class TestSystemComplete:
    
    @pytest.mark.critical
    def test_complete_system_flow(self, test_client, test_data, test_image):
        """CRÍTICO: Test flujo completo - Predicción stroke → Añadir imagen → Historial combinado"""
        
        # === FASE 1: PREDICCIÓN DE STROKE ===
        print("🩺 Fase 1: Creando predicción de stroke...")
        stroke_response = test_client.post("/predict/stroke", json=test_data)
        assert stroke_response.status_code == 200
        
        stroke_result = stroke_response.json()
        assert "prediction_id" in stroke_result
        assert "risk_level" in stroke_result
        assert "probability" in stroke_result
        
        prediction_id = stroke_result["prediction_id"]
        print(f"✅ Predicción creada con ID: {prediction_id}")
        
        # === FASE 2: AÑADIR IMAGEN ===
        print("🖼️ Fase 2: Añadiendo imagen...")
        files = {"file": ("test.jpg", test_image, "image/jpeg")}
        image_response = test_client.post(f"/predict/image/{prediction_id}", files=files)
        assert image_response.status_code == 200
        
        image_result = image_response.json()
        assert "image_id" in image_result
        assert "filename" in image_result
        assert image_result["filename"] == "test.jpg"
        
        image_id = image_result["image_id"]
        print(f"✅ Imagen añadida con ID: {image_id}")
        
        # === FASE 3: VERIFICAR HISTORIAL COMBINADO ===
        print("📊 Fase 3: Verificando historial combinado...")
        history_response = test_client.get("/predictions/stroke")
        assert history_response.status_code == 200
        
        predictions = history_response.json()
        assert isinstance(predictions, list)
        assert len(predictions) > 0
        
        # Buscar nuestra predicción en el historial
        our_prediction = None
        for pred in predictions:
            if pred.get("id") == prediction_id:
                our_prediction = pred
                break
        
        assert our_prediction is not None, "Nuestra predicción debe estar en el historial"
        assert "image_url" in our_prediction, "Debe tener campo image_url"
        
        print(f"✅ Historial verificado - Predicción encontrada con imagen")
        
        # === FASE 4: VERIFICAR INTEGRIDAD DE DATOS ===
        print("🔍 Fase 4: Verificando integridad de datos...")
        
        # Verificar que los datos originales se mantienen
        assert our_prediction["risk_level"] == stroke_result["risk_level"]
        assert our_prediction["probability"] == stroke_result["probability"]
        
        print("✅ Integridad de datos verificada")
        
        return {
            "prediction_id": prediction_id,
            "image_id": image_id,
            "stroke_result": stroke_result,
            "image_result": image_result,
            "history_prediction": our_prediction
        }
    
    @pytest.mark.critical
    def test_cross_system_validation_no_image_without_stroke(self, test_client, test_image):
        """CRÍTICO: Test validación cross-sistema - No imagen sin stroke existente"""
        print("🚫 Test: Intentando subir imagen sin predicción de stroke...")
        
        # Intentar subir imagen sin predicción previa
        files = {"file": ("test.jpg", test_image, "image/jpeg")}
        response = test_client.post("/predict/image/999999", files=files)
        
        # Debe fallar porque no existe la predicción
        assert response.status_code == 404, "Debe fallar con 404 para ID inexistente"
        print("✅ Validación cross-sistema: Imagen rechazada sin stroke existente")
    
    def test_system_data_consistency(self, test_client, test_data, test_image):
        """Test de consistencia de datos en todo el sistema"""
        print("🔄 Test: Verificando consistencia de datos...")
        
        # Crear predicción
        stroke_response = test_client.post("/predict/stroke", json=test_data)
        prediction_id = stroke_response.json()["prediction_id"]
        
        # Añadir imagen
        files = {"file": ("test.jpg", test_image, "image/jpeg")}
        image_response = test_client.post(f"/predict/image/{prediction_id}", files=files)
        
        # Obtener historial
        history_response = test_client.get("/predictions/stroke")
        predictions = history_response.json()
        
        # Verificar consistencia
        stroke_data = stroke_response.json()
        image_data = image_response.json()
        
        # Los datos deben ser consistentes entre todas las respuestas
        assert stroke_data["prediction_id"] == prediction_id
        assert image_data["filename"] == "test.jpg"
        
        # Encontrar en historial
        history_pred = next((p for p in predictions if p["id"] == prediction_id), None)
        assert history_pred is not None
        assert history_pred["risk_level"] == stroke_data["risk_level"]
        
        print("✅ Consistencia de datos verificada")
    
    @pytest.mark.critical
    def test_system_error_recovery(self, test_client, test_data):
        """CRÍTICO: Test de recuperación de errores del sistema"""
        print("🛠️ Test: Verificando recuperación de errores...")
        
        # Test 1: Datos inválidos en predicción
        invalid_data = {
            "gender": "Masculino",
            "age": 150,  # Edad inválida
            "avg_glucose_level": 600  # Glucosa inválida
        }
        
        response = test_client.post("/predict/stroke", json=invalid_data)
        assert response.status_code == 422, "Debe rechazar datos inválidos"
        
        # Test 2: Crear predicción válida después del error
        valid_response = test_client.post("/predict/stroke", json=test_data)
        assert valid_response.status_code == 200, "Debe funcionar después del error"
        
        print("✅ Recuperación de errores verificada")
    
    def test_system_performance_under_load(self, test_client, test_data):
        """Test de rendimiento del sistema bajo carga"""
        print("⚡ Test: Verificando rendimiento bajo carga...")
        
        start_time = time.time()
        
        # Crear múltiples predicciones
        prediction_ids = []
        for i in range(5):
            response = test_client.post("/predict/stroke", json=test_data)
            assert response.status_code == 200
            prediction_ids.append(response.json()["prediction_id"])
        
        end_time = time.time()
        total_time = end_time - start_time
        
        # Verificar que el rendimiento es aceptable
        assert total_time < 10.0, f"Tiempo total muy alto: {total_time:.2f}s"
        assert len(prediction_ids) == 5, "Deben crearse todas las predicciones"
        
        print(f"✅ Rendimiento verificado: {total_time:.2f}s para 5 predicciones")
    
    @pytest.mark.critical
    def test_system_data_persistence(self, test_client, test_data, test_image):
        """CRÍTICO: Test de persistencia de datos del sistema"""
        print("💾 Test: Verificando persistencia de datos...")
        
        # Fase 1: Crear datos
        stroke_response = test_client.post("/predict/stroke", json=test_data)
        prediction_id = stroke_response.json()["prediction_id"]
        
        files = {"file": ("test.jpg", test_image, "image/jpeg")}
        image_response = test_client.post(f"/predict/image/{prediction_id}", files=files)
        image_id = image_response.json()["image_id"]
        
        # Fase 2: Verificar que los datos persisten
        history_response = test_client.get("/predictions/stroke")
        predictions = history_response.json()
        
        # Buscar nuestros datos
        our_prediction = next((p for p in predictions if p["id"] == prediction_id), None)
        assert our_prediction is not None, "Los datos deben persistir"
        assert "image_url" in our_prediction, "La imagen debe estar vinculada"
        
        print("✅ Persistencia de datos verificada")
    
    def test_system_concurrent_operations(self, test_client, test_data):
        """Test de operaciones concurrentes del sistema"""
        print("🔄 Test: Verificando operaciones concurrentes...")
        
        import threading
        import queue
        
        results = queue.Queue()
        
        def create_prediction():
            try:
                response = test_client.post("/predict/stroke", json=test_data)
                results.put(("success", response.status_code))
            except Exception as e:
                results.put(("error", str(e)))
        
        # Crear múltiples hilos
        threads = []
        for _ in range(3):
            thread = threading.Thread(target=create_prediction)
            threads.append(thread)
            thread.start()
        
        # Esperar a que terminen
        for thread in threads:
            thread.join()
        
        # Verificar resultados
        success_count = 0
        while not results.empty():
            status, result = results.get()
            if status == "success" and result == 200:
                success_count += 1
        
        assert success_count >= 2, "Al menos 2 de 3 operaciones concurrentes deben tener éxito"
        
        print(f"✅ Operaciones concurrentes verificadas: {success_count}/3 exitosas") 