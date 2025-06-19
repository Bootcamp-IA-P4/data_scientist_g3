import pytest
from unittest.mock import Mock, patch, MagicMock
import sys
import os

# Añadir ruta para importar módulos
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'backend', 'app'))

@pytest.mark.integration
@pytest.mark.critical
class TestSupabaseClient:
    
    @pytest.mark.critical
    @pytest.mark.asyncio
    async def test_save_stroke_prediction_data_integrity(self):
        """CRÍTICO: Test save_stroke_prediction() - Integridad de datos"""
        try:
            from database.supabase_client import save_stroke_prediction
            
            # Mock de psycopg2.connect
            with patch('database.supabase_client.psycopg2.connect') as mock_connect:
                mock_conn = Mock()
                mock_cursor = Mock()
                mock_connect.return_value = mock_conn
                mock_conn.cursor.return_value = mock_cursor
                
                # Datos de prueba
                prediction_data = {
                    "gender": "Masculino",
                    "age": 65,
                    "hypertension": "Sí",
                    "heart_disease": "No",
                    "ever_married": "Sí",
                    "work_type": "Privado",
                    "residence_type": "Urbano",
                    "avg_glucose_level": 180,
                    "bmi": 28.5,
                    "smoking_status": "Nunca fumó",
                    "prediction": 0,
                    "probability": 0.3,
                    "risk_level": "BAJO"
                }
                
                # Ejecutar test
                result = await save_stroke_prediction(prediction_data)
                
                # Verificar resultado
                assert result is True
                
                # Verificar que se llamó al método correcto
                mock_cursor.execute.assert_called_once()
                mock_conn.commit.assert_called_once()
                
        except ImportError:
            pytest.skip("Módulo database.supabase_client no disponible")
    
    @pytest.mark.critical
    @pytest.mark.asyncio
    async def test_get_combined_predictions_with_without_image(self):
        """CRÍTICO: Test get_combined_predictions() - Manejo casos con/sin imagen"""
        try:
            from database.supabase_client import get_combined_predictions
            
            with patch('database.supabase_client.psycopg2.connect') as mock_connect:
                mock_conn = Mock()
                mock_cursor = Mock()
                mock_connect.return_value = mock_conn
                mock_conn.cursor.return_value = mock_cursor
                
                # Configurar mock para predicciones con imagen
                # Estructura de datos que espera la función get_combined_predictions
                mock_cursor.fetchall.return_value = [
                    # stroke_data + image_data combinados
                    (1, "2024-01-01", "Masculino", 65, "Sí", "No", "Sí", "Privado", "Urbano", 180, 28.5, "Nunca fumó", 0, 0.5, "MEDIO", 1, 1, 0.7, "ALTO", 0.9, "image1.jpg", "http://example.com/image1.jpg", 1024, "JPEG", 100, True),
                    (2, "2024-01-01", "Femenino", 70, "No", "Sí", "Sí", "Público", "Rural", 200, 30.0, "Fumó antes", 1, 0.7, "ALTO", 2, 1, 0.8, "ALTO", 0.95, "image2.jpg", "http://example.com/image2.jpg", 2048, "PNG", 150, True)
                ]
                
                # Test con imagen
                result_with_image = await get_combined_predictions(limit=10)
                assert len(result_with_image) > 0
                assert "stroke_data" in result_with_image[0]
                assert "image_data" in result_with_image[0]
                assert result_with_image[0]["has_image"] is True
                
                # Configurar mock para predicciones sin imagen
                mock_cursor.fetchall.return_value = [
                    # stroke_data sin image_data
                    (3, "2024-01-01", "Masculino", 55, "No", "No", "Sí", "Autónomo", "Urbano", 150, 25.0, "Nunca fumó", 0, 0.2, "BAJO", None, None, None, None, None, None, None, None, None, None, False)
                ]
                
                # Test sin imagen
                result_without_image = await get_combined_predictions(limit=10)
                assert len(result_without_image) > 0
                assert "stroke_data" in result_without_image[0]
                assert result_without_image[0]["has_image"] is False
                
        except ImportError:
            pytest.skip("Módulo database.supabase_client no disponible")
    
    @pytest.mark.asyncio
    async def test_connection_and_constraint_validations(self):
        """Test conexión y validaciones de restricciones"""
        try:
            from database.supabase_client import save_stroke_prediction
            
            with patch('database.supabase_client.psycopg2.connect') as mock_connect:
                # Test conexión exitosa
                mock_conn = Mock()
                mock_cursor = Mock()
                mock_connect.return_value = mock_conn
                mock_conn.cursor.return_value = mock_cursor
                
                prediction_data = {
                    "gender": "Masculino",
                    "age": 65,
                    "hypertension": "Sí",
                    "heart_disease": "No",
                    "ever_married": "Sí",
                    "work_type": "Privado",
                    "residence_type": "Urbano",
                    "avg_glucose_level": 180,
                    "bmi": 28.5,
                    "smoking_status": "Nunca fumó",
                    "prediction": 0,
                    "probability": 0.3,
                    "risk_level": "BAJO"
                }
                
                result = await save_stroke_prediction(prediction_data)
                assert result is True
                
                # Test manejo de error de conexión
                mock_connect.side_effect = Exception("Error de conexión")
                
                result = await save_stroke_prediction(prediction_data)
                assert result is False
                    
        except ImportError:
            pytest.skip("Módulo database.supabase_client no disponible")
    
    @pytest.mark.critical
    @pytest.mark.asyncio
    async def test_save_stroke_prediction_error_handling(self):
        """CRÍTICO: Test de manejo de errores en save_stroke_prediction"""
        try:
            from database.supabase_client import save_stroke_prediction
            
            with patch('database.supabase_client.psycopg2.connect') as mock_connect:
                mock_conn = Mock()
                mock_cursor = Mock()
                mock_connect.return_value = mock_conn
                mock_conn.cursor.return_value = mock_cursor
                
                # Configurar mock para error de inserción
                mock_cursor.execute.side_effect = Exception("Error de base de datos")
                
                prediction_data = {
                    "gender": "Masculino",
                    "age": 65,
                    "hypertension": "Sí",
                    "heart_disease": "No",
                    "ever_married": "Sí",
                    "work_type": "Privado",
                    "residence_type": "Urbano",
                    "avg_glucose_level": 180,
                    "bmi": 28.5,
                    "smoking_status": "Nunca fumó",
                    "prediction": 0,
                    "probability": 0.3,
                    "risk_level": "BAJO"
                }
                
                # Verificar que se maneja el error
                result = await save_stroke_prediction(prediction_data)
                assert result is False
                    
        except ImportError:
            pytest.skip("Módulo database.supabase_client no disponible")
    
    @pytest.mark.asyncio
    async def test_get_combined_predictions_empty_result(self):
        """Test de manejo de resultados vacíos en get_combined_predictions"""
        try:
            from database.supabase_client import get_combined_predictions
            
            with patch('database.supabase_client.psycopg2.connect') as mock_connect:
                mock_conn = Mock()
                mock_cursor = Mock()
                mock_connect.return_value = mock_conn
                mock_conn.cursor.return_value = mock_cursor
                
                # Configurar mock para resultado vacío
                mock_cursor.fetchall.return_value = []
                
                # Test con resultado vacío
                result = await get_combined_predictions(limit=10)
                assert isinstance(result, list)
                assert len(result) == 0
                
        except ImportError:
            pytest.skip("Módulo database.supabase_client no disponible")
    
    @pytest.mark.critical
    @pytest.mark.asyncio
    async def test_data_validation_before_save(self):
        """CRÍTICO: Test de validación de datos antes de guardar"""
        try:
            from database.supabase_client import save_stroke_prediction
            
            with patch('database.supabase_client.psycopg2.connect') as mock_connect:
                mock_conn = Mock()
                mock_cursor = Mock()
                mock_connect.return_value = mock_conn
                mock_conn.cursor.return_value = mock_cursor
                
                # Test con datos válidos
                valid_data = {
                    "gender": "Masculino",
                    "age": 65,
                    "hypertension": "Sí",
                    "heart_disease": "No",
                    "ever_married": "Sí",
                    "work_type": "Privado",
                    "residence_type": "Urbano",
                    "avg_glucose_level": 180,
                    "bmi": 28.5,
                    "smoking_status": "Nunca fumó",
                    "prediction": 0,
                    "probability": 0.3,
                    "risk_level": "BAJO"
                }
                
                result = await save_stroke_prediction(valid_data)
                assert result is True
                
                # Test con datos incompletos (debería fallar)
                invalid_data = {
                    "gender": "Masculino",
                    "age": 65
                    # Faltan campos requeridos
                }
                
                result = await save_stroke_prediction(invalid_data)
                assert result is False
                
        except ImportError:
            pytest.skip("Módulo database.supabase_client no disponible")
    
    @pytest.mark.asyncio
    async def test_connection_timeout_handling(self):
        """Test de manejo de timeouts de conexión"""
        try:
            from database.supabase_client import save_stroke_prediction
            
            with patch('database.supabase_client.psycopg2.connect') as mock_connect:
                # Simular timeout
                def slow_operation(*args, **kwargs):
                    import time
                    time.sleep(0.1)  # Simular operación lenta
                    raise Exception("Timeout de conexión")
                
                mock_connect.side_effect = slow_operation
                
                prediction_data = {
                    "gender": "Masculino",
                    "age": 65,
                    "hypertension": "Sí",
                    "heart_disease": "No",
                    "ever_married": "Sí",
                    "work_type": "Privado",
                    "residence_type": "Urbano",
                    "avg_glucose_level": 180,
                    "bmi": 28.5,
                    "smoking_status": "Nunca fumó",
                    "prediction": 0,
                    "probability": 0.3,
                    "risk_level": "BAJO"
                }
                
                result = await save_stroke_prediction(prediction_data)
                assert result is False
                
        except ImportError:
            pytest.skip("Módulo database.supabase_client no disponible") 