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
    def test_save_stroke_prediction_data_integrity(self):
        """CRÍTICO: Test save_stroke_prediction() - Integridad de datos"""
        try:
            from database.supabase_client import SupabaseClient
            
            # Mock del cliente Supabase
            with patch('database.supabase_client.create_client') as mock_create_client:
                mock_client = Mock()
                mock_create_client.return_value = mock_client
                
                # Configurar mock para inserción exitosa
                mock_client.table.return_value.insert.return_value.execute.return_value.data = [{
                    "id": 123,
                    "risk_level": "MEDIO",
                    "probability": 0.5,
                    "created_at": "2024-01-01T00:00:00Z"
                }]
                
                # Crear instancia del cliente
                config = {"url": "test_url", "key": "test_key"}
                client = SupabaseClient(config)
                
                # Datos de prueba
                prediction_data = {
                    "risk_level": "MEDIO",
                    "probability": 0.5,
                    "age": 65,
                    "gender": "Masculino"
                }
                
                # Ejecutar test
                result = client.save_stroke_prediction(prediction_data)
                
                # Verificar resultado
                assert result is not None
                assert "id" in result
                assert result["risk_level"] == "MEDIO"
                assert result["probability"] == 0.5
                
                # Verificar que se llamó al método correcto
                mock_client.table.assert_called_with("stroke_predictions")
                
        except ImportError:
            pytest.skip("Módulo database.supabase_client no disponible")
    
    @pytest.mark.critical
    def test_get_combined_predictions_with_without_image(self):
        """CRÍTICO: Test get_combined_predictions() - Manejo casos con/sin imagen"""
        try:
            from database.supabase_client import SupabaseClient
            
            with patch('database.supabase_client.create_client') as mock_create_client:
                mock_client = Mock()
                mock_create_client.return_value = mock_client
                
                # Configurar mock para predicciones con imagen
                mock_client.table.return_value.select.return_value.eq.return_value.execute.return_value.data = [
                    {
                        "id": 1,
                        "risk_level": "MEDIO",
                        "probability": 0.5,
                        "image_url": "http://example.com/image1.jpg"
                    },
                    {
                        "id": 2,
                        "risk_level": "ALTO",
                        "probability": 0.7,
                        "image_url": "http://example.com/image2.jpg"
                    }
                ]
                
                config = {"url": "test_url", "key": "test_key"}
                client = SupabaseClient(config)
                
                # Test con imagen
                result_with_image = client.get_combined_predictions(has_image=True)
                assert len(result_with_image) > 0
                assert "image_url" in result_with_image[0]
                assert result_with_image[0]["image_url"] is not None
                
                # Configurar mock para predicciones sin imagen
                mock_client.table.return_value.select.return_value.eq.return_value.execute.return_value.data = [
                    {
                        "id": 3,
                        "risk_level": "BAJO",
                        "probability": 0.2
                        # Sin image_url
                    }
                ]
                
                # Test sin imagen
                result_without_image = client.get_combined_predictions(has_image=False)
                assert len(result_without_image) > 0
                assert "image_url" not in result_without_image[0]
                
        except ImportError:
            pytest.skip("Módulo database.supabase_client no disponible")
    
    def test_connection_and_constraint_validations(self):
        """Test conexión y validaciones de restricciones"""
        try:
            from database.supabase_client import SupabaseClient
            
            with patch('database.supabase_client.create_client') as mock_create_client:
                mock_client = Mock()
                mock_create_client.return_value = mock_client
                
                # Test conexión exitosa
                config = {"url": "test_url", "key": "test_key"}
                client = SupabaseClient(config)
                
                # Verificar que se creó el cliente
                mock_create_client.assert_called_with("test_url", "test_key")
                
                # Test manejo de error de conexión
                mock_create_client.side_effect = Exception("Error de conexión")
                
                with pytest.raises(Exception, match="Error de conexión"):
                    SupabaseClient(config)
                    
        except ImportError:
            pytest.skip("Módulo database.supabase_client no disponible")
    
    @pytest.mark.critical
    def test_save_stroke_prediction_error_handling(self):
        """CRÍTICO: Test de manejo de errores en save_stroke_prediction"""
        try:
            from database.supabase_client import SupabaseClient
            
            with patch('database.supabase_client.create_client') as mock_create_client:
                mock_client = Mock()
                mock_create_client.return_value = mock_client
                
                # Configurar mock para error de inserción
                mock_client.table.return_value.insert.return_value.execute.side_effect = Exception("Error de base de datos")
                
                config = {"url": "test_url", "key": "test_key"}
                client = SupabaseClient(config)
                
                prediction_data = {
                    "risk_level": "MEDIO",
                    "probability": 0.5
                }
                
                # Verificar que se maneja el error
                with pytest.raises(Exception, match="Error de base de datos"):
                    client.save_stroke_prediction(prediction_data)
                    
        except ImportError:
            pytest.skip("Módulo database.supabase_client no disponible")
    
    def test_get_combined_predictions_empty_result(self):
        """Test de manejo de resultados vacíos en get_combined_predictions"""
        try:
            from database.supabase_client import SupabaseClient
            
            with patch('database.supabase_client.create_client') as mock_create_client:
                mock_client = Mock()
                mock_create_client.return_value = mock_client
                
                # Configurar mock para resultado vacío
                mock_client.table.return_value.select.return_value.eq.return_value.execute.return_value.data = []
                
                config = {"url": "test_url", "key": "test_key"}
                client = SupabaseClient(config)
                
                # Test con resultado vacío
                result = client.get_combined_predictions(has_image=True)
                assert isinstance(result, list)
                assert len(result) == 0
                
        except ImportError:
            pytest.skip("Módulo database.supabase_client no disponible")
    
    @pytest.mark.critical
    def test_data_validation_before_save(self):
        """CRÍTICO: Test de validación de datos antes de guardar"""
        try:
            from database.supabase_client import SupabaseClient
            
            with patch('database.supabase_client.create_client') as mock_create_client:
                mock_client = Mock()
                mock_create_client.return_value = mock_client
                
                config = {"url": "test_url", "key": "test_key"}
                client = SupabaseClient(config)
                
                # Test con datos inválidos
                invalid_data_cases = [
                    {},  # Datos vacíos
                    {"risk_level": "INVALIDO"},  # Nivel de riesgo inválido
                    {"probability": 1.5},  # Probabilidad fuera de rango
                    {"age": -1},  # Edad negativa
                ]
                
                for invalid_data in invalid_data_cases:
                    with pytest.raises((ValueError, TypeError)):
                        client.save_stroke_prediction(invalid_data)
                        
        except ImportError:
            pytest.skip("Módulo database.supabase_client no disponible")
    
    def test_connection_timeout_handling(self):
        """Test de manejo de timeouts de conexión"""
        try:
            from database.supabase_client import SupabaseClient
            import time
            
            with patch('database.supabase_client.create_client') as mock_create_client:
                mock_client = Mock()
                mock_create_client.return_value = mock_client
                
                # Simular timeout
                def slow_operation(*args, **kwargs):
                    time.sleep(0.1)  # Simular operación lenta
                    raise Exception("Timeout de conexión")
                
                mock_client.table.return_value.insert.return_value.execute.side_effect = slow_operation
                
                config = {"url": "test_url", "key": "test_key"}
                client = SupabaseClient(config)
                
                prediction_data = {"risk_level": "MEDIO", "probability": 0.5}
                
                with pytest.raises(Exception, match="Timeout de conexión"):
                    client.save_stroke_prediction(prediction_data)
                    
        except ImportError:
            pytest.skip("Módulo database.supabase_client no disponible") 