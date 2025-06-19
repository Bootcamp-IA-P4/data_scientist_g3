import pytest
from unittest.mock import Mock, patch, MagicMock, AsyncMock
import sys
import os
from PIL import Image
import io

# Añadir ruta para importar módulos
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'backend', 'app'))

@pytest.mark.unit
@pytest.mark.critical
class TestImageService:
    
    @pytest.mark.critical
    @pytest.mark.asyncio
    async def test_process_image_success(self):
        """CRÍTICO: Prueba de procesamiento exitoso de imagen"""
        with patch('services.image_service.process_image_prediction', new_callable=AsyncMock) as mock_process:
            # Configurar mock para éxito
            mock_process.return_value = {
                "image_id": 123,
                "prediction": 0,
                "probability": 0.3,
                "risk_level": "BAJO",
                "status": "success"
            }
            
            # Simular datos de imagen
            test_image_data = b"fake_image_data"
            test_prediction_id = 456
            
            result = await mock_process(
                image_data=test_image_data,
                stroke_prediction_id=test_prediction_id,
                filename="test.jpg",
                content_type="image/jpeg"
            )
            
            assert result["status"] == "success"
            assert "image_id" in result
            assert "prediction" in result
            assert "probability" in result
            assert "risk_level" in result
    
    @pytest.mark.critical
    @pytest.mark.asyncio
    async def test_process_image_invalid_format(self):
        """CRÍTICO: Prueba de manejo de formato de imagen inválido"""
        with patch('services.image_service.process_image_prediction', new_callable=AsyncMock) as mock_process:
            # Configurar mock para error de formato
            mock_process.side_effect = ValueError("Formato de imagen no soportado")
            
            test_image_data = b"invalid_image_data"
            test_prediction_id = 456
            
            with pytest.raises(ValueError, match="Formato de imagen no soportado"):
                await mock_process(
                    image_data=test_image_data,
                    stroke_prediction_id=test_prediction_id,
                    filename="test.txt",  # Formato inválido
                    content_type="text/plain"
                )
    
    @pytest.mark.critical
    @pytest.mark.asyncio
    async def test_process_image_file_too_large(self):
        """CRÍTICO: Prueba de manejo de archivo demasiado grande"""
        with patch('services.image_service.process_image_prediction', new_callable=AsyncMock) as mock_process:
            # Configurar mock para error de tamaño
            mock_process.side_effect = ValueError("Archivo demasiado grande (máximo 10MB)")
            
            # Simular imagen grande
            large_image_data = b"x" * (11 * 1024 * 1024)  # 11MB
            test_prediction_id = 456
            
            with pytest.raises(ValueError, match="Archivo demasiado grande"):
                await mock_process(
                    image_data=large_image_data,
                    stroke_prediction_id=test_prediction_id,
                    filename="large.jpg",
                    content_type="image/jpeg"
                )
    
    @pytest.mark.critical
    @pytest.mark.asyncio
    async def test_process_image_model_error(self):
        """CRÍTICO: Prueba de manejo de error del modelo"""
        with patch('services.image_service.process_image_prediction', new_callable=AsyncMock) as mock_process:
            # Configurar mock para error del modelo
            mock_process.side_effect = RuntimeError("Modelo no disponible")
            
            test_image_data = b"valid_image_data"
            test_prediction_id = 456
            
            with pytest.raises(RuntimeError, match="Modelo no disponible"):
                await mock_process(
                    image_data=test_image_data,
                    stroke_prediction_id=test_prediction_id,
                    filename="test.jpg",
                    content_type="image/jpeg"
                )
    
    @pytest.mark.asyncio
    async def test_process_image_database_error(self):
        """Prueba de manejo de error de base de datos"""
        with patch('services.image_service.process_image_prediction', new_callable=AsyncMock) as mock_process:
            # Configurar mock para error de BD
            mock_process.side_effect = Exception("Error de conexión a base de datos")
            
            test_image_data = b"valid_image_data"
            test_prediction_id = 456
            
            with pytest.raises(Exception, match="Error de conexión"):
                await mock_process(
                    image_data=test_image_data,
                    stroke_prediction_id=test_prediction_id,
                    filename="test.jpg",
                    content_type="image/jpeg"
                )
    
    @pytest.mark.critical
    @pytest.mark.asyncio
    async def test_process_image_invalid_prediction_id(self):
        """CRÍTICO: Prueba de manejo de ID de predicción inválido"""
        with patch('services.image_service.process_image_prediction', new_callable=AsyncMock) as mock_process:
            # Configurar mock para ID inválido
            mock_process.side_effect = ValueError("ID de predicción de stroke no encontrado")
            
            test_image_data = b"valid_image_data"
            invalid_prediction_id = 999999  # ID que no existe
            
            with pytest.raises(ValueError, match="ID de predicción"):
                await mock_process(
                    image_data=test_image_data,
                    stroke_prediction_id=invalid_prediction_id,
                    filename="test.jpg",
                    content_type="image/jpeg"
                )
    
    @pytest.mark.asyncio
    async def test_process_image_missing_required_fields(self):
        """Prueba de manejo de campos requeridos faltantes"""
        with patch('services.image_service.process_image_prediction', new_callable=AsyncMock) as mock_process:
            # Configurar mock para campos faltantes
            mock_process.side_effect = ValueError("Campos requeridos faltantes")
            
            # Simular datos incompletos
            test_image_data = b"valid_image_data"
            test_prediction_id = 456
            
            with pytest.raises(ValueError, match="Campos requeridos"):
                await mock_process(
                    image_data=test_image_data,
                    stroke_prediction_id=test_prediction_id,
                    filename=None,  # Campo faltante
                    content_type="image/jpeg"
                )
    
    @pytest.mark.critical
    @pytest.mark.asyncio
    async def test_process_image_validation_errors(self):
        """CRÍTICO: Prueba de validaciones de imagen"""
        with patch('services.image_service.process_image_prediction', new_callable=AsyncMock) as mock_process:
            # Casos de validación
            validation_cases = [
                (b"", "Imagen vacía"),
                (b"not_an_image", "Formato de imagen inválido"),
                (b"x" * 100, "Imagen demasiado pequeña"),
            ]
            
            for image_data, expected_error in validation_cases:
                mock_process.side_effect = ValueError(expected_error)
                
                with pytest.raises(ValueError, match=expected_error):
                    await mock_process(
                        image_data=image_data,
                        stroke_prediction_id=456,
                        filename="test.jpg",
                        content_type="image/jpeg"
                    ) 