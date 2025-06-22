import pytest
from PIL import Image
import io
from pathlib import Path
import tempfile
import numpy as np
import torch
from src.pipeline.image_pipeline import StrokeImagePipeline
import time

            
@pytest.mark.unit
@pytest.mark.critical
class TestStrokeImagePipeline:
    @pytest.mark.critical
    def test_validate_image_valid_input(self, test_image):
        """Prueba de imagen válida"""
        pipeline = StrokeImagePipeline()
        result = pipeline.validate_image(test_image)
        
        assert result["valid"] is True
        assert len(result["errors"]) == 0
        assert result["metadata"]["width"] == 224
        assert result["metadata"]["height"] == 224
        assert result["metadata"]["format"] in ['JPEG', 'PNG', 'WEBP', 'BMP']

    @pytest.mark.critical
    def test_validate_image_from_path(self):
        """Prueba de validación de imagen desde ruta de archivo"""
        pipeline = StrokeImagePipeline()
        temp_file = None
        
        try:
            # Crear archivo temporal de imagen
            with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
                temp_file = tmp.name
                img = Image.new('RGB', (224, 224))
                img.save(temp_file, format='PNG')
            
            # Cerrar archivo antes de la validación
            result = pipeline.validate_image(temp_file)
            assert result["valid"] is True
            assert result["metadata"]["file_size_mb"] is not None
            
        finally:
            # Eliminar archivo temporal después de todas las operaciones
            if temp_file and Path(temp_file).exists():
                Path(temp_file).unlink()

    @pytest.mark.critical
    def test_validate_image_too_small(self):
        """Prueba de imagen demasiado pequeña"""
        pipeline = StrokeImagePipeline()
        small_image = Image.new('RGB', (16, 16))
        
        result = pipeline.validate_image(small_image)
        assert result["valid"] is False
        assert "muy pequeña" in result["errors"][0]

    @pytest.mark.critical
    def test_validate_image_too_large(self):
        """Prueba de imagen demasiado grande"""
        pipeline = StrokeImagePipeline()
        
        # Crear imagen grande
        large_image = Image.new('RGB', (10000, 10000))
        img_byte_arr = io.BytesIO()
        
        # Guardar en formato BMP (sin compresión)
        large_image.save(img_byte_arr, format='BMP')
        img_byte_arr = img_byte_arr.getvalue()
        
        # Mostrar tamaño para depuración
        print(f"Tamaño del archivo: {len(img_byte_arr) / (1024*1024):.2f} MB")
        
        result = pipeline.validate_image(img_byte_arr)
        print(f"Resultado de validación: {result}")
        
        assert result["valid"] is False
        assert "muy grande" in result["errors"][0]

    @pytest.mark.critical
    def test_validate_image_unsupported_format(self):
        """Prueba de formato no soportado"""
        pipeline = StrokeImagePipeline()
        temp_file = None
        
        try:
            # Crear imagen en formato no soportado
            with tempfile.NamedTemporaryFile(suffix='.tiff', delete=False) as tmp:
                temp_file = tmp.name
                img = Image.new('RGB', (224, 224))
                img.save(temp_file, format='TIFF')
            
            # Cerrar archivo antes de la validación
            result = pipeline.validate_image(temp_file)
            assert result["valid"] is False
            assert "Formato no soportado" in result["errors"][0]
            
        finally:
            # Eliminar archivo temporal después de todas las operaciones
            if temp_file and Path(temp_file).exists():
                Path(temp_file).unlink()

    @pytest.mark.critical
    def test_validate_image_multiple_errors(self):
        """Prueba de múltiples errores"""
        pipeline = StrokeImagePipeline()
        temp_file = None
        
        try:
            # Crear imagen que viola múltiples reglas
            with tempfile.NamedTemporaryFile(suffix='.tiff', delete=False) as tmp:
                temp_file = tmp.name
                img = Image.new('RGB', (16, 16))
                img.save(temp_file, format='TIFF')
            
            # Cerrar archivo antes de la validación
            result = pipeline.validate_image(temp_file)
            assert result["valid"] is False
            assert len(result["errors"]) > 1
            assert any("muy pequeña" in error for error in result["errors"])
            assert any("Formato no soportado" in error for error in result["errors"])
            
        finally:
            # Eliminar archivo temporal después de todas las operaciones
            if temp_file and Path(temp_file).exists():
                Path(temp_file).unlink()

    @pytest.mark.critical
    def test_validate_image_metadata_completeness(self, test_image):
        """Prueba de completitud de metadatos"""
        pipeline = StrokeImagePipeline()
        result = pipeline.validate_image(test_image)
        
        metadata = result["metadata"]
        assert all(key in metadata for key in ["width", "height", "format", "file_size_mb"])
        assert isinstance(metadata["width"], int)
        assert isinstance(metadata["height"], int)
        assert isinstance(metadata["format"], str)
        assert metadata["file_size_mb"] is None or isinstance(metadata["file_size_mb"], float)
    
    
    def test_predict_bajo_riesgo(self, mock_ml_model, mock_model_class, test_image):
        """Prueba de predicción con nivel de riesgo bajo"""
        mock_ml_model.model = mock_model_class(0.2)
        
        result = mock_ml_model.predict(test_image)
        
        assert result['prediction'] in [0, 1]
        assert abs(result['probability'] - 0.2) < 0.01
        assert result['risk_level'] == "Bajo"
        assert isinstance(result['processing_time_ms'], int)
        assert 0 <= result['model_confidence'] <= 1
    
    def test_predict_medio_riesgo(self, mock_ml_model, mock_model_class, test_image):
        """Prueba de predicción con nivel de riesgo medio"""
        mock_ml_model.model = mock_model_class(0.5)
        result = mock_ml_model.predict(test_image)
        
        assert result['prediction'] in [0, 1]
        assert abs(result['probability'] - 0.5) < 0.01
        assert result['risk_level'] == "Medio"
        assert isinstance(result['processing_time_ms'], int)
        assert 0 <= result['model_confidence'] <= 1

    def test_predict_alto_riesgo(self, mock_ml_model, mock_model_class, test_image):
        """Prueba de predicción con nivel de riesgo alto"""
        mock_ml_model.model = mock_model_class(0.7)
        result = mock_ml_model.predict(test_image)
        
        assert result['prediction'] in [0, 1]
        assert abs(result['probability'] - 0.7) < 0.01
        assert result['risk_level'] == "Alto"
        assert isinstance(result['processing_time_ms'], int)
        assert 0 <= result['model_confidence'] <= 1

    def test_predict_critico_riesgo(self, mock_ml_model, mock_model_class, test_image):
        """Prueba de predicción con nivel de riesgo crítico"""
        mock_ml_model.model = mock_model_class(0.95)
        result = mock_ml_model.predict(test_image)
        
        assert result['prediction'] in [0, 1]
        assert abs(result['probability'] - 0.95) < 0.01
        assert result['risk_level'] == "Crítico"
        assert isinstance(result['processing_time_ms'], int)
        assert 0 <= result['model_confidence'] <= 1

    def test_predict_modelo_no_cargado(self, test_image):
        """Prueba de error cuando el modelo no está cargado"""
        pipeline = StrokeImagePipeline()
        pipeline.model_loaded = False
        
        with pytest.raises(Exception) as exc_info:
            pipeline.predict(test_image)
        
        assert "Modelo no cargado" in str(exc_info.value)

    def test_predict_processing_time(self, mock_ml_model, test_image):
        """Prueba del tiempo de procesamiento"""
        pipeline = mock_ml_model
        
        start_time = time.time()
        result = pipeline.predict(test_image)
        end_time = time.time()
        
        # Verificar que el tiempo de procesamiento en el resultado es cercano al real
        real_time = int((end_time - start_time) * 1000)
        assert abs(result['processing_time_ms'] - real_time) < 100  # tolerancia 100ms

    def test_predict_output_structure(self, mock_ml_model, test_image):
        """Prueba de la estructura de datos de salida"""
        result = mock_ml_model.predict(test_image)
        
        # Verificar presencia de todos los campos necesarios
        assert 'prediction' in result
        assert 'probability' in result
        assert 'risk_level' in result
        assert 'processing_time_ms' in result
        assert 'model_confidence' in result
        
        # Verificar tipos de datos
        assert isinstance(result['prediction'], int)
        assert isinstance(result['probability'], float)
        assert isinstance(result['risk_level'], str)
        assert isinstance(result['processing_time_ms'], int)
        assert isinstance(result['model_confidence'], float)
        
        # Verificar rangos de valores
        assert result['prediction'] in [0, 1]
        assert 0 <= result['probability'] <= 1
        assert result['risk_level'] in ["Bajo", "Medio", "Alto", "Crítico"]
        assert result['processing_time_ms'] >= 0
        assert 0 <= result['model_confidence'] <= 1
    
    def test_preprocess_image_from_bytes(self, mock_ml_model):
        """Prueba de preprocesamiento de imagen desde bytes"""
        # Crear imagen de prueba
        image = Image.new('RGB', (224, 224), color='red')
        img_byte_arr = io.BytesIO()
        image.save(img_byte_arr, format='PNG')
        img_byte_arr = img_byte_arr.getvalue()
        
        # Procesar imagen
        result = mock_ml_model._preprocess_image(img_byte_arr)
        
        # Verificar resultado
        assert isinstance(result, torch.Tensor)
        assert result.shape == (3, 224, 224)  # [canales, altura, ancho]
        assert result.dtype == torch.float32

    def test_preprocess_image_from_path(self, mock_ml_model, tmp_path):
        """Prueba de preprocesamiento de imagen desde ruta"""
        # Crear archivo temporal de imagen
        image = Image.new('RGB', (224, 224), color='blue')
        image_path = tmp_path / "test_image.png"
        image.save(image_path)
        
        # Procesar imagen
        result = mock_ml_model._preprocess_image(str(image_path))
        
        # Verificar resultado
        assert isinstance(result, torch.Tensor)
        assert result.shape == (3, 224, 224)
        assert result.dtype == torch.float32

    def test_preprocess_image_from_pil(self, mock_ml_model):
        """Prueba de preprocesamiento de imagen desde PIL Image"""
        # Crear PIL Image
        image = Image.new('RGB', (224, 224), color='green')
        
        # Procesar imagen
        result = mock_ml_model._preprocess_image(image)
        
        # Verificar resultado
        assert isinstance(result, torch.Tensor)
        assert result.shape == (3, 224, 224)
        assert result.dtype == torch.float32

    def test_preprocess_image_convert_to_rgb(self, mock_ml_model):
        """Prueba de conversión de imagen a RGB"""
        # Crear imagen en modo L (escala de grises)
        image = Image.new('L', (224, 224), color=128)
        
        # Procesar imagen
        result = mock_ml_model._preprocess_image(image)
        
        # Verificar resultado
        assert isinstance(result, torch.Tensor)
        assert result.shape == (3, 224, 224)  # Debe tener 3 canales después de la conversión a RGB
        assert result.dtype == torch.float32

    def test_preprocess_image_deterministic(self, mock_ml_model):
        """Prueba de determinismo en el preprocesamiento"""
        # Crear imagen de prueba
        image = Image.new('RGB', (224, 224), color='red')
        
        # Procesar imagen dos veces
        result1 = mock_ml_model._preprocess_image(image)
        result2 = mock_ml_model._preprocess_image(image)
        
        # Verificar que los resultados son idénticos
        assert torch.allclose(result1, result2)

    def test_preprocess_image_invalid_input(self, mock_ml_model):
        """Prueba de procesamiento de entrada inválida"""
        # Crear datos inválidos
        invalid_data = 'datos de imagen inválidos'
        
        # Verificar que la función lanza una excepción
        with pytest.raises(Exception):
            mock_ml_model._preprocess_image(invalid_data)