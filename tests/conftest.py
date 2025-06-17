import pytest
import torch
from PIL import Image
import io
import os
import sys
import json
from pathlib import Path
from src.pipeline.image_pipeline import StrokeImagePipeline
from src.pipeline.stroke_pipeline import StrokePipeline

# Añadir ruta al código fuente
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Fixtures para datos de prueba
@pytest.fixture(scope="session")
def test_data():
    """Carga de datos de prueba"""
    data_path = Path(__file__).parent / "fixtures" / "test_data.json"
    with open(data_path) as f:
        return json.load(f)

@pytest.fixture(scope="session")
def test_image():
    """Fixture para imagen de prueba"""
    # Crear imagen
    img = Image.new('RGB', (224, 224))
    
    # Guardar en bytes
    img_byte_arr = io.BytesIO()
    img.save(img_byte_arr, format='PNG')
    img_byte_arr.seek(0)
    
    return img_byte_arr.getvalue()

@pytest.fixture
def mock_model_class():
    """Fixture para clase modelo mock"""
    class MockModel(torch.nn.Module):
        def __init__(self, target_probability):
            super().__init__()
            # Calcular logits para cualquier probabilidad objetivo
            # Usar fórmula: log(p/(1-p))
            logit = torch.log(torch.tensor(target_probability / (1 - target_probability)))
            self.logits = torch.tensor([0.0, logit], dtype=torch.float32)
            
        def forward(self, x):
            return self.logits.unsqueeze(0)  # Añadir dimensión de batch
    
    return MockModel

@pytest.fixture
def mock_ml_model(mock_model_class):
    """Fixture para pipeline con modelo mock"""
    pipeline = StrokeImagePipeline()
    pipeline.model_loaded = True
    pipeline.device = torch.device('cpu')
    return pipeline

@pytest.fixture(scope="session")
def test_db():
    """Fixture para base de datos de prueba"""
    # Configuración de BD de prueba
    return {
        "url": "test_db_url",
        "key": "test_key"
    }

# Fixtures para pruebas de API
@pytest.fixture(scope="function")
def test_client():
    """Fixture para cliente de prueba de API"""
    from fastapi.testclient import TestClient
    from src.app import app
    return TestClient(app)

@pytest.fixture
def valid_patient_data():
    """Fixture con datos válidos de paciente para pruebas"""
    return {
        'gender': "Masculino",
        'age': 65,
        'hypertension': "Sí",
        'heart_disease': "No",
        'ever_married': "Sí",
        'work_type': "Privado",
        'residence_type': "Urbano",
        'avg_glucose_level': 180,
        'bmi': 28.5,
        'smoking_status': "Fumó antes"
    }

@pytest.fixture
def stroke_pipeline():
    """Fixture para crear instancia de StrokePipeline"""
    return StrokePipeline()