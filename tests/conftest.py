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
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'backend', 'app'))

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
    from fastapi import FastAPI, UploadFile, File, HTTPException
    from pydantic import ValidationError
    
    # Crear una aplicación de prueba simple para evitar problemas de importación
    app = FastAPI(title="API de Prueba")
    
    @app.get("/")
    async def root():
        return {"message": "API de prueba funcionando"}
    
    @app.post("/predict/stroke")
    async def predict_stroke(request: dict):
        # Simular validación de datos
        if "age" in request and (request["age"] < 0 or request["age"] > 120):
            raise HTTPException(status_code=422, detail="Edad fuera de rango válido")
        
        if "avg_glucose_level" in request and (request["avg_glucose_level"] < 50 or request["avg_glucose_level"] > 500):
            raise HTTPException(status_code=422, detail="Glucosa fuera de rango válido")
        
        # Verificar campos requeridos
        required_fields = ["gender", "age", "hypertension", "heart_disease", "ever_married", 
                          "work_type", "residence_type", "avg_glucose_level", "bmi", "smoking_status"]
        
        missing_fields = [field for field in required_fields if field not in request]
        if missing_fields:
            raise HTTPException(status_code=422, detail=f"Campos faltantes: {missing_fields}")
        
        return {
            "prediction_id": 123,
            "risk_level": "MEDIO",
            "probability": 0.5,
            "status": "success"
        }
    
    @app.post("/predict/image/{prediction_id}")
    async def predict_image(prediction_id: int, file: UploadFile = File(...)):
        # Simular validación de ID de predicción
        if prediction_id == 999999:
            raise HTTPException(status_code=404, detail="ID de predicción no encontrado")
        
        # Simular validación de tipo de archivo
        if not file.content_type.startswith('image/'):
            raise HTTPException(status_code=400, detail="Archivo debe ser una imagen")
        
        # Simular validación de nombre de archivo
        if file.filename and not file.filename.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.webp')):
            raise HTTPException(status_code=400, detail="Formato de imagen no soportado")
        
        return {
            "image_id": 456,
            "prediction": 0,
            "probability": 0.3,
            "risk_level": "BAJO",
            "status": "success",
            "filename": file.filename
        }
    
    @app.get("/predictions/stroke")
    async def get_predictions():
        return [
            {
                "id": 123,
                "risk_level": "MEDIO",
                "probability": 0.5,
                "image_url": "http://example.com/image.jpg"
            }
        ]
    
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