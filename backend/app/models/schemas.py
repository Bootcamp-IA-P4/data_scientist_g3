"""Modelos Pydantic para validación de API y documentación"""

from pydantic import BaseModel, Field
from typing import Literal, Optional, List
from datetime import datetime

# ==== MODELOS STROKE ====

class StrokeRequest(BaseModel):
    """Datos del frontend para predicción de stroke"""
    
    gender: Literal["Masculino", "Femenino", "Otro"] = Field(..., description="Género del paciente")
    age: int = Field(..., ge=0, le=120, description="Edad en años")
    hypertension: Literal["Sí", "No"] = Field(..., description="¿Tiene hipertensión?")
    heart_disease: Literal["Sí", "No"] = Field(..., description="¿Tiene enfermedad cardíaca?")
    ever_married: Literal["Sí", "No"] = Field(..., description="¿Alguna vez se ha casado?")
    work_type: Literal["Empleado Público", "Privado", "Autónomo", "Niño", "Nunca trabajó"] = Field(..., description="Tipo de trabajo")
    residence_type: Literal["Urbano", "Rural"] = Field(..., description="Tipo de residencia")
    avg_glucose_level: float = Field(..., ge=50.0, le=500.0, description="Nivel de glucosa (mg/dL)")
    bmi: Optional[float] = Field(None, ge=10.0, le=60.0, description="Índice de masa corporal")
    smoking_status: Literal["Nunca fumó", "Fuma", "Fumó antes", "NS/NC"] = Field(..., description="Estado de fumador")

    class Config:
        json_schema_extra = {
            "example": {
                "gender": "Masculino",
                "age": 45,
                "hypertension": "No",
                "heart_disease": "No",
                "ever_married": "Sí",
                "work_type": "Privado",
                "residence_type": "Urbano",
                "avg_glucose_level": 120.5,
                "bmi": 25.3,
                "smoking_status": "Nunca fumó"
            }
        }

class StrokeResponse(BaseModel):
    """Respuesta de la API con predicción de stroke"""
    
    prediction: Literal[0, 1] = Field(..., description="0=Sin riesgo, 1=Riesgo de stroke")
    probability: float = Field(..., ge=0.0, le=1.0, description="Probabilidad de stroke")
    risk_level: Literal["Bajo", "Medio", "Alto", "Crítico"] = Field(..., description="Nivel de riesgo")
    processing_time_ms: Optional[int] = Field(None, description="Tiempo de procesamiento en milisegundos")

# ==== MODELOS IMAGEN ====

class ImageUploadInfo(BaseModel):
    """Información de restricciones para upload de imagen (solo documentación)"""
    
    max_file_size_mb: int = Field(10, description="Tamaño máximo archivo en MB")
    min_dimensions: int = Field(32, description="Dimensiones mínimas en píxeles (ancho x alto)")
    max_dimensions: int = Field(4096, description="Dimensiones máximas en píxeles (ancho x alto)")
    supported_formats: List[str] = Field(["JPEG", "PNG", "WEBP", "BMP"], description="Formatos de imagen soportados")
    content_types: List[str] = Field(["image/jpeg", "image/png", "image/webp", "image/bmp"], description="Content-Types válidos")

    class Config:
        json_schema_extra = {
            "example": {
                "max_file_size_mb": 10,
                "min_dimensions": 32,
                "max_dimensions": 4096,
                "supported_formats": ["JPEG", "PNG", "WEBP", "BMP"],
                "content_types": ["image/jpeg", "image/png", "image/webp", "image/bmp"],
                "note": "Subir imagen de tomografía computarizada del cerebro en escala de grises"
            }
        }

class ImageResponse(BaseModel):
    """Respuesta de la API con predicción de imagen"""
    
    prediction: Literal[0, 1] = Field(..., description="0=Normal, 1=Stroke detectado")
    probability: float = Field(..., ge=0.0, le=1.0, description="Probabilidad de stroke")
    risk_level: Literal["Bajo", "Medio", "Alto", "Crítico"] = Field(..., description="Nivel de riesgo")
    model_confidence: float = Field(..., ge=0.0, le=1.0, description="Confianza del modelo CNN")
    processing_time_ms: int = Field(..., description="Tiempo de procesamiento en milisegundos")
    stroke_prediction_id: int = Field(..., description="ID de la predicción de stroke asociada")
    message: str = Field(..., description="Mensaje de confirmación")

    class Config:
        json_schema_extra = {
            "example": {
                "prediction": 1,
                "probability": 0.8234,
                "risk_level": "Alto",
                "model_confidence": 0.9123,
                "processing_time_ms": 1845,
                "stroke_prediction_id": 123,
                "message": "Imagen procesada correctamente"
            }
        }

class ImagePredictionDB(BaseModel):
    """Modelo de base de datos para predicciones de imagen"""
    
    id: Optional[int] = None
    fecha_creacion: Optional[str] = None
    stroke_prediction_id: int
    image_filename: str
    image_url: str
    image_size: Optional[int] = None
    image_format: Optional[str] = None
    prediction: int
    probability: float
    risk_level: str
    model_confidence: Optional[float] = None
    processing_time_ms: Optional[int] = None

    class Config:
        arbitrary_types_allowed = True

class ImagePredictionsList(BaseModel):
    """Lista de predicciones de imagen para combinar con stroke en frontend"""
    
    predictions: List[ImagePredictionDB] = Field(..., description="Lista de predicciones de imagen")
    total: int = Field(..., description="Número total de predicciones")
    message: str = Field(..., description="Mensaje de estado")

    class Config:
        json_schema_extra = {
            "example": {
                "predictions": [
                    {
                        "id": 1,
                        "fecha_creacion": "15/06/2025 14:30",
                        "stroke_prediction_id": 123,
                        "prediction": 1,
                        "probability": 0.82,
                        "risk_level": "Alto",
                        "image_filename": "scan_brain.jpg",
                        "image_url": "https://storage.supabase.co/..."
                    }
                ],
                "total": 1,
                "message": "Predicciones de imagen obtenidas correctamente"
            }
        }

# ==== MODELOS BASE DE DATOS ====

class StrokePredictionDB(BaseModel):
    """Modelo de base de datos para guardar predicciones"""
    
    id: Optional[int] = None
    fecha_creacion: Optional[str] = None
    gender: str
    age: int
    hypertension: str
    heart_disease: str
    ever_married: str
    work_type: str
    residence_type: str
    avg_glucose_level: float
    bmi: Optional[float]
    smoking_status: str
    prediction: int
    probability: float
    risk_level: str
    processing_time_ms: Optional[int] = None

    class Config:
        arbitrary_types_allowed = True

class StrokePredictionsList(BaseModel):
    """Lista de predicciones de stroke"""
    
    predictions: List[StrokePredictionDB] = Field(..., description="Lista de predicciones")
    total: int = Field(..., description="Número total de predicciones")
    page: int = Field(default=1, description="Página actual")
    limit: int = Field(default=10, description="Elementos por página")

# ==== MODELOS GENERALES ====

class HealthResponse(BaseModel):
    """Respuesta del health check"""
    
    status: Literal["healthy", "unhealthy"] = Field(..., description="Estado de la API")
    timestamp: datetime = Field(..., description="Timestamp de la verificación")
    version: str = Field(default="1.0.0", description="Versión de la API")
    database_connected: bool = Field(..., description="Estado de conexión a base de datos")