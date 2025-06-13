"""Modelos Pydantic para validación de API y documentación"""

from pydantic import BaseModel, Field
from typing import Literal, Optional, List
from datetime import datetime

# Modelos para predicción de stroke
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

# Modelos de respuesta general
class HealthResponse(BaseModel):
    """Respuesta del health check"""
    
    status: Literal["healthy", "unhealthy"] = Field(..., description="Estado de la API")
    timestamp: datetime = Field(..., description="Timestamp de la verificación")
    version: str = Field(default="1.0.0", description="Versión de la API")
    database_connected: bool = Field(..., description="Estado de conexión a base de datos")

class StrokePredictionsList(BaseModel):
    """Lista de predicciones de stroke"""
    
    predictions: List[StrokePredictionDB] = Field(..., description="Lista de predicciones")
    total: int = Field(..., description="Número total de predicciones")
    page: int = Field(default=1, description="Página actual")
    limit: int = Field(default=10, description="Elementos por página")