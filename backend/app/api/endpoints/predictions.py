"""Endpoints de predicción para stroke y clasificación de imágenes"""

from fastapi import APIRouter, HTTPException
from datetime import datetime
from typing import List

from models.schemas import (
    StrokeRequest,
    StrokeResponse,
    HealthResponse,
    StrokePredictionsList
)
from services.stroke_service import predict_stroke, get_all_predictions
from database.supabase_client import test_db_connection

router = APIRouter()

# Endpoint de predicción de stroke
@router.post("/predict/stroke", response_model=StrokeResponse)
async def predict_stroke_endpoint(request: StrokeRequest):
    """Realizar predicción de stroke y guardar en base de datos"""
    try:
        result = await predict_stroke(request)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error en predicción: {str(e)}")

# Endpoint de predicción de imágenes (placeholder)
@router.post("/predict/image")
async def predict_image_endpoint():
    """Clasificación de imágenes - Próximamente"""
    raise HTTPException(
        status_code=501,
        detail="Predicción de imágenes aún no implementada. Llegará en el próximo sprint."
    )

# Obtener predicciones de stroke
@router.get("/predictions/stroke", response_model=StrokePredictionsList)
async def get_stroke_predictions():
    """Obtener todas las predicciones de stroke de la base de datos"""
    try:
        predictions = await get_all_predictions()
        return StrokePredictionsList(
            predictions=predictions,
            total=len(predictions),
            page=1,
            limit=len(predictions)
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error de base de datos: {str(e)}")

# Obtener predicciones de imágenes (placeholder)
@router.get("/predictions/images")
async def get_image_predictions():
    """Obtener predicciones de imágenes - Próximamente"""
    raise HTTPException(
        status_code=501,
        detail="Predicciones de imágenes aún no implementadas. Llegarán en el próximo sprint."
    )

# Endpoint de health check
@router.get("/health", response_model=HealthResponse)
async def health_check():
    """Verificación de salud de la API"""
    db_connected = await test_db_connection()
    
    return HealthResponse(
        status="healthy" if db_connected else "unhealthy",
        timestamp=datetime.now(),
        version="1.0.0",
        database_connected=db_connected
    )