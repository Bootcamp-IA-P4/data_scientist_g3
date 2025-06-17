"""Endpoints de predicción para stroke y clasificación de imágenes"""

from fastapi import APIRouter, HTTPException, UploadFile, File
from datetime import datetime
from typing import List

from models.schemas import (
    StrokeRequest,
    StrokeResponse,
    HealthResponse,
    StrokePredictionsList,
    ImageUploadInfo,
    ImageResponse,
    ImagePredictionsList
)
from services.stroke_service import predict_stroke, get_all_predictions, get_pipeline_status
from services.image_service import (
    process_image_prediction, 
    get_images_for_stroke, 
    get_all_image_predictions,
    get_image_pipeline_status
)
from database.supabase_client import test_db_connection

router = APIRouter()

# ====ENDPOINTS STROKE ====

@router.post("/predict/stroke", response_model=StrokeResponse)
async def predict_stroke_endpoint(request: StrokeRequest):
    """Realizar predicción de stroke y guardar en base de datos"""
    try:
        result = await predict_stroke(request)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error en predicción: {str(e)}")

@router.get("/predictions/stroke", response_model=StrokePredictionsList)
async def get_stroke_predictions():
    """Obtener todas las predicciones de stroke para tabla del frontend"""
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

# ==== ENDPOINTS IMAGEN ====

@router.get("/image/upload-info", response_model=ImageUploadInfo)
async def get_image_upload_info():
    """Obtener restricciones y validaciones para upload de imagen"""
    return ImageUploadInfo()

@router.post("/predict/image/{stroke_prediction_id}", response_model=ImageResponse)
async def predict_image_endpoint(stroke_prediction_id: int, image: UploadFile = File(...)):
    """
    Procesar imagen para predicción de stroke vinculada a predicción existente
    
    ⚠️ Consultar GET /image/upload-info para ver restricciones:
    - Tamaño máximo: 10MB
    - Dimensiones: 32x32 a 4096x4096 píxeles  
    - Formatos: JPEG, PNG, WEBP, BMP
    - Tipo: Tomografía computarizada del cerebro en escala de grises
    """
    try:
        # Validar formato de archivo
        if not image.content_type.startswith('image/'):
            raise HTTPException(status_code=400, detail="Archivo debe ser una imagen")
        
        # Leer datos de imagen
        image_data = await image.read()
        
        # Procesar con CNN
        result = await process_image_prediction(
            image_data=image_data,
            stroke_prediction_id=stroke_prediction_id,
            filename=image.filename,
            content_type=image.content_type
        )
        
        # Verificar si hay error
        if 'error' in result:
            raise HTTPException(status_code=400, detail=result['error'])
        
        return result
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error procesando imagen: {str(e)}")

@router.get("/predictions/images", response_model=ImagePredictionsList)
async def get_image_predictions():
    """Obtener todas las predicciones de imagen para combinar con stroke en frontend"""
    try:
        predictions = await get_all_image_predictions(limit=50)
        return ImagePredictionsList(
            predictions=predictions,
            total=len(predictions),
            message="Predicciones de imagen obtenidas correctamente"
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error obteniendo imágenes: {str(e)}")

# ==== ENDPOINTS ESTADO/SALUD ====

@router.get("/pipeline/status")
async def get_pipeline_status_endpoint():
    """Verificar estado de ambos pipelines (stroke + imagen)"""
    try:
        stroke_status = await get_pipeline_status()
        image_status = await get_image_pipeline_status()
        
        return {
            "stroke_pipeline": stroke_status,
            "image_pipeline": image_status,
            "timestamp": datetime.now().isoformat(),
            "both_ready": stroke_status.get('is_loaded', False) and image_status.get('pipeline_loaded', False)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error obteniendo estado: {str(e)}")

@router.get("/health", response_model=HealthResponse)
async def health_check():
    """Verificación de salud completa de la API"""
    db_connected = await test_db_connection()
    
    # Verificar ambos pipelines
    try:
        stroke_status = await get_pipeline_status()
        stroke_healthy = stroke_status.get('is_loaded', False)
    except:
        stroke_healthy = False
    
    try:
        image_status = await get_image_pipeline_status()
        image_healthy = image_status.get('pipeline_loaded', False)
    except:
        image_healthy = False
    
    overall_status = "healthy" if (db_connected and stroke_healthy and image_healthy) else "unhealthy"
    
    return HealthResponse(
        status=overall_status,
        timestamp=datetime.now(),
        version="1.0.0",
        database_connected=db_connected
    )