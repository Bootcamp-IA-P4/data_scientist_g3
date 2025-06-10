"""Lógica de negocio para predicción de stroke"""

import time
from typing import Dict, List

from models.schemas import StrokeRequest, StrokeResponse, StrokePredictionDB
from database.supabase_client import save_stroke_prediction, get_stroke_predictions

async def predict_stroke(request: StrokeRequest) -> StrokeResponse:
    """
    Procesa una solicitud de predicción de stroke y la guarda en base de datos
    
    Args:
        request: Datos de la solicitud de predicción
        
    Returns:
        StrokeResponse: Resultados de la predicción
    """
    start_time = time.time()
    
    # Obtener predicción del pipeline
    prediction_result = await _call_stroke_pipeline(request)
    risk_level = _calculate_risk_level(prediction_result['probability'])
    
    # Calcular tiempo de procesamiento
    processing_time = int((time.time() - start_time) * 1000)
    
    # Crear respuesta
    response = StrokeResponse(
        prediction=prediction_result['prediction'],
        probability=prediction_result['probability'],
        risk_level=risk_level,
        processing_time_ms=processing_time
    )
    
    # Guardar en base de datos
    await _save_prediction_to_db(request, response, processing_time)
    
    return response

async def get_all_predictions() -> List[StrokePredictionDB]:
    """Obtiene todas las predicciones de stroke de la base de datos"""
    predictions = await get_stroke_predictions()
    return [StrokePredictionDB(**pred) for pred in predictions]

async def _call_stroke_pipeline(request: StrokeRequest) -> Dict:
    """
    Llama al pipeline de predicción de stroke
    
    Args:
        request: Datos de la solicitud de predicción
        
    Returns:
        Dict: Resultado del pipeline con predicción y probabilidad
    """
    # MOCK TEMPORAL - Reemplazar con pipeline cuando esté listo
    return _mock_prediction(request)
    
    # TODO: Descomentar cuando el pipeline esté listo
    # try:
    #     from src.pipeline.stroke_pipeline import predict_stroke_risk
    #     
    #     pipeline_data = {
    #         'gender': request.gender,
    #         'age': request.age,
    #         'hypertension': request.hypertension == "Sí",
    #         'heart_disease': request.heart_disease == "Sí",
    #         'ever_married': request.ever_married,
    #         'work_type': request.work_type,
    #         'residence_type': request.residence_type,
    #         'avg_glucose_level': request.avg_glucose_level,
    #         'bmi': request.bmi,
    #         'smoking_status': request.smoking_status
    #     }
    #     
    #     result = predict_stroke_risk(pipeline_data)
    #     return result
    #     
    # except ImportError:
    #     raise Exception("Pipeline no disponible. Implementar src.pipeline.stroke_pipeline")
    # except Exception as e:
    #     raise Exception(f"Error en pipeline: {str(e)}")

def _mock_prediction(request: StrokeRequest) -> Dict:
    """
    Predicción simulada para testing - lógica realista basada en factores médicos
    """
    risk_score = 0.0
    
    # Factor edad (predictor más fuerte)
    if request.age > 75:
        risk_score += 0.4
    elif request.age > 65:
        risk_score += 0.25
    elif request.age > 50:
        risk_score += 0.15
    elif request.age > 35:
        risk_score += 0.05
    
    # Historial médico (alto impacto)
    if request.hypertension == "Sí":
        risk_score += 0.2
    if request.heart_disease == "Sí":
        risk_score += 0.25
    
    # Nivel de glucosa (indicador de diabetes)
    if request.avg_glucose_level > 200:
        risk_score += 0.2
    elif request.avg_glucose_level > 140:
        risk_score += 0.1
    
    # Factor BMI
    if request.bmi:
        if request.bmi > 35:  # Obesidad severa
            risk_score += 0.15
        elif request.bmi > 30:  # Obesidad
            risk_score += 0.1
        elif request.bmi < 18.5:  # Bajo peso
            risk_score += 0.05
    
    # Estado de fumador
    if request.smoking_status == "Fuma":
        risk_score += 0.15
    elif request.smoking_status == "Fumó antes":
        risk_score += 0.08
    
    # Factor género (leve)
    if request.gender == "Masculino" and request.age > 45:
        risk_score += 0.05
    elif request.gender == "Femenino" and request.age > 55:
        risk_score += 0.05
    
    # Estado civil (factor social)
    if request.ever_married == "No" and request.age > 40:
        risk_score += 0.03
    
    # Asegurar que la probabilidad sea realista (entre 0.01 y 0.95)
    probability = max(0.01, min(risk_score, 0.95))
    prediction = 1 if probability > 0.5 else 0
    
    return {
        'prediction': prediction,
        'probability': round(probability, 4)
    }

def _calculate_risk_level(probability: float) -> str:
    """Calcula el nivel de riesgo basado en la probabilidad"""
    if probability < 0.3:
        return "Bajo"
    elif probability < 0.6:
        return "Medio"
    elif probability < 0.9:
        return "Alto"
    else:
        return "Crítico"

async def _save_prediction_to_db(request: StrokeRequest, response: StrokeResponse, processing_time: int):
    """Guarda la predicción en la base de datos"""
    prediction_data = {
        'gender': request.gender,
        'age': request.age,
        'hypertension': request.hypertension == "Sí",
        'heart_disease': request.heart_disease == "Sí",
        'ever_married': request.ever_married,
        'work_type': request.work_type,
        'residence_type': request.residence_type,
        'avg_glucose_level': request.avg_glucose_level,
        'bmi': request.bmi,
        'smoking_status': request.smoking_status,
        'prediction': response.prediction,
        'probability': response.probability,
        'risk_level': response.risk_level,
    }
    
    await save_stroke_prediction(prediction_data)