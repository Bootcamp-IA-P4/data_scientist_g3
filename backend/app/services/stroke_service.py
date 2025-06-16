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
    
    # Mostrar factores de riesgo si están disponibles (del modelo XGBoost)
    if 'feature_importance' in prediction_result:
        print(f"📊 Factores de riesgo principales:")
        for factor in prediction_result['feature_importance']['top_risk_factors'][:3]:
            print(f"   • {factor['label']}: {factor['contribution_percentage']}%")
    
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
    # PIPELINE ACTIVADO - Usar modelo XGBoost
    try:
        # Agregar la ruta raíz del proyecto al sys.path para encontrar src/
        import sys
        from pathlib import Path
        
        # Obtener la ruta raíz del proyecto (subir 3 niveles desde backend/app/services)
        current_dir = Path(__file__).resolve().parent
        project_root = current_dir.parent.parent.parent  # services -> app -> backend -> proyecto
        
        # Agregar al sys.path si no está
        if str(project_root) not in sys.path:
            sys.path.insert(0, str(project_root))
        
        # Importar el pipeline
        from src.pipeline import predict_stroke_risk
        
        pipeline_data = {
            'gender': request.gender,
            'age': request.age,
            'hypertension': request.hypertension,
            'heart_disease': request.heart_disease,
            'ever_married': request.ever_married,
            'work_type': request.work_type,
            'residence_type': request.residence_type,
            'avg_glucose_level': request.avg_glucose_level,
            'bmi': request.bmi,
            'smoking_status': request.smoking_status
        }
        
        result = predict_stroke_risk(pipeline_data)
        print(f"🤖 Pipeline XGBoost ejecutado: predicción={result['prediction']}, probabilidad={result['probability']:.4f}")
        return result
        
    except ImportError as e:
        # Si no encuentra el pipeline, fallar con error claro
        error_msg = f"❌ Pipeline no disponible: {str(e)}. Verificar que existe src/pipeline/stroke_pipeline.py en la raíz del proyecto"
        print(error_msg)
        raise Exception(error_msg)
        
    except Exception as e:
        # Si hay error en el pipeline, fallar para debugging
        error_msg = f"❌ Error en pipeline XGBoost: {str(e)}"
        print(error_msg)
        raise Exception(error_msg)

# MOCK COMENTADO PARA FORZAR USO DEL PIPELINE
# def _mock_prediction(request: StrokeRequest) -> Dict:
#     """
#     Predicción simulada para testing - lógica realista basada en factores médicos
#     COMENTADO: Para forzar el uso del pipeline XGBoost
#     """
#     risk_score = 0.0
#     
#     # Factor edad (predictor más fuerte)
#     if request.age > 75:
#         risk_score += 0.4
#     elif request.age > 65:
#         risk_score += 0.25
#     elif request.age > 50:
#         risk_score += 0.15
#     elif request.age > 35:
#         risk_score += 0.05
#     
#     # Historial médico (alto impacto)
#     if request.hypertension == "Sí":
#         risk_score += 0.2
#     if request.heart_disease == "Sí":
#         risk_score += 0.25
#     
#     # Nivel de glucosa (indicador de diabetes)
#     if request.avg_glucose_level > 200:
#         risk_score += 0.2
#     elif request.avg_glucose_level > 140:
#         risk_score += 0.1
#     
#     # Factor BMI
#     if request.bmi:
#         if request.bmi > 35:  # Obesidad severa
#             risk_score += 0.15
#         elif request.bmi > 30:  # Obesidad
#             risk_score += 0.1
#         elif request.bmi < 18.5:  # Bajo peso
#             risk_score += 0.05
#     
#     # Estado de fumador
#     if request.smoking_status == "Fuma":
#         risk_score += 0.15
#     elif request.smoking_status == "Fumó antes":
#         risk_score += 0.08
#     
#     # Factor género (leve)
#     if request.gender == "Masculino" and request.age > 45:
#         risk_score += 0.05
#     elif request.gender == "Femenino" and request.age > 55:
#         risk_score += 0.05
#     
#     # Estado civil (factor social)
#     if request.ever_married == "No" and request.age > 40:
#         risk_score += 0.03
#     
#     # Asegurar que la probabilidad sea realista (entre 0.01 y 0.95)
#     probability = max(0.01, min(risk_score, 0.95))
#     prediction = 1 if probability > 0.5 else 0
#     
#     return {
#         'prediction': prediction,
#         'probability': round(probability, 4)
#     }

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
        'hypertension': request.hypertension,  
        'heart_disease': request.heart_disease, 
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

# Función adicional para verificar el estado del pipeline
async def get_pipeline_status() -> Dict:
    """
    Verifica el estado del pipeline de predicción
    
    Returns:
        Dict: Información sobre el estado del pipeline
    """
    try:
        # Agregar la ruta raíz del proyecto al sys.path
        import sys
        from pathlib import Path
        
        current_dir = Path(__file__).resolve().parent
        project_root = current_dir.parent.parent.parent  # services -> app -> backend -> proyecto
        
        if str(project_root) not in sys.path:
            sys.path.insert(0, str(project_root))
        
        from src.pipeline import get_pipeline_status
        return get_pipeline_status()
    except ImportError:
        return {
            'is_loaded': False,
            'model_type': None,
            'status': 'Pipeline no disponible - Verificar src/pipeline/ en la raíz del proyecto'
        }