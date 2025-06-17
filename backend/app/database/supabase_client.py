"""Cliente Supabase para operaciones de base de datos"""

import os
import psycopg2
from dotenv import load_dotenv
from typing import List, Dict, Optional

load_dotenv()

async def save_stroke_prediction(prediction_data: Dict) -> bool:
    """Guardar predicción de stroke"""
    try:
        conn = psycopg2.connect(os.getenv("DATABASE_URL"))
        cursor = conn.cursor()
        
        insert_sql = """
            INSERT INTO stroke_predictions 
            (gender, age, hypertension, heart_disease, ever_married, work_type, 
             residence_type, avg_glucose_level, bmi, smoking_status, 
             prediction, probability, risk_level)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        """
        
        cursor.execute(insert_sql, (
            prediction_data['gender'], prediction_data['age'], prediction_data['hypertension'],
            prediction_data['heart_disease'], prediction_data['ever_married'], prediction_data['work_type'],
            prediction_data['residence_type'], prediction_data['avg_glucose_level'], prediction_data.get('bmi'),
            prediction_data['smoking_status'], prediction_data['prediction'], 
            prediction_data['probability'], prediction_data['risk_level']
        ))
        
        conn.commit()
        cursor.close()
        conn.close()
        return True
        
    except Exception as e:
        print(f"Error en base de datos: {e}")
        return False

async def get_stroke_predictions(limit: int = 50) -> List[Dict]:
    """Obtener predicciones de stroke"""
    try:
        conn = psycopg2.connect(os.getenv("DATABASE_URL"))
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT id, fecha_creacion, gender, age, hypertension, heart_disease, 
                   ever_married, work_type, residence_type, avg_glucose_level, 
                   bmi, smoking_status, prediction, probability, risk_level
            FROM stroke_predictions_formatted 
            ORDER BY id DESC LIMIT %s
        """, (limit,))
        
        columns = ['id', 'fecha_creacion', 'gender', 'age', 'hypertension', 'heart_disease',
                  'ever_married', 'work_type', 'residence_type', 'avg_glucose_level',
                  'bmi', 'smoking_status', 'prediction', 'probability', 'risk_level']
        
        predictions = [dict(zip(columns, row)) for row in cursor.fetchall()]
        
        cursor.close()
        conn.close()
        return predictions
        
    except Exception as e:
        print(f"Error en base de datos: {e}")
        return []

async def save_image_prediction(prediction_data: Dict) -> bool:
    """Guardar predicción de imagen"""
    try:
        conn = psycopg2.connect(os.getenv("DATABASE_URL"))
        cursor = conn.cursor()
        
        cursor.execute("""
            INSERT INTO image_predictions 
            (image_filename, image_url, image_size, image_format,
             prediction, probability, risk_level, model_confidence, 
             processing_time_ms, stroke_prediction_id)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        """, (
            prediction_data['image_filename'], prediction_data['image_url'],
            prediction_data['image_size'], prediction_data['image_format'],
            prediction_data['prediction'], prediction_data['probability'],
            prediction_data['risk_level'], prediction_data['model_confidence'],
            prediction_data['processing_time_ms'], prediction_data['stroke_prediction_id']
        ))
        
        conn.commit()
        cursor.close()
        conn.close()
        return True
        
    except Exception as e:
        print(f"Error guardando imagen: {e}")
        return False

async def get_image_predictions(stroke_prediction_id: Optional[int] = None, limit: int = 50) -> List[Dict]:
    """Obtener predicciones de imagen"""
    try:
        conn = psycopg2.connect(os.getenv("DATABASE_URL"))
        cursor = conn.cursor()
        
        if stroke_prediction_id:
            cursor.execute("""
                SELECT id, fecha_creacion, image_filename, image_url, image_size, 
                       image_format, prediction, probability, risk_level, 
                       model_confidence, processing_time_ms, stroke_prediction_id
                FROM image_predictions_formatted 
                WHERE stroke_prediction_id = %s ORDER BY id DESC LIMIT %s
            """, (stroke_prediction_id, limit))
        else:
            cursor.execute("""
                SELECT id, fecha_creacion, image_filename, image_url, image_size, 
                       image_format, prediction, probability, risk_level, 
                       model_confidence, processing_time_ms, stroke_prediction_id
                FROM image_predictions_formatted 
                ORDER BY id DESC LIMIT %s
            """, (limit,))
        
        columns = ['id', 'fecha_creacion', 'image_filename', 'image_url', 'image_size',
                  'image_format', 'prediction', 'probability', 'risk_level',
                  'model_confidence', 'processing_time_ms', 'stroke_prediction_id']
        
        predictions = [dict(zip(columns, row)) for row in cursor.fetchall()]
        
        cursor.close()
        conn.close()
        return predictions
        
    except Exception as e:
        print(f"Error obteniendo imágenes: {e}")
        return []

async def get_combined_predictions(limit: int = 50) -> List[Dict]:
    """Obtener predicciones combinadas (stroke + imagen)"""
    try:
        conn = psycopg2.connect(os.getenv("DATABASE_URL"))
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT 
                sp.id, sp.fecha_creacion, sp.gender, sp.age, sp.hypertension, sp.heart_disease,
                sp.ever_married, sp.work_type, sp.residence_type, sp.avg_glucose_level,
                sp.bmi, sp.smoking_status, sp.prediction as stroke_prediction,
                sp.probability as stroke_probability, sp.risk_level as stroke_risk_level,
                ip.id as image_id, ip.prediction as image_prediction,
                ip.probability as image_probability, ip.risk_level as image_risk_level,
                ip.model_confidence as image_confidence, ip.image_filename, ip.image_url,
                ip.image_size, ip.image_format, ip.processing_time_ms as image_processing_time,
                CASE WHEN ip.id IS NOT NULL THEN true ELSE false END as has_image
            FROM stroke_predictions_formatted sp
            LEFT JOIN image_predictions ip ON sp.id = ip.stroke_prediction_id
            ORDER BY sp.id DESC LIMIT %s
        """, (limit,))
        
        columns = ['id', 'fecha_creacion', 'gender', 'age', 'hypertension', 'heart_disease',
                  'ever_married', 'work_type', 'residence_type', 'avg_glucose_level',
                  'bmi', 'smoking_status', 'stroke_prediction', 'stroke_probability', 'stroke_risk_level',
                  'image_id', 'image_prediction', 'image_probability', 'image_risk_level', 'image_confidence',
                  'image_filename', 'image_url', 'image_size', 'image_format', 'image_processing_time', 'has_image']
        
        combined_predictions = []
        for row in cursor.fetchall():
            prediction = dict(zip(columns, row))
            
            stroke_data = {k: prediction[k] for k in columns[:15]}
            stroke_data['prediction'] = prediction['stroke_prediction']
            stroke_data['probability'] = prediction['stroke_probability']
            stroke_data['risk_level'] = prediction['stroke_risk_level']
            
            image_data = None
            if prediction['has_image']:
                image_data = {
                    'id': prediction['image_id'],
                    'prediction': prediction['image_prediction'],
                    'probability': prediction['image_probability'],
                    'risk_level': prediction['image_risk_level'],
                    'model_confidence': prediction['image_confidence'],
                    'image_filename': prediction['image_filename'],
                    'image_url': prediction['image_url'],
                    'image_size': prediction['image_size'],
                    'image_format': prediction['image_format'],
                    'processing_time_ms': prediction['image_processing_time']
                }
            
            combined_predictions.append({
                'stroke_data': stroke_data,
                'image_data': image_data,
                'has_image': prediction['has_image']
            })
        
        cursor.close()
        conn.close()
        return combined_predictions
        
    except Exception as e:
        print(f"Error obteniendo predicciones combinadas: {e}")
        return []

async def test_db_connection() -> bool:
    """Probar conexión a base de datos"""
    try:
        conn = psycopg2.connect(os.getenv("DATABASE_URL"))
        cursor = conn.cursor()
        cursor.execute("SELECT 1")
        cursor.fetchone()
        cursor.close()
        conn.close()
        return True
    except Exception:
        return False