"""Cliente Supabase para operaciones de base de datos"""

import os
import psycopg2
from dotenv import load_dotenv
from typing import List, Dict
from datetime import datetime

# Cargar variables de entorno
load_dotenv()

async def save_stroke_prediction(prediction_data: Dict) -> bool:
    """Guardar predicción de stroke en base de datos"""
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
            prediction_data['gender'],
            prediction_data['age'],
            prediction_data['hypertension'],
            prediction_data['heart_disease'],
            prediction_data['ever_married'],
            prediction_data['work_type'],
            prediction_data['residence_type'],
            prediction_data['avg_glucose_level'],
            prediction_data.get('bmi'),
            prediction_data['smoking_status'],
            prediction_data['prediction'],
            prediction_data['probability'],
            prediction_data['risk_level']
        ))
        
        conn.commit()
        cursor.close()
        conn.close()
        return True
        
    except Exception as e:
        print(f"Error en base de datos: {e}")
        return False

async def get_stroke_predictions(limit: int = 50) -> List[Dict]:
    """Obtener todas las predicciones de stroke de la base de datos"""
    try:
        conn = psycopg2.connect(os.getenv("DATABASE_URL"))
        cursor = conn.cursor()
        
        select_sql = """
            SELECT id, fecha_creacion, gender, age, hypertension, heart_disease, 
                   ever_married, work_type, residence_type, avg_glucose_level, 
                   bmi, smoking_status, prediction, probability, risk_level
            FROM stroke_predictions_formatted 
            ORDER BY id DESC 
            LIMIT %s
        """
        
        cursor.execute(select_sql, (limit,))
        rows = cursor.fetchall()
        
        columns = [
            'id', 'fecha_creacion', 'gender', 'age', 'hypertension', 'heart_disease',
            'ever_married', 'work_type', 'residence_type', 'avg_glucose_level',
            'bmi', 'smoking_status', 'prediction', 'probability', 'risk_level'
        ]
        
        predictions = []
        for row in rows:
            prediction = dict(zip(columns, row))
            predictions.append(prediction)
        
        cursor.close()
        conn.close()
        return predictions
        
    except Exception as e:
        print(f"Error en base de datos: {e}")
        return []

async def test_db_connection() -> bool:
    """Probar conexión a base de datos para health check"""
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