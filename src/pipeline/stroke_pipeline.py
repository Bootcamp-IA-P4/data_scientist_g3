"""
Pipeline corregido de predicción de stroke usando XGBoost con StandardScaler REAL
"""

import pickle
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional

class StrokePipeline:
    """Pipeline completo para predicción de riesgo de stroke"""
    
    def __init__(self, 
                 model_filename: str = "xgboost_stroke_optimized_20250609_124848.pkl",
                 scaler_filename: str = "scaler_recreated.pkl"):
        self.model = None
        self.scaler = None
        self.model_loaded = False
        self.scaler_loaded = False
        self.model_path = None
        self.scaler_path = None
        self.model_filename = model_filename
        self.scaler_filename = scaler_filename
        
        # Mapeos exactos del preprocesamiento original
        self.gender_mapping_es_to_en = {
            "Masculino": "Male",
            "Femenino": "Female", 
            "Otro": "Other"
        }
        
        self.work_type_mapping_es_to_en = {
            "Niño": "children",
            "Empleado Público": "Govt_job",
            "Nunca trabajó": "Never_worked",
            "Privado": "Private",
            "Autónomo": "Self-employed"
        }
        
        self.smoking_mapping_es_to_en = {
            "Fumó antes": "formerly smoked",
            "Nunca fumó": "never smoked",
            "Fuma": "smokes",
            "NS/NC": "Unknown"
        }
        
        self._load_model()
        self._load_scaler()
    
    def _load_model(self):
        """Carga el modelo XGBoost desde models/xgboost/"""
        try:
            # Buscar modelo desde la raíz del proyecto
            current_dir = Path(__file__).resolve().parent
            project_root = current_dir.parent.parent  
            model_path = project_root / "models" / "xgboost" / self.model_filename
            
            if not model_path.exists():
                print(f"❌ Modelo no encontrado en: {model_path}")
                return
            
            with open(model_path, 'rb') as f:
                self.model = pickle.load(f)
            
            self.model_loaded = True
            self.model_path = str(model_path)
            print(f"✅ Modelo XGBoost cargado desde: {model_path}")
            
        except Exception as e:
            print(f"❌ Error cargando modelo: {str(e)}")
            self.model_loaded = False
    
    def _load_scaler(self):
        """Carga el StandardScaler REAL recreado"""
        try:
            # Buscar scaler desde la raíz del proyecto
            current_dir = Path(__file__).resolve().parent
            project_root = current_dir.parent.parent  
            scaler_path = project_root / "models" / self.scaler_filename
            
            if not scaler_path.exists():
                print(f"❌ Scaler no encontrado en: {scaler_path}")
                return
            
            with open(scaler_path, 'rb') as f:
                self.scaler = pickle.load(f)
            
            self.scaler_loaded = True
            self.scaler_path = str(scaler_path)
            print(f"✅ StandardScaler REAL cargado desde: {scaler_path}")
            
        except Exception as e:
            print(f"❌ Error cargando scaler: {str(e)}")
            self.scaler_loaded = False
    
    def predict_stroke_risk(self, patient_data: Dict) -> Dict:
        """Predice el riesgo de stroke con análisis de importancia"""
        if not self.model_loaded:
            raise Exception("Modelo no cargado. Verificar ruta del archivo.")
        
        if not self.scaler_loaded:
            raise Exception("Scaler no cargado. Verificar ruta del archivo.")
        
        try:
            # Transformar datos según el preprocesamiento original
            features = self._transform_data(patient_data)
            prediction = self.model.predict(features)[0]
            probabilities = self.model.predict_proba(features)[0]
            probability = probabilities[1]
            confidence = max(probabilities)
            
            # Análisis de importancia
            feature_importance = self._get_feature_importance(features[0])
            
            return {
                'prediction': int(prediction),
                'probability': float(probability),
                'model_confidence': float(confidence),
                'feature_importance': feature_importance
            }
            
        except Exception as e:
            raise Exception(f"Error en predicción: {str(e)}")
    
    def _transform_data(self, patient_data: Dict) -> np.ndarray:
        """
        Transforma datos EXACTAMENTE como en el preprocesamiento original
        """
        
        # PASO 1: Convertir español → inglés
        gender_en = self.gender_mapping_es_to_en.get(patient_data['gender'], 'Other')
        work_type_en = self.work_type_mapping_es_to_en.get(patient_data['work_type'], 'Private')
        smoking_en = self.smoking_mapping_es_to_en.get(patient_data['smoking_status'], 'Unknown')
        
        # PASO 2: Variables numéricas (aplicar transformaciones)
        age = float(patient_data['age'])
        avg_glucose_level = float(patient_data['avg_glucose_level'])
        glucose_log = np.log1p(avg_glucose_level)  # USAR log1p como en preprocesamiento
        bmi = float(patient_data['bmi']) if patient_data['bmi'] is not None else 28.498  # Media del dataset
        
        # PASO 3: Variables categóricas exactas del preprocesamiento
        # Gender mapping: Male=0, Female=1, Other=2
        if gender_en == 'Male':
            gender = 0
        elif gender_en == 'Female':
            gender = 1
        else:  # Other
            gender = 2
        
        # Binary variables: No=0, Yes=1
        hypertension = 1 if patient_data['hypertension'] == "Sí" else 0
        heart_disease = 1 if patient_data['heart_disease'] == "Sí" else 0
        ever_married = 1 if patient_data['ever_married'] == "Sí" else 0
        
        # Residence mapping: Urban=0, Rural=1 (como en preprocesamiento)
        residence = 0 if patient_data['residence_type'] == "Urbano" else 1
        
        # PASO 4: One-hot encoding work_type (orden exacto del preprocesamiento)
        work_type_children = 1 if work_type_en == "children" else 0
        work_type_govt = 1 if work_type_en == "Govt_job" else 0
        work_type_never = 1 if work_type_en == "Never_worked" else 0
        work_type_private = 1 if work_type_en == "Private" else 0
        work_type_self = 1 if work_type_en == "Self-employed" else 0
        
        # PASO 5: One-hot encoding smoking_status (orden exacto del preprocesamiento)
        smoke_former = 1 if smoking_en == "formerly smoked" else 0
        smoke_never = 1 if smoking_en == "never smoked" else 0
        smoke_current = 1 if smoking_en == "smokes" else 0
        smoke_unknown = 1 if smoking_en == "Unknown" else 0
        
        # PASO 6: Estandarizar variables numéricas con el SCALER REAL
        numeric_features = np.array([[age, glucose_log, bmi]])
        numeric_features_scaled = self.scaler.transform(numeric_features)[0]
        age_scaled, glucose_log_scaled, bmi_scaled = numeric_features_scaled
        
        # PASO 7: Array final en ORDEN EXACTO del preprocesamiento
        # Orden: num_cols + bin_cat_cols + cat_cols
        features = np.array([
            age_scaled,           # 0 - age (estandarizada)
            glucose_log_scaled,   # 1 - avg_glucose_level_log (estandarizada)
            bmi_scaled,           # 2 - bmi (estandarizada)
            hypertension,         # 3 - hypertension
            heart_disease,        # 4 - heart_disease
            gender,               # 5 - gender
            ever_married,         # 6 - ever_married
            residence,            # 7 - Residence_type
            work_type_children,   # 8 - work_type_children
            work_type_govt,       # 9 - work_type_Govt_job
            work_type_never,      # 10 - work_type_Never_worked
            work_type_private,    # 11 - work_type_Private
            work_type_self,       # 12 - work_type_Self-employed
            smoke_former,         # 13 - smoking_status_formerly smoked
            smoke_never,          # 14 - smoking_status_never smoked
            smoke_current,        # 15 - smoking_status_smokes
            smoke_unknown         # 16 - smoking_status_Unknown
        ]).reshape(1, -1)
        
        return features
    
    def _get_feature_importance(self, patient_features: np.ndarray) -> Dict:
        """Calcula importancia de características usando los valores OFICIALES del modelo"""
        try:
            # Importancia OFICIAL del modelo optimizado (del JSON)
            official_importance = {
                "age": 0.2773195207118988,
                "avg_glucose_level_log": 0.0915384516119957,
                "bmi": 0.05708884447813034,
                "hypertension": 0.11351834237575531,
                "heart_disease": 0.056257277727127075,
                "gender": 0.0192119088023901,
                "ever_married": 0.09477852284908295,
                "Residence_type": 0.015864524990320206,
                "work_type_children": 0.08318447321653366,
                "work_type_Govt_job": 0.024294931441545486,
                "work_type_Never_worked": 0.0,
                "work_type_Private": 0.016853569075465202,
                "work_type_Self-employed": 0.031471095979213715,
                "smoking_status_formerly smoked": 0.034214090555906296,
                "smoking_status_never smoked": 0.03221003711223602,
                "smoking_status_smokes": 0.021826256066560745,
                "smoking_status_Unknown": 0.03036813624203205
            }
            
            # Nombres exactos del preprocesamiento
            feature_names = [
                'age', 'avg_glucose_level_log', 'bmi', 'hypertension', 'heart_disease',
                'gender', 'ever_married', 'Residence_type', 'work_type_children', 
                'work_type_Govt_job', 'work_type_Never_worked', 'work_type_Private', 
                'work_type_Self-employed', 'smoking_status_formerly smoked', 
                'smoking_status_never smoked', 'smoking_status_smokes', 'smoking_status_Unknown'
            ]
            
            # Etiquetas en español para mostrar al usuario
            feature_labels = {
                'age': 'Edad', 'avg_glucose_level_log': 'Nivel de Glucosa', 'bmi': 'BMI',
                'hypertension': 'Hipertensión', 'heart_disease': 'Enfermedad Cardíaca',
                'gender': 'Género', 'ever_married': 'Estado Civil',
                'Residence_type': 'Tipo de Residencia', 'work_type_children': 'Trabajo: Niño',
                'work_type_Govt_job': 'Trabajo: Empleado Público', 'work_type_Never_worked': 'Trabajo: Nunca trabajó',
                'work_type_Private': 'Trabajo: Privado', 'work_type_Self-employed': 'Trabajo: Autónomo',
                'smoking_status_formerly smoked': 'Ex-fumador', 'smoking_status_never smoked': 'Nunca fumó',
                'smoking_status_smokes': 'Fumador actual', 'smoking_status_Unknown': 'Estado fumador desconocido'
            }
            
            # Usar importancia oficial del modelo
            global_importance = np.array([official_importance[name] for name in feature_names])
            
            # Normalizar
            if global_importance.sum() > 0:
                global_importance = global_importance / global_importance.sum()
            
            # Calcular contribución por paciente
            patient_importance = {}
            for i, (name, importance, value) in enumerate(zip(feature_names, global_importance, patient_features)):
                # Para variables estandarizadas, la contribución es directa
                contribution = importance * abs(value) if i < 3 else importance * value
                
                patient_importance[name] = {
                    'global_importance': float(importance),
                    'patient_value': float(value),
                    'contribution': float(contribution),
                    'percentage': float((contribution / max(global_importance.sum(), 0.001)) * 100),
                    'label': feature_labels.get(name, name)
                }
            
            # Ordenar por contribución
            sorted_importance = dict(sorted(
                patient_importance.items(), 
                key=lambda x: x[1]['contribution'], 
                reverse=True
            ))
            
            # Top factores
            top_factors = []
            for i, (feature, data) in enumerate(list(sorted_importance.items())[:5]):
                if data['contribution'] > 0:
                    factor = {
                        'rank': i + 1,
                        'feature': feature,
                        'label': data['label'],
                        'contribution_percentage': round(data['percentage'], 1),
                        'value': data['patient_value'],
                        'impact': 'Alto' if data['percentage'] > 20 else 'Medio' if data['percentage'] > 10 else 'Bajo'
                    }
                    top_factors.append(factor)
            
            return {
                'top_risk_factors': top_factors,
                'detailed_analysis': sorted_importance
            }
            
        except Exception as e:
            print(f"⚠️ Error calculando importancia: {e}")
            return {'top_risk_factors': [], 'detailed_analysis': {}, 'error': str(e)}
    
    def get_pipeline_status(self) -> Dict:
        """Estado del pipeline"""
        status = {
            'is_loaded': self.model_loaded and self.scaler_loaded,
            'model_type': 'XGBoost' if self.model_loaded else None,
            'model_path': self.model_path,
            'scaler_path': self.scaler_path,
            'model_filename': self.model_filename,
            'scaler_filename': self.scaler_filename,
            'model_loaded': self.model_loaded,
            'scaler_loaded': self.scaler_loaded,
            'status': 'Pipeline cargado correctamente' if (self.model_loaded and self.scaler_loaded) else 'Error cargando componentes del pipeline'
        }
        
        if self.model_loaded and self.scaler_loaded:
            status.update({
                'model_features': 17,
                'model_classes': 2,
                'algorithm': 'XGBoost Classifier',
                'preprocessing': 'StandardScaler REAL + One-hot encoding',
                'scaler_params': {
                    'mean_': self.scaler.mean_.tolist() if self.scaler else None,
                    'scale_': self.scaler.scale_.tolist() if self.scaler else None
                }
            })
        
        return status

# Instancia global (Singleton)
_pipeline_instance: Optional[StrokePipeline] = None

def get_pipeline() -> StrokePipeline:
    """Obtiene la instancia global del pipeline"""
    global _pipeline_instance
    if _pipeline_instance is None:
        _pipeline_instance = StrokePipeline()
    return _pipeline_instance

def predict_stroke_risk(patient_data: Dict) -> Dict:
    """Función principal para el backend"""
    pipeline = get_pipeline()
    return pipeline.predict_stroke_risk(patient_data)

def get_pipeline_status() -> Dict:
    """Estado del pipeline"""
    pipeline = get_pipeline()
    return pipeline.get_pipeline_status()

# Test
if __name__ == "__main__":
    test_data = {
        'gender': "Masculino", 'age': 65, 'hypertension': "Sí",
        'heart_disease': "No", 'ever_married': "Sí", 'work_type': "Privado",
        'residence_type': "Urbano", 'avg_glucose_level': 180, 'bmi': 28.5,
        'smoking_status': "Fumó antes"
    }
    
    try:
        result = predict_stroke_risk(test_data)
        print(f"✅ Predicción: {result['prediction']}")
        print(f"📊 Probabilidad: {result['probability']:.4f}")
        print(f"🔥 Top factores:")
        for factor in result['feature_importance']['top_risk_factors'][:3]:
            print(f"   • {factor['label']}: {factor['contribution_percentage']}%")
            
        status = get_pipeline_status()
        print(f"📋 Estado: {status['status']}")
        print(f"🔧 Scaler cargado: {status['scaler_loaded']}")
        if status['scaler_loaded']:
            print(f"📈 Parámetros scaler - Media: {status['scaler_params']['mean_'][:3]}")
        
    except Exception as e:
        print(f"❌ Error: {e}")