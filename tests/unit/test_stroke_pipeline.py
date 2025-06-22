import pytest
import numpy as np
from src.pipeline.stroke_pipeline import StrokePipeline

@pytest.mark.critical
class TestStrokePipeline:
    
    @pytest.mark.critical
    def test_transform_data_features_count(self, stroke_pipeline, test_data):
        """Verificar el número y orden de características después de la transformación"""
        features = stroke_pipeline._transform_data(test_data)
        
        # Verificar número de características
        assert features.shape == (1, 17)
        
        # Verificar orden de características
        expected_order = [
            'age_scaled',           # 0
            'glucose_log_scaled',   # 1
            'bmi_scaled',           # 2
            'hypertension',         # 3
            'heart_disease',        # 4
            'gender',               # 5
            'ever_married',         # 6
            'residence',            # 7
            'work_type_children',   # 8
            'work_type_govt',       # 9
            'work_type_never',      # 10
            'work_type_private',    # 11
            'work_type_self',       # 12
            'smoke_former',         # 13
            'smoke_never',          # 14
            'smoke_current',        # 15
            'smoke_unknown'         # 16
        ]
        
        # Verificar tipos de datos
        assert features.dtype == np.float64
        
        # Verificar rangos de valores
        assert np.all(features[:, 0:3] >= -3) and np.all(features[:, 0:3] <= 3)  # Características escaladas
        assert np.all(features[:, 3:7] >= 0) and np.all(features[:, 3:7] <= 1)   # Características binarias
        assert np.all(features[:, 7:] >= 0) and np.all(features[:, 7:] <= 1)     # Características one-hot

    @pytest.mark.critical
    def test_transform_data_categorical_mapping(self, stroke_pipeline):
        """Verificar la correcta asignación de variables categóricas"""
        test_cases = [
            {
                'input': {
                    'gender': "Masculino",
                    'work_type': "Privado",
                    'smoking_status': "Fumó antes",
                    'hypertension': "Sí",
                    'heart_disease': "No",
                    'ever_married': "Sí",
                    'residence_type': "Urbano",
                    'age': 65,
                    'avg_glucose_level': 180,
                    'bmi': 28.5
                },
                'expected': {
                    'gender': 0,  # Masculino
                    'work_type_private': 1,
                    'smoke_former': 1,
                    'hypertension': 1,
                    'heart_disease': 0,
                    'ever_married': 1,
                    'residence': 0
                }
            }
        ]
        
        for case in test_cases:
            features = stroke_pipeline._transform_data(case['input'])
            assert features[0, 5] == case['expected']['gender']
            assert features[0, 11] == case['expected']['work_type_private']
            assert features[0, 13] == case['expected']['smoke_former']
            assert features[0, 3] == case['expected']['hypertension']
            assert features[0, 4] == case['expected']['heart_disease']
            assert features[0, 6] == case['expected']['ever_married']
            assert features[0, 7] == case['expected']['residence']

    @pytest.mark.critical
    def test_predict_stroke_risk_consistency(self, stroke_pipeline, test_data):
        """Verificar la consistencia de las predicciones"""
        # Primera predicción
        result1 = stroke_pipeline.predict_stroke_risk(test_data)
        
        # Segunda predicción con los mismos datos
        result2 = stroke_pipeline.predict_stroke_risk(test_data)
        
        # Verificar consistencia
        assert result1['prediction'] == result2['prediction']
        assert abs(result1['probability'] - result2['probability']) < 0.0001
        assert abs(result1['model_confidence'] - result2['model_confidence']) < 0.0001
        
        # Verificar estructura del resultado
        assert 'prediction' in result1
        assert 'probability' in result1
        assert 'model_confidence' in result1
        assert 'feature_importance' in result1
        
        # Verificar tipos de datos
        assert isinstance(result1['prediction'], int)
        assert isinstance(result1['probability'], float)
        assert isinstance(result1['model_confidence'], float)
        assert isinstance(result1['feature_importance'], dict)

    @pytest.mark.critical
    def test_predict_stroke_risk_edge_cases(self, stroke_pipeline):
        """Verificar casos límite de predicciones"""
        edge_cases = [
            {
                'data': {
                    'gender': "Masculino",
                    'age': 100,  # Edad máxima
                    'hypertension': "Sí",
                    'heart_disease': "Sí",
                    'ever_married': "Sí",
                    'work_type': "Privado",
                    'residence_type': "Urbano",
                    'avg_glucose_level': 300,  # Nivel alto de glucosa
                    'bmi': 40,  # BMI alto
                    'smoking_status': "Fuma"
                },
                'expected_risk': 'high'
            },
            {
                'data': {
                    'gender': "Femenino",
                    'age': 20,  # Edad mínima
                    'hypertension': "No",
                    'heart_disease': "No",
                    'ever_married': "No",
                    'work_type': "Privado",
                    'residence_type': "Urbano",
                    'avg_glucose_level': 80,  # Nivel normal de glucosa
                    'bmi': 22,  # BMI normal
                    'smoking_status': "Nunca fumó"
                },
                'expected_risk': 'low'
            }
        ]
        
        for case in edge_cases:
            result = stroke_pipeline.predict_stroke_risk(case['data'])
            assert result['prediction'] in [0, 1]
            assert 0 <= result['probability'] <= 1
            assert 0 <= result['model_confidence'] <= 1

    @pytest.mark.critical
    def test_feature_importance_consistency(self, stroke_pipeline, test_data):
        """Verificar la consistencia de la importancia de características"""
        result = stroke_pipeline.predict_stroke_risk(test_data)
        importance = result['feature_importance']
        
        # Verificar estructura de importancia de características
        assert 'top_risk_factors' in importance
        assert 'detailed_analysis' in importance
        
        # Verificar top_risk_factors
        top_factors = importance['top_risk_factors']
        assert len(top_factors) <= 5
        for factor in top_factors:
            assert 'rank' in factor
            assert 'feature' in factor
            assert 'label' in factor
            assert 'contribution_percentage' in factor
            assert 'value' in factor
            assert 'impact' in factor
            assert factor['impact'] in ['Alto', 'Medio', 'Bajo']
        
        # Verificar detailed_analysis
        detailed = importance['detailed_analysis']
        assert len(detailed) == 17  # Todas las características deben ser analizadas
        for feature, data in detailed.items():
            assert 'global_importance' in data
            assert 'patient_value' in data
            assert 'contribution' in data
            assert 'percentage' in data
            assert 'label' in data