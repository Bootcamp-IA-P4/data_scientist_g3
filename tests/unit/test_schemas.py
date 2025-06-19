import pytest
import sys
import os

# Añadir ruta para importar módulos
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'backend', 'app'))

@pytest.mark.unit
@pytest.mark.critical
class TestSchemas:
    
    @pytest.mark.critical
    def test_stroke_request_valid_ranges(self):
        """CRÍTICO: Prueba de rangos válidos para StrokeRequest"""
        try:
            from models.schemas import StrokeRequest
            from pydantic import ValidationError
            
            # Datos válidos según el schema
            valid_data = {
                "gender": "Masculino",
                "age": 65,
                "hypertension": "Sí",
                "heart_disease": "No",
                "ever_married": "Sí",
                "work_type": "Privado",
                "residence_type": "Urbano",
                "avg_glucose_level": 180,
                "bmi": 28.5,
                "smoking_status": "Nunca fumó"
            }
            
            request = StrokeRequest(**valid_data)
            assert request.age == 65
            assert request.avg_glucose_level == 180
            
        except ImportError:
            pytest.skip("Módulo models.schemas no disponible")
    
    @pytest.mark.critical
    def test_stroke_request_age_validation(self):
        """CRÍTICO: Prueba de validación de edad (0-120)"""
        try:
            from models.schemas import StrokeRequest
            from pydantic import ValidationError
            
            # Casos de prueba para edad
            age_test_cases = [
                (0, True),      # Límite inferior válido
                (120, True),    # Límite superior válido
                (-1, False),    # Edad negativa inválida
                (121, False),   # Edad mayor al límite inválida
                (65, True),     # Edad normal válida
            ]
            
            base_data = {
                "gender": "Masculino",
                "hypertension": "Sí",
                "heart_disease": "No",
                "ever_married": "Sí",
                "work_type": "Privado",
                "residence_type": "Urbano",
                "avg_glucose_level": 180,
                "bmi": 28.5,
                "smoking_status": "Nunca fumó"
            }
            
            for age, should_be_valid in age_test_cases:
                test_data = {**base_data, "age": age}
                
                if should_be_valid:
                    request = StrokeRequest(**test_data)
                    assert request.age == age
                else:
                    with pytest.raises(ValidationError):
                        StrokeRequest(**test_data)
                        
        except ImportError:
            pytest.skip("Módulo models.schemas no disponible")
    
    @pytest.mark.critical
    def test_stroke_request_glucose_validation(self):
        """CRÍTICO: Prueba de validación de glucosa (50-500)"""
        try:
            from models.schemas import StrokeRequest
            from pydantic import ValidationError
            
            # Casos de prueba para glucosa
            glucose_test_cases = [
                (50, True),     # Límite inferior válido
                (500, True),    # Límite superior válido
                (49, False),    # Glucosa menor al límite inválida
                (501, False),   # Glucosa mayor al límite inválida
                (180, True),    # Glucosa normal válida
            ]
            
            base_data = {
                "gender": "Masculino",
                "age": 65,
                "hypertension": "Sí",
                "heart_disease": "No",
                "ever_married": "Sí",
                "work_type": "Privado",
                "residence_type": "Urbano",
                "bmi": 28.5,
                "smoking_status": "Nunca fumó"
            }
            
            for glucose, should_be_valid in glucose_test_cases:
                test_data = {**base_data, "avg_glucose_level": glucose}
                
                if should_be_valid:
                    request = StrokeRequest(**test_data)
                    assert request.avg_glucose_level == glucose
                else:
                    with pytest.raises(ValidationError):
                        StrokeRequest(**test_data)
                        
        except ImportError:
            pytest.skip("Módulo models.schemas no disponible")
    
    def test_stroke_request_bmi_validation(self):
        """Prueba de validación de BMI"""
        try:
            from models.schemas import StrokeRequest
            from pydantic import ValidationError
            
            # Casos de prueba para BMI
            bmi_test_cases = [
                (10.0, True),   # BMI bajo pero válido
                (50.0, True),   # BMI alto pero válido
                (28.5, True),   # BMI normal
                (-1.0, False),  # BMI negativo inválido
                (0.0, False),   # BMI cero inválido
            ]
            
            base_data = {
                "gender": "Masculino",
                "age": 65,
                "hypertension": "Sí",
                "heart_disease": "No",
                "ever_married": "Sí",
                "work_type": "Privado",
                "residence_type": "Urbano",
                "avg_glucose_level": 180,
                "smoking_status": "Nunca fumó"
            }
            
            for bmi, should_be_valid in bmi_test_cases:
                test_data = {**base_data, "bmi": bmi}
                
                if should_be_valid:
                    request = StrokeRequest(**test_data)
                    assert request.bmi == bmi
                else:
                    with pytest.raises(ValidationError):
                        StrokeRequest(**test_data)
                        
        except ImportError:
            pytest.skip("Módulo models.schemas no disponible")
    
    @pytest.mark.critical
    def test_stroke_request_categorical_validation(self):
        """CRÍTICO: Prueba de validación de variables categóricas"""
        try:
            from models.schemas import StrokeRequest
            from pydantic import ValidationError
            
            # Datos base completos
            base_data = {
                "gender": "Masculino",
                "age": 65,
                "hypertension": "Sí",
                "heart_disease": "No",
                "ever_married": "Sí",
                "work_type": "Privado",
                "residence_type": "Urbano",
                "avg_glucose_level": 180,
                "bmi": 28.5,
                "smoking_status": "Nunca fumó"
            }
            
            # Probar valores válidos para cada campo categórico
            valid_values = {
                "gender": ["Masculino", "Femenino", "Otro"],
                "hypertension": ["Sí", "No"],
                "heart_disease": ["Sí", "No"],
                "ever_married": ["Sí", "No"],
                "work_type": ["Empleado Público", "Privado", "Autónomo", "Niño", "Nunca trabajó"],
                "residence_type": ["Urbano", "Rural"],
                "smoking_status": ["Nunca fumó", "Fuma", "Fumó antes", "NS/NC"]
            }
            
            # Probar cada campo categórico
            for field, values in valid_values.items():
                for value in values:
                    test_data = {**base_data, field: value}
                    request = StrokeRequest(**test_data)
                    assert getattr(request, field) == value
                
                # Probar valor inválido
                test_data = {**base_data, field: "VALOR_INVÁLIDO"}
                with pytest.raises(ValidationError):
                    StrokeRequest(**test_data)
                    
        except ImportError:
            pytest.skip("Módulo models.schemas no disponible")
    
    def test_stroke_request_required_fields(self):
        """Prueba de campos requeridos"""
        try:
            from models.schemas import StrokeRequest
            from pydantic import ValidationError
            
            # Datos completos válidos
            complete_data = {
                "gender": "Masculino",
                "age": 65,
                "hypertension": "Sí",
                "heart_disease": "No",
                "ever_married": "Sí",
                "work_type": "Privado",
                "residence_type": "Urbano",
                "avg_glucose_level": 180,
                "bmi": 28.5,
                "smoking_status": "Nunca fumó"
            }
            
            # Probar que funciona con todos los campos
            request = StrokeRequest(**complete_data)
            assert request is not None
            
            # Probar que falla sin campos requeridos
            required_fields = ["gender", "age", "hypertension", "heart_disease", "ever_married", 
                             "work_type", "residence_type", "avg_glucose_level", "smoking_status"]
            
            for field in required_fields:
                incomplete_data = {k: v for k, v in complete_data.items() if k != field}
                with pytest.raises(ValidationError):
                    StrokeRequest(**incomplete_data)
                    
        except ImportError:
            pytest.skip("Módulo models.schemas no disponible") 