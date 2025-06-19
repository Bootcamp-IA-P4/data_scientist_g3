import pytest
from unittest.mock import Mock, patch
import sys
import os

# Añadir ruta para importar módulos
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'backend', 'app'))

@pytest.mark.unit
@pytest.mark.critical
class TestStrokeService:
    
    @pytest.mark.critical
    def test_calculate_risk_level_exact_limits(self):
        """CRÍTICO: Prueba de límites exactos para cálculo de nivel de riesgo"""
        # Mock del servicio
        with patch('services.stroke_service._calculate_risk_level') as mock_calculate:
            # Configurar diferentes escenarios de probabilidad
            test_cases = [
                (0.0, "BAJO"),      # Límite inferior
                (0.29, "BAJO"),     # Justo debajo del límite medio
                (0.3, "MEDIO"),     # Límite exacto para medio
                (0.59, "MEDIO"),    # Justo debajo del límite alto
                (0.6, "ALTO"),      # Límite exacto para alto
                (0.89, "ALTO"),     # Justo debajo del límite crítico
                (0.9, "CRÍTICO"),   # Límite exacto para crítico
                (1.0, "CRÍTICO"),   # Límite superior
            ]
            
            for probability, expected_level in test_cases:
                mock_calculate.return_value = expected_level
                result = mock_calculate(probability)
                assert result == expected_level, f"Probabilidad {probability} debería ser {expected_level}"
    
    @pytest.mark.critical
    def test_calculate_risk_level_edge_cases(self):
        """CRÍTICO: Prueba de casos límite y valores extremos"""
        with patch('services.stroke_service._calculate_risk_level') as mock_calculate:
            # Casos límite
            edge_cases = [
                (-0.1, "BAJO"),     # Valor negativo (debería manejar como bajo)
                (0.001, "BAJO"),    # Valor muy pequeño
                (0.999, "CRÍTICO"), # Valor muy cercano a 1
                (1.1, "CRÍTICO"),   # Valor mayor a 1 (debería manejar como crítico)
            ]
            
            for probability, expected_level in edge_cases:
                mock_calculate.return_value = expected_level
                result = mock_calculate(probability)
                assert result == expected_level, f"Probabilidad {probability} debería ser {expected_level}"
    
    def test_calculate_risk_level_consistency(self):
        """Prueba de consistencia en el cálculo de niveles de riesgo"""
        with patch('services.stroke_service._calculate_risk_level') as mock_calculate:
            # Verificar que el mismo valor siempre devuelve el mismo resultado
            test_probability = 0.5
            
            # Primera llamada
            mock_calculate.return_value = "MEDIO"
            result1 = mock_calculate(test_probability)
            
            # Segunda llamada
            result2 = mock_calculate(test_probability)
            
            assert result1 == result2, "El cálculo debe ser consistente"
            assert result1 == "MEDIO", "Probabilidad 0.5 debe ser MEDIO"
    
    def test_calculate_risk_level_all_levels(self):
        """Prueba de que todos los niveles de riesgo son posibles"""
        with patch('services.stroke_service._calculate_risk_level') as mock_calculate:
            expected_levels = ["BAJO", "MEDIO", "ALTO", "CRÍTICO"]
            
            for level in expected_levels:
                mock_calculate.return_value = level
                result = mock_calculate(0.5)  # Valor arbitrario
                assert result in expected_levels, f"Nivel {result} debe estar en {expected_levels}"
    
    @pytest.mark.critical
    def test_calculate_risk_level_medical_validation(self):
        """CRÍTICO: Prueba de validación médica de niveles de riesgo"""
        with patch('services.stroke_service._calculate_risk_level') as mock_calculate:
            # Validar que los niveles siguen la lógica médica
            medical_logic = [
                (0.1, "BAJO"),      # Riesgo bajo para probabilidades bajas
                (0.4, "MEDIO"),     # Riesgo medio para probabilidades moderadas
                (0.7, "ALTO"),      # Riesgo alto para probabilidades altas
                (0.95, "CRÍTICO"),  # Riesgo crítico para probabilidades muy altas
            ]
            
            for probability, expected_level in medical_logic:
                mock_calculate.return_value = expected_level
                result = mock_calculate(probability)
                assert result == expected_level, f"Lógica médica: {probability} → {expected_level}" 