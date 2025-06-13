import os
from dotenv import load_dotenv

load_dotenv()

API_BASE_URL = os.getenv('API_BASE_URL', 'http://localhost:8000')
FRONTEND_PORT = int(os.getenv('FRONTEND_PORT', 8050))

# Valores exactos para dropdowns
DROPDOWN_VALUES = {
    'genero': [
        {'label': 'Masculino', 'value': 'Masculino'},
        {'label': 'Femenino', 'value': 'Femenino'},
        {'label': 'Otro', 'value': 'Otro'}
    ],
    'hipertension': [
        {'label': 'Sí', 'value': 'Sí'},
        {'label': 'No', 'value': 'No'}
    ],
    'enfermedad': [
        {'label': 'Sí', 'value': 'Sí'},
        {'label': 'No', 'value': 'No'}
    ],
    'casado': [
        {'label': 'Sí', 'value': 'Sí'},
        {'label': 'No', 'value': 'No'}
    ],
    'trabajo': [
        {'label': 'Empleado Público', 'value': 'Empleado Público'},
        {'label': 'Privado', 'value': 'Privado'},
        {'label': 'Autónomo', 'value': 'Autónomo'},
        {'label': 'Niño', 'value': 'Niño'},
        {'label': 'Nunca trabajó', 'value': 'Nunca trabajó'}
    ],
    'residencia': [
        {'label': 'Urbano', 'value': 'Urbano'},
        {'label': 'Rural', 'value': 'Rural'}
    ],
    'fumador': [
        {'label': 'Nunca fumó', 'value': 'Nunca fumó'},
        {'label': 'Fuma', 'value': 'Fuma'},
        {'label': 'Fumó antes', 'value': 'Fumó antes'},
        {'label': 'NS/NC', 'value': 'NS/NC'}
    ]
}

# Configuración de estilos
COLORS = {
    "Bajo": "#28a745",
    "Medio": "#ffc107", 
    "Alto": "#fd7e14",
    "Crítico": "#dc3545"
}

CSS_CLASSES = {
    "Bajo": "result-card-low",
    "Medio": "result-card-medium", 
    "Alto": "result-card-high",
    "Crítico": "result-card-critical"
}

# Recomendaciones médicas
RECOMMENDATIONS = {
    "Bajo": "Mantenga un estilo de vida saludable y chequeos médicos regulares.",
    "Medio": "Considere revisar sus factores de riesgo con un médico.",
    "Alto": "Consulte a un médico pronto para evaluación detallada.",
    "Crítico": "Busque atención médica inmediata. Consulte a un especialista."
}