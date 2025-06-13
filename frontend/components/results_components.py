# components/results_components.py
from dash import html
from config.settings import CSS_CLASSES, RECOMMENDATIONS

def create_result_card(prediction, probability, risk_level):
    """
    Crea la tarjeta de resultados con los 3 elementos principales:
    1. Diagnóstico principal (stroke/no stroke)
    2. Porcentaje del modelo (probability * 100)
    3. Nivel de riesgo del backend
    """
    
    # 1. Diagnóstico principal según prediction
    if prediction == 0:
        diagnosis_message = "✅ SIN RIESGO DE STROKE"
        diagnosis_color = "#28a745"  # Verde
    else:
        diagnosis_message = "⚠️ RIESGO DE STROKE DETECTADO"
        diagnosis_color = "#dc3545"  # Rojo
    
    # 2. Porcentaje del modelo
    percentage = probability * 100
    
    # 3. Nivel de riesgo del backend
    card_class = get_risk_color_class(risk_level)
    recommendation = get_recommendation(risk_level)
    
    # Crear tarjeta completa
    return html.Div([
        # Mensaje principal
        html.Div(
            diagnosis_message, 
            className="diagnosis-message", 
            style={'color': diagnosis_color}
        ),
        
        # Porcentaje grande
        html.Div(
            f"{percentage:.1f}%", 
            className="percentage-display"
        ),
        
        # Nivel de riesgo
        html.Div(
            f"Nivel de riesgo: {risk_level}", 
            className="risk-level"
        ),
        
        # Recomendación médica
        html.Div(
            recommendation, 
            className="recommendation"
        )
        
    ], className=f"result-card {card_class}")

def create_error_message(error_text):
    """Crea un mensaje de error estilizado"""
    
    return html.Div([
        html.Div(
            error_text, 
            style={
                'color': 'red', 
                'text-align': 'center', 
                'padding': '20px',
                'background-color': '#f8d7da', 
                'border-radius': '5px',
                'border': '1px solid #f5c6cb'
            }
        )
    ])

def get_risk_color_class(risk_level):
    """
    Retorna la clase CSS según el nivel de riesgo
    Colores según la guía:
    - "Bajo" → Verde
    - "Medio" → Amarillo
    - "Alto" → Naranja  
    - "Crítico" → Rojo
    """
    return CSS_CLASSES.get(risk_level, "result-card-low")

def get_recommendation(risk_level):
    """
    Retorna recomendación médica según el nivel de riesgo
    Recomendaciones específicas por nivel
    """
    return RECOMMENDATIONS.get(
        risk_level, 
        "Consulte a un profesional médico."
    )

def create_disclaimer():
    """Crea el disclaimer médico"""
    
    return html.Div([
        html.P(
            "⚠️ Disclaimer: Esta herramienta es solo para fines educativos y "
            "no sustituye el consejo médico profesional. Siempre consulte a "
            "un médico para evaluaciones médicas reales."
        )
    ], className="disclaimer")