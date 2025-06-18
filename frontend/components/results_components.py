from dash import html
from config.settings import CSS_CLASSES, RECOMMENDATIONS

def create_result_card(prediction, probability, risk_level, show_image_button= True):
    """
    Crea la tarjeta de resultados con los 3 elementos principales:
    1. Diagnóstico principal (stroke/no stroke)
    2. Porcentaje del modelo (probability * 100)
    3. Nivel de riesgo del backend

    Args:
        show_image_button: si mostrar botón para análisis de imagen
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

    action_buttons = []
    action_buttons.append(
        html.Button(
            "📊 Ver Historial",
            id='view-history-button',
            className="btn-secondary",
            n_clicks=0
        )
    )
    
    # Botón de análisis de imagen (condicional)
    if show_image_button:
        action_buttons.append(
            html.A(
                "🧠 Predecir Tomografía",
                href="/image-prediction?stroke_id=LATEST",
                className="btn-primary btn-image-analysis",
                id="predict-image-link"
            )
        )

    # Creando tarjeta completa
    card_content = [
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
        
    ]

    if action_buttons:
        card_content.append(
            html.Div(action_buttons, className="result-actions")
        )
    
    
    return html.Div(card_content, className=f"result-card {card_class}")

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

def create_success_message(message: str, subtitle: str = ""):
    """Crear mensaje de éxito"""
    return html.Div([
        html.I(className="fas fa-check-circle success-icon"),
        html.H4(message),
        html.P(subtitle) if subtitle else "",
    ], className="success-message")

def create_loading_message(message: str = "Procesando..."):
    """Crear mensaje de carga"""
    return html.Div([
        html.I(className="fas fa-spinner fa-spin loading-icon"),
        html.H4(message),
        html.P("Por favor espere mientras procesamos su solicitud.")
    ], className="loading-message")

def create_risk_explanation_card(risk_level: str):
    """Crear tarjeta explicativa del nivel de riesgo"""
    
    explanations = {
        "Bajo": {
            "emoji": "✅",
            "title": "Riesgo Bajo",
            "description": "Sus factores de riesgo actuales sugieren una probabilidad baja de stroke.",
            "actions": [
                "Mantener estilo de vida saludable",
                "Ejercicio regular",
                "Dieta balanceada",
                "Chequeos médicos anuales"
            ],
            "color": "#28a745"
        },
        "Medio": {
            "emoji": "⚠️",
            "title": "Riesgo Moderado", 
            "description": "Algunos factores de riesgo están presentes. Se recomienda atención preventiva.",
            "actions": [
                "Consultar con médico de cabecera",
                "Monitorear presión arterial",
                "Controlar niveles de glucosa",
                "Adoptar hábitos más saludables"
            ],
            "color": "#ffc107"
        },
        "Alto": {
            "emoji": "🚨",
            "title": "Riesgo Elevado",
            "description": "Múltiples factores de riesgo detectados. Se requiere evaluación médica.",
            "actions": [
                "Consultar médico urgentemente",
                "Evaluación cardiológica",
                "Control estricto de factores de riesgo",
                "Posible medicación preventiva"
            ],
            "color": "#fd7e14"
        },
        "Crítico": {
            "emoji": "🆘",
            "title": "Riesgo Muy Alto",
            "description": "Factores de riesgo críticos presentes. Atención médica inmediata requerida.",
            "actions": [
                "Buscar atención médica INMEDIATA",
                "Evaluación neurológica urgente", 
                "Tratamiento preventivo agresivo",
                "Monitoreo médico continuo"
            ],
            "color": "#dc3545"
        }
    }
    
    info = explanations.get(risk_level, explanations["Bajo"])
    
    return html.Div([
        html.Div([
            html.Span(info["emoji"], className="risk-emoji"),
            html.H3(info["title"])
        ], className="risk-header"),
        
        html.P(info["description"], className="risk-description"),
        
        html.Div([
            html.H4("📋 Acciones Recomendadas:"),
            html.Ul([
                html.Li(action) for action in info["actions"]
            ])
        ], className="risk-actions"),
        
    ], className="risk-explanation-card", style={"border-left": f"4px solid {info['color']}"})

def create_prediction_summary(stroke_data: dict, image_data: dict = None):
    """Crear resumen combinado de predicción stroke + imagen"""
    
    stroke_risk = stroke_data.get('risk_level', 'N/A')
    stroke_prob = stroke_data.get('probability', 0) * 100
    
    summary_content = [
        html.H3("📊 Resumen de Predicción Completa"),
        
        # Predicción de Stroke
        html.Div([
            html.Div([
                html.H4("🧠 Análisis Clínico"),
                html.Div([
                    html.Span(f"{stroke_prob:.1f}%", className="summary-percentage"),
                    html.Span(f"Riesgo: {stroke_risk}", className="summary-risk")
                ])
            ], className="summary-section stroke-summary")
        ])
    ]
    
    # Predicción de Imagen (si disponible)
    if image_data:
        image_risk = image_data.get('risk_level', 'N/A')
        image_prob = image_data.get('probability', 0)
        
        summary_content.append(
            html.Div([
                html.H4("📷 Análisis por Imagen"),
                html.Div([
                    html.Span(f"{image_prob:.1f}%", className="summary-percentage"),
                    html.Span(f"Riesgo: {image_risk}", className="summary-risk")
                ])
            ], className="summary-section image-summary")
        )
        
        # Correlación entre ambos análisis
        correlation = calculate_prediction_correlation(stroke_data, image_data)
        summary_content.append(create_correlation_analysis(correlation))
    
    else:
        # Sugerir análisis de imagen
        summary_content.append(
            html.Div([
                html.H4("📷 Análisis por Imagen"),
                html.P("Añada una tomografía computarizada para análisis completo"),
                html.A(
                    "📸 Subir Imagen",
                    href=f"/image-prediction?stroke_id={stroke_data.get('id', 'LATEST')}",
                    className="btn-primary btn-add-image"
                )
            ], className="summary-section image-pending")
        )
    
    return html.Div(summary_content, className="prediction-summary-card")

def calculate_prediction_correlation(stroke_data: dict, image_data: dict) -> dict:
    """Calcular correlación entre predicción clínica e imagen"""
    
    stroke_prob = stroke_data.get('probability', 0) * 100
    image_prob = image_data.get('probability', 0)
    
    # Diferencia entre predicciones
    diff = abs(stroke_prob - image_prob)
    
    # Nivel de concordancia
    if diff <= 10:
        concordance = "Alta"
        concordance_class = "concordance-high"
        interpretation = "Ambos análisis muestran resultados muy similares."
    elif diff <= 25:
        concordance = "Moderada"
        concordance_class = "concordance-medium"
        interpretation = "Los análisis muestran cierta variación, pero son generalmente consistentes."
    else:
        concordance = "Baja"
        concordance_class = "concordance-low"
        interpretation = "Existe discrepancia significativa entre los análisis. Consulte con un especialista."
    
    return {
        'difference': diff,
        'concordance': concordance,
        'concordance_class': concordance_class,
        'interpretation': interpretation,
        'stroke_higher': stroke_prob > image_prob
    }

def create_correlation_analysis(correlation: dict):
    """Crear análisis de correlación entre predicciones"""
    
    return html.Div([
        html.H4("🔄 Correlación de Análisis"),
        html.Div([
            html.Div([
                html.Span("Concordancia:", className="correlation-label"),
                html.Span(correlation['concordance'], className=f"correlation-value {correlation['concordance_class']}")
            ], className="correlation-metric"),
            
            html.Div([
                html.Span("Diferencia:", className="correlation-label"),
                html.Span(f"{correlation['difference']:.1f}%", className="correlation-value")
            ], className="correlation-metric")
        ], className="correlation-metrics"),
        
        html.P(correlation['interpretation'], className="correlation-interpretation"),
        
        # Recomendación específica
        html.Div([
            html.I(className="fas fa-lightbulb"),
            html.Span(get_correlation_recommendation(correlation))
        ], className="correlation-recommendation")
        
    ], className="correlation-analysis")

def get_correlation_recommendation(correlation: dict) -> str:
    """Obtener recomendación basada en correlación"""
    
    if correlation['concordance'] == "Alta":
        return "Los resultados son consistentes. Siga las recomendaciones médicas estándar."
    elif correlation['concordance'] == "Moderada":
        return "Considere evaluación médica adicional para confirmar el diagnóstico."
    else:
        return "La discrepancia requiere evaluación especializada inmediata para interpretación correcta."

def create_quick_actions_card():
    """Crear tarjeta de acciones rápidas"""
    
    return html.Div([
        html.H4("⚡ Acciones Rápidas"),
        html.Div([
            html.Button([
                html.I(className="fas fa-plus"),
                html.Span("Nueva Predicción")
            ], className="btn-primary quick-action-btn", id="new-prediction-quick"),
            
            html.Button([
                html.I(className="fas fa-camera"),
                html.Span("Subir Imagen")
            ], className="btn-secondary quick-action-btn", id="upload-image-quick"),
            
            html.Button([
                html.I(className="fas fa-chart-bar"),
                html.Span("Ver Estadísticas")
            ], className="btn-secondary quick-action-btn", id="view-stats-quick"),
            
            html.Button([
                html.I(className="fas fa-download"),
                html.Span("Exportar Datos")
            ], className="btn-secondary quick-action-btn", id="export-data-quick")
        ], className="quick-actions-grid")
    ], className="quick-actions-card")