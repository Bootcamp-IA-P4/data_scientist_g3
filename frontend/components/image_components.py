from dash import html, dcc, dash_table
import base64
from typing import List, Dict, Optional

def create_upload_restrictions_info():
    """Información sobre restricciones de upload - versión simplificada"""
    return html.Div([
        html.H3("Especificaciones Técnicas"),
        html.Div([
            html.P([
                "", html.Strong("Formatos soportados: "), "JPEG, PNG, WEBP, BMP"
            ]),
            html.P([
                "", html.Strong("Tamaño máximo: "), "10 MB"
            ]),
        ], className="specs-list")
    ], className="upload-restrictions-simple")

def create_image_upload_form():
    """Formulario de upload de imagen con dropdown de stroke ID"""
    return html.Div([
        html.H3("Subir Imagen para Análisis"),
        
        # Dropdown para seleccionar stroke prediction ID
        html.Div([
            html.Label("Seleccionar Predicción de Stroke:", className="form-label"),
            dcc.Dropdown(
                id='stroke-id-dropdown',
                placeholder="Seleccione una predicción de stroke...",
                className="stroke-id-dropdown",
                disabled=False  # Se controlará por callback
            ),
            html.Div(id='stroke-id-info', className="stroke-id-info")
        ], className="form-group"),
        
        # Upload de imagen
        html.Div([
            html.Label("Subir Imagen de Tomografía:", className="form-label"),
            dcc.Upload(
                id='image-upload',
                children=html.Div([
                    html.I(className="fas fa-cloud-upload-alt upload-icon"),
                    html.P("Arrastra y suelta una imagen aquí o"),
                    html.Button("Seleccionar Archivo", className="btn-upload-select"),
                    html.P("Formatos soportados: JPEG, PNG, WEBP, BMP", className="upload-hint")
                ]),
                className="image-upload-area",
                disabled=False
            ),
            html.Div(id='image-upload-status', className="upload-status")
        ], className="form-group"),
        
        # Botón de análisis
        html.Div([
            html.Button(
                "🔍 Analizar Imagen",
                id='analyze-image-button',
                className="btn-primary btn-analyze",
                disabled=True,
                n_clicks=0
            )
        ], className="form-actions")
        
    ], className="image-upload-form")

def create_image_preview(filename: str, file_content: str):
    """Crear preview de la imagen subida - FUNCIÓN MANTENIDA PARA COMPATIBILIDAD"""
    return html.Div([
        html.H4("📷 Vista Previa"),
        html.Div([
            html.Img(
                src=f"data:image/png;base64,{file_content}",
                className="image-preview"
            ),
            html.Div([
                html.P(f"📁 {filename}", className="preview-filename"),
                html.Button(
                    "❌ Remover",
                    id='remove-image-button',
                    className="btn-remove-image",
                    n_clicks=0
                )
            ], className="preview-info")
        ], className="preview-content")
    ], className="image-preview-card")

def create_image_result_card(prediction: int, probability: float, risk_level: str, 
                           processing_time: int, stroke_id: int, model_confidence: float):
    """Crear tarjeta de resultados de predicción de imagen"""
    
    # Determinar clase CSS y emoji según riesgo
    risk_classes = {
        "Bajo": "result-card-low",
        "Medio": "result-card-medium", 
        "Alto": "result-card-high",
        "Crítico": "result-card-critical"
    }
    
    risk_emojis = {
        "Bajo": "✅",
        "Medio": "⚠️", 
        "Alto": "🚨",
        "Crítico": "🆘"
    }
    
    risk_class = risk_classes.get(risk_level, "result-card-low")
    risk_emoji = risk_emojis.get(risk_level, "✅")
    
    # Mensaje principal
    if prediction == 1:
        main_message = f"{risk_emoji} INDICADORES DE STROKE DETECTADOS"
        diagnosis_class = "diagnosis-positive"
    else:
        main_message = f"✅ NO SE DETECTARON INDICADORES DE STROKE"
        diagnosis_class = "diagnosis-negative"
    
    # Recomendaciones según nivel de riesgo
    recommendations = {
        "Bajo": "Las imágenes no muestran indicadores significativos de stroke. Mantenga controles regulares.",
        "Medio": "Se detectaron algunas anomalías menores. Considere evaluación médica adicional.",
        "Alto": "Se detectaron indicadores importantes. Consulte urgentemente con un neurólogo.",
        "Crítico": "Se detectaron indicadores críticos. Busque atención neurológica inmediata."
    }
    
    return html.Div([
        # Mensaje principal
        html.Div([
            html.H2(main_message, className=f"diagnosis-message {diagnosis_class}"),
            html.Div(f"{probability:.1f}%", className="percentage-display"),
            html.P(f"Nivel de riesgo por imagen: {risk_level}", className="risk-level")
        ], className="result-header"),
        
        # Métricas técnicas
        html.Div([
            html.Div([
                html.Div([
                    html.Span("🎯 Confianza del Modelo"),
                    html.Span(f"{model_confidence:.1f}%", className="metric-value")
                ], className="metric-item"),
                
                html.Div([
                    html.Span("⚡ Tiempo de Procesamiento"),
                    html.Span(f"{processing_time} ms", className="metric-value")
                ], className="metric-item"),
                
                html.Div([
                    html.Span("🔗 ID Stroke Vinculado"),
                    html.Span(f"#{stroke_id}", className="metric-value")
                ], className="metric-item")
            ], className="metrics-grid")
        ], className="technical-metrics"),
        
        # Recomendación
        html.Div([
            html.H4("📋 Interpretación Clínica"),
            html.P(recommendations.get(risk_level, recommendations["Bajo"]))
        ], className="recommendation"),
        
        # Acciones
        html.Div([
            html.Button(
                "📊 Ver Historial Completo",
                id='view-history-from-image',
                className="btn-secondary",
                n_clicks=0
            ),
            html.Button(
                "🔄 Analizar Nueva Imagen",
                id='analyze-new-image',
                className="btn-primary",
                n_clicks=0
            )
        ], className="result-actions")
        
    ], className=f"result-card image-result-card {risk_class}")

def create_stroke_id_options(stroke_predictions: List[Dict]) -> List[Dict]:
    """Crear opciones para el dropdown de stroke IDs"""
    options = []
    
    for pred in stroke_predictions:
        # Formato: "ID #1 - Juan P. (Riesgo Alto, 78.5%)"
        label = (f"ID #{pred.get('id')} - "
                f"{pred.get('patient_name', 'Paciente')} "
                f"({pred.get('risk_level', 'N/A')}, {pred.get('probability', 0):.1f}%)")
        
        options.append({
            'label': label,
            'value': pred.get('id')
        })
    
    return options

def create_processing_animation():
    """Animación de procesamiento de imagen"""
    return html.Div([
        html.Div([
            html.I(className="fas fa-brain processing-icon"),
            html.H3("🧠 Analizando Imagen..."),
            html.P("La red neuronal está procesando la tomografía"),
            html.Div([
                html.Div(className="processing-bar"),
                html.Div(className="processing-fill")
            ], className="processing-progress")
        ], className="processing-content")
    ], className="processing-animation")

def create_upload_error_message(error_msg: str):
    """Mensaje de error para upload"""
    return html.Div([
        html.I(className="fas fa-exclamation-triangle error-icon"),
        html.H4("Error en la Imagen"),
        html.P(error_msg),
        html.Button(
            "🔄 Intentar de Nuevo",
            id='retry-upload-button',
            className="btn-retry",
            n_clicks=0
        )
    ], className="upload-error-message")

def format_file_size(size_bytes: int) -> str:
    """Formatear tamaño de archivo"""
    if size_bytes < 1024:
        return f"{size_bytes} B"
    elif size_bytes < 1024 * 1024:
        return f"{size_bytes / 1024:.1f} KB"
    else:
        return f"{size_bytes / (1024 * 1024):.1f} MB"

def validate_image_file(filename: str, content: str) -> Dict:
    """Validar archivo de imagen antes del upload"""
    try:
        # Decodificar para obtener tamaño
        file_content = base64.b64decode(content.split(',')[1])
        file_size = len(file_content)
        
        # Validar extensión
        valid_extensions = ['.jpg', '.jpeg', '.png', '.webp', '.bmp']
        file_ext = '.' + filename.lower().split('.')[-1]
        
        if file_ext not in valid_extensions:
            return {
                'valid': False,
                'error': f"Formato no soportado. Use: {', '.join(valid_extensions)}"
            }
        
        # Validar tamaño (10MB)
        max_size = 10 * 1024 * 1024  # 10MB
        if file_size > max_size:
            return {
                'valid': False,
                'error': f"Archivo muy grande ({format_file_size(file_size)}). Máximo: 10MB"
            }
        
        return {
            'valid': True,
            'size': file_size,
            'formatted_size': format_file_size(file_size),
            'extension': file_ext
        }
        
    except Exception as e:
        return {
            'valid': False,
            'error': f"Error procesando archivo: {str(e)}"
        }