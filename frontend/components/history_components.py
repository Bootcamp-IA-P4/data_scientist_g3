# components/history_components.py
from dash import html, dash_table

def create_history_table(history_data):
    """
    Crea la tabla del historial de predicciones
    Muestra: ID, Fecha, Edad, Género, Porcentaje, Nivel, Resultado
    """
    
    if not history_data:
        return html.Div([
            html.H3("📊 Historial de Predicciones"),
            html.P("No hay predicciones en el historial o error al cargar datos.",
                  style={'text-align': 'center', 'color': '#6c757d', 'font-style': 'italic'})
        ], className="history-section")
    
    # Preparar datos para la tabla según la guía
    table_data = []
    for pred in history_data:
        # Determinar resultado según prediction
        resultado = "Sin riesgo" if pred.get('prediction', 0) == 0 else "Con riesgo"
        
        # Formatear datos según lo que viene del GET /predictions/stroke
        table_data.append({
            'ID': pred.get('id', ''),
            'Fecha': pred.get('fecha_creacion', ''),  # Ya viene en formato "DD/MM/YYYY HH:MM"
            'Edad': pred.get('age', ''),
            'Género': pred.get('gender', ''),
            'Porcentaje': f"{(pred.get('probability', 0) * 100):.1f}%",  # probability * 100
            'Nivel': pred.get('risk_level', ''),  # risk_level del backend
            'Resultado': resultado
        })
    
    # Crear tabla con estilos
    table = dash_table.DataTable(
        data=table_data,
        columns=[
            {'name': 'ID', 'id': 'ID'},
            {'name': 'Fecha', 'id': 'Fecha'},
            {'name': 'Edad', 'id': 'Edad'},
            {'name': 'Género', 'id': 'Género'},
            {'name': 'Porcentaje', 'id': 'Porcentaje'},
            {'name': 'Nivel', 'id': 'Nivel'},
            {'name': 'Resultado', 'id': 'Resultado'}
        ],
        style_cell={
            'textAlign': 'center', 
            'padding': '10px',
            'fontFamily': 'Segoe UI, Tahoma, Geneva, Verdana, sans-serif'
        },
        style_header={
            'backgroundColor': '#007bff', 
            'color': 'white', 
            'fontWeight': 'bold'
        },
        style_data_conditional=[
            # Colorear filas según resultado
            {
                'if': {'filter_query': '{Resultado} = "Con riesgo"'},
                'backgroundColor': '#f8d7da',
                'color': 'black',
            },
            {
                'if': {'filter_query': '{Resultado} = "Sin riesgo"'},
                'backgroundColor': '#d4edda',
                'color': 'black',
            }
        ],
        sort_action="native",  # Permitir ordenar columnas
        filter_action="native",  # Permitir filtrar
        page_action="native",   # Paginación
        page_current=0,
        page_size=10,
    )
    
    return html.Div([
        html.H3("📊 Historial de Predicciones", 
                style={'color': '#495057', 'margin-bottom': '20px'}),
        table
    ], className="history-section")

def prepare_history_data(raw_data):
    """
    Procesa los datos del historial según la estructura que viene del backend
    
    Estructura esperada del GET /predictions/stroke:
    pred['id']                    # 1, 2, 3...
    pred['fecha_creacion']        # "12/06/2025 14:30"  
    pred['gender']                # "Masculino", "Femenino"
    pred['age']                   # 45, 67...
    pred['hypertension']          # "Sí", "No"
    pred['heart_disease']         # "Sí", "No"
    pred['ever_married']          # "Sí", "No"
    pred['work_type']             # "Privado", "Empleado Público"...
    pred['residence_type']        # "Urbano", "Rural"
    pred['avg_glucose_level']     # 120.5, 180.2...
    pred['bmi']                   # 25.3, 28.7... (puede ser None)
    pred['smoking_status']        # "Nunca fumó", "Fuma"...
    pred['prediction']            # 0, 1
    pred['probability']           # 0.1234, 0.7892...
    pred['risk_level']            # "Bajo", "Medio", "Alto", "Crítico"
    """
    
    if not raw_data:
        return []
    
    processed_data = []
    
    for item in raw_data:
        # Validar que tenemos los campos mínimos necesarios
        if not all(key in item for key in ['id', 'prediction', 'probability']):
            continue
        
        processed_item = {
            'id': item.get('id', ''),
            'fecha_creacion': item.get('fecha_creacion', ''),
            'age': item.get('age', ''),
            'gender': item.get('gender', ''),
            'prediction': item.get('prediction', 0),
            'probability': item.get('probability', 0.0),
            'risk_level': item.get('risk_level', 'Bajo')
        }
        
        processed_data.append(processed_item)
    
    return processed_data