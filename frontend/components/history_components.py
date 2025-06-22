from datetime import datetime
from typing import List, Dict
from dash import html
try:
    from dash import dash_table
except ImportError:
    # Fallback si dash_table no está disponible
    dash_table = None


def create_history_table(history_data):
    """
    Crea la tabla del historial de predicciones
    Muestra: ID, Fecha, Edad, Género, Porcentaje, Nivel, Resultado
    """
    
    if not history_data:
        return html.Div([
            html.H3("Historial de Predicciones"),
            html.P("No hay predicciones en el historial o error al cargar datos.",
                  style={'text-align': 'center', 'color': "#6c757d", 'font-style': 'italic'})
        ], className="history-section")
    
    # Preparar datos para la tabla según la guía
    table_data = []
    try:
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
    except Exception as e:
        print(f"Error procesando datos del historial: {e}")
        return html.Div([
            html.H3("Historial de Predicciones"),
            html.P("Error al procesar los datos del historial.",
                  style={'text-align': 'center', 'color': 'red'})
        ], className="history-section")
    
    # Verificar si dash_table está disponible
    if dash_table is None:
        # Crear tabla HTML simple si dash_table no funciona
        table_rows = []
        
        # Header
        header = html.Tr([
            html.Th('ID'), html.Th('Fecha'), html.Th('Edad'), 
            html.Th('Género'), html.Th('Porcentaje'), html.Th('Nivel'), html.Th('Resultado')
        ])
        
        # Filas de datos
        for row in table_data:
            table_rows.append(html.Tr([
                html.Td(row['ID']),
                html.Td(row['Fecha']),
                html.Td(row['Edad']),
                html.Td(row['Género']),
                html.Td(row['Porcentaje']),
                html.Td(row['Nivel']),
                html.Td(row['Resultado'])
            ]))
        
        table = html.Table([
            html.Thead(header),
            html.Tbody(table_rows)
        ], style={
            'width': '100%',
            'border-collapse': 'collapse',
            'margin': '20px 0'
        })
    else:
        # Crear tabla con dash_table si está disponible
        try:
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
                filter_action="none",  # Permitir filtrar
                page_action="native",   # Paginación
                page_current=0,
                page_size=10,
            )
        except Exception as e:
            print(f"Error creando DataTable: {e}")
            # Fallback a tabla HTML
            table = html.P("Error al crear la tabla del historial.")
    
    return html.Div([
        html.H3("Historial de Predicciones", 
                style={'color': '#495057', 'margin-bottom': '20px'}),
        table
    ], className="history-section")


# En history_components.py, reemplaza la función create_combined_history_table:

def create_combined_history_table(stroke_data: List[Dict], image_data: List[Dict]):
    """
    Crea la tabla combinada del historial de predicciones de stroke e imagen
    ✅ ARREGLADO: Botón de tomografía funciona correctamente
    """
    
    if not stroke_data:
        return html.Div([
            html.H3("Historial Combinado de Predicciones"),
            html.P("No hay predicciones en el historial.",
                  style={'text-align': 'center', 'color': "#6c757d", 'font-style': 'italic'})
        ], className="history-section")
    
    # Crear mapa de imágenes por stroke_prediction_id
    images_by_stroke_id = {}
    for img in image_data:
        stroke_id = img.get('stroke_prediction_id')
        if stroke_id:
            images_by_stroke_id[stroke_id] = img
    
    # Combinar datos
    combined_table_data = []
    try:
        for stroke in stroke_data:
            stroke_id = stroke.get('id')
            image_info = images_by_stroke_id.get(stroke_id, None)
            
            # Datos de stroke
            stroke_resultado = "Sin riesgo" if stroke.get('prediction', 0) == 0 else "Con riesgo"
            stroke_percentage = f"{(stroke.get('probability', 0) * 100):.1f}%"
            
            # Datos de imagen
            if image_info:
                image_percentage = f"{(image_info.get('probability', 0) * 100):.1f}%"
                image_risk = image_info.get('risk_level', 'N/A')
                image_status = "✅ Completado"
            else:
                image_percentage = "N/A"
                image_risk = "N/A"
                # ✅ CAMBIO CRÍTICO: Formato markdown para crear enlace
                image_status = f"[AÑADIR TOMOGRAFÍA](/image-prediction?stroke_id={stroke_id})"
            
            combined_table_data.append({
                'ID': stroke_id,
                'Fecha': stroke.get('fecha_creacion', ''),
                'Edad': stroke.get('age', ''),
                'Género': stroke.get('gender', ''),
                'Stroke %': stroke_percentage,
                'Riesgo Stroke': stroke.get('risk_level', ''),
                'Estado Imagen': image_status,
                'Imagen %': image_percentage,
                'Riesgo Imagen': image_risk
            })
            
    except Exception as e:
        print(f"Error procesando datos combinados: {e}")
        return html.Div([
            html.H3("Historial Combinado de Predicciones"),
            html.P("Error al procesar los datos del historial.",
                  style={'text-align': 'center', 'color': 'red'})
        ], className="history-section")
    
    # Crear tabla
    if dash_table is None:
        table = create_html_combined_table(combined_table_data)
    else:
        try:
            # ✅ ARREGLADO: DataTable con botones funcionales y cabeceras actualizadas
            table = dash_table.DataTable(
                id='combined-history-table',
                data=combined_table_data,
                columns=[
                    {'name': 'ID', 'id': 'ID', 'type': 'numeric'},
                    {'name': 'Fecha', 'id': 'Fecha'},
                    {'name': 'Edad', 'id': 'Edad', 'type': 'numeric'},
                    {'name': 'Género', 'id': 'Género'},
                    {'name': 'Stroke %', 'id': 'Stroke %'},
                    {'name': 'Riesgo', 'id': 'Riesgo Stroke'},
                    {'name': 'Tomografía', 'id': 'Estado Imagen', 'presentation': 'markdown'},  # ✅ CRUCIAL: markdown activado
                    {'name': 'Stroke Tomografía', 'id': 'Imagen %'},
                    {'name': 'Riesgo Tomografía', 'id': 'Riesgo Imagen'}
                ],
                style_cell={
                    'textAlign': 'center', 
                    'padding': '12px 8px',
                    'fontFamily': 'Inter, sans-serif',
                    'fontSize': '0.9rem'
                },
                style_header={
                    'backgroundColor': 'linear-gradient(135deg, #2563EB, #8B5CF6)', 
                    'color': 'white', 
                    'fontWeight': 'bold',
                    'border': 'none'
                },
                style_data={
                    'backgroundColor': 'rgba(30, 41, 59, 0.3)',
                    'color': '#F8FAFC',
                    'border': '1px solid rgba(255, 255, 255, 0.1)'
                },
                style_data_conditional=[
                    # Filas con riesgo alto de stroke
                    {
                        'if': {'filter_query': '{Riesgo Stroke} = "Alto"'},
                        'backgroundColor': 'rgba(239, 68, 68, 0.2)',
                        'color': '#FEF2F2',
                    },
                    {
                        'if': {'filter_query': '{Riesgo Stroke} = "Crítico"'},
                        'backgroundColor': 'rgba(239, 68, 68, 0.3)',
                        'color': '#FEF2F2',
                    },
                    # ✅ Filas con botón de añadir tomografía
                    {
                        'if': {'filter_query': '{Estado Imagen} contains "AÑADIR"'},
                        'backgroundColor': 'rgba(245, 158, 11, 0.1)',
                        'border': '1px solid rgba(245, 158, 11, 0.3)'
                    }
                ],
                sort_action="native",
                filter_action="native",
                page_action="native",
                page_current=0,
                page_size=15,
                style_table={'overflowX': 'auto'},
                # ✅ CRÍTICO: Habilitar markdown
                markdown_options={
                    "html": False,
                    "link_target": "_self"
                }
            )
        except Exception as e:
            print(f"Error creando DataTable combinada: {e}")
            table = html.P("Error al crear la tabla combinada del historial.")
    
    return html.Div([
        html.H3("Historial Combinado de Predicciones"),
        table
    ], className="history-section combined-history")


def create_html_combined_table(table_data: List[Dict]):
    """
    Crear tabla HTML combinada como fallback
    ✅ ARREGLADO: Botones de tomografía funcionan con hrefs y cabeceras actualizadas
    """
    if not table_data:
        return html.P("No hay datos para mostrar.")
    
    # ✅ CABECERAS ACTUALIZADAS
    headers = ['ID', 'Fecha', 'Edad', 'Género', 'Stroke %', 'Riesgo', 'Tomografía', 'Stroke Tomografía', 'Riesgo Tomografía']
    columns = ['ID', 'Fecha', 'Edad', 'Género', 'Stroke %', 'Riesgo Stroke', 'Estado Imagen', 'Imagen %', 'Riesgo Imagen']
    
    # Header
    header_row = html.Tr([html.Th(header) for header in headers])
    
    # Filas de datos
    table_rows = []
    for row in table_data:
        cells = []
        for col in columns:
            cell_value = row.get(col, '')
            
            # ✅ ARREGLADO: Tratamiento especial para botón de añadir tomografía
            if col == 'Estado Imagen' and str(cell_value) == "Añadir Tomografía":
                stroke_id = row.get('ID')  # ✅ Obtener stroke_id de la fila actual
                cells.append(html.Td([
                    html.A(
                        "Añadir Tomografía",
                        href=f"/image-prediction?stroke_id={stroke_id}",
                        className="btn-add-image-small",
                        target="_self"
                    )
                ]))
            else:
                # ✅ Mostrar texto normal para casos completados
                cells.append(html.Td(cell_value))
        
        table_rows.append(html.Tr(cells))
    
    return html.Table([
        html.Thead(header_row),
        html.Tbody(table_rows)
    ], className="fallback-table", style={
        'width': '100%',
        'border-collapse': 'collapse',
        'margin': '20px 0'
    })


def create_history_stats_summary(stroke_data: List[Dict], image_data: List[Dict]):
    """
    Crear resumen de estadísticas del historial
    """
    total_stroke = len(stroke_data)
    total_images = len(image_data)
    completion_rate = (total_images / total_stroke * 100) if total_stroke > 0 else 0
    
    # Análisis de riesgo
    high_risk_count = sum(1 for s in stroke_data if s.get('risk_level') in ['Alto', 'Crítico'])
    high_risk_percentage = (high_risk_count / total_stroke * 100) if total_stroke > 0 else 0
    
    return html.Div([
        html.H3("Resumen del Historial"),
        
        html.Div([
            html.Div([
                html.Div("📋", className="stat-icon"),
                html.Div([
                    html.H4(str(total_stroke)),
                    html.P("Predicciones Stroke")
                ], className="stat-content")
            ], className="stat-card"),
            
            html.Div([
                html.Div("📷", className="stat-icon"),
                html.Div([
                    html.H4(str(total_images)),
                    html.P("Análisis de Imagen")
                ], className="stat-content")
            ], className="stat-card"),
            
            html.Div([
                html.Div("✅", className="stat-icon"),
                html.Div([
                    html.H4(f"{completion_rate:.1f}%"),
                    html.P("Completitud")
                ], className="stat-content")
            ], className="stat-card"),
            
            html.Div([
                html.Div("🚨", className="stat-icon"),
                html.Div([
                    html.H4(f"{high_risk_percentage:.1f}%"),
                    html.P("Alto Riesgo")
                ], className="stat-content")
            ], className="stat-card")
            
        ], className="stats-grid")
    ], className="history-stats-section")


def filter_combined_data(combined_data: List[Dict], risk_filter: str, image_status_filter: str):
    """
    Filtrar datos combinados según criterios seleccionados
    """
    filtered_data = combined_data.copy()
    
    # Filtro por riesgo de stroke
    if risk_filter != 'all':
        filtered_data = [item for item in filtered_data if item.get('Riesgo Stroke') == risk_filter]
    
    # Filtro por estado de imagen
    if image_status_filter == 'with_image':
        filtered_data = [item for item in filtered_data if 'Completado' in item.get('Estado Imagen', '')]
    elif image_status_filter == 'without_image':
        filtered_data = [item for item in filtered_data if 'Añadir' in item.get('Estado Imagen', '')]
    
    return filtered_data


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