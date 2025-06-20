import dash
from dash import dcc, html, Input, Output, State, callback
from config.settings import FRONTEND_PORT
from pages.about import get_about_layout
from pages.history import get_history_layout
from services.api_client import api_client
from components.form_components import (create_form_layout, validate_form_data, prepare_form_data)
from components.results_components import (create_result_card, create_error_message, create_disclaimer)
from components.history_components import (create_combined_history_table, create_history_stats_summary, filter_combined_data)
from components.navbar_components import create_navbar
from pages.image_prediction import get_image_prediction_layout
from components.image_components import (create_image_preview, create_image_result_card, create_processing_animation, 
    create_upload_error_message,
    create_stroke_id_options,
    validate_image_file
)
import base64

def resolve_latest_stroke_id():
    """Resolver 'LATEST' al ID más reciente"""
    try:
        history_data = api_client.get_predictions_history()
        if history_data and len(history_data) > 0:
            return history_data[0].get('id')
        return None
    except:
        return None

# Inicializar la aplicación Dash
app = dash.Dash(
    __name__,
    suppress_callback_exceptions=True,  # Oculta errores de callbacks
    show_undo_redo=False,              # Oculta botones undo/redo
    external_stylesheets=['assets/style.css', 'assets/about.css', 'assets/image_prediction.css', 'assets/history.css'] # Añadir estilos de historial            
)

app.title = "Predictor de Riesgo de Stroke"

# Layout principal
def get_home_layout():
    return html.Div([
        
        html.Div([
        html.Video(
            src='assets/background-video.mp4',
            autoPlay=True,
            muted=True,
            loop=True
        )
        ], className="video-background"),
        
        # Overlay oscuro
        html.Div(className="video-overlay"),
        
        create_navbar(),
        
        # Contenido principal
        html.Div([
        html.H1("Predictor de Riesgo de Stroke"),
        
        create_form_layout(),
        
        html.Div(id='results-container'),
        
        # Disclaimer médico
        create_disclaimer(),
        
        # Store para guardar datos temporalmente
        dcc.Store(id='prediction-store')
        ], style={'position': 'relative', 'z-index': '1'})
    ])

app.layout = html.Div([
    dcc.Location(id='url', refresh=False),
    html.Div(id='page-content')
])

@callback(
    Output('page-content', 'children'),
    [Input('url', 'pathname')]
)
def display_page(pathname):
    if pathname == '/about':
        return get_about_layout()
    elif pathname == '/image-prediction':
        return get_image_prediction_layout()
    elif pathname == '/history':
        return get_history_layout()
    else:
        return get_home_layout()

# Callback para manejar predicciones
@callback(
    [Output('results-container', 'children'),
     Output('prediction-store', 'data')],
    [Input('predict-button', 'n_clicks')],
    [State('edad-input', 'value'),
     State('genero-dropdown', 'value'),
     State('glucosa-input', 'value'),
     State('bmi-input', 'value'),
     State('hipertension-dropdown', 'value'),
     State('enfermedad-dropdown', 'value'),
     State('casado-dropdown', 'value'),
     State('trabajo-dropdown', 'value'),
     State('residencia-dropdown', 'value'),
     State('fumador-dropdown', 'value')]
)
def handle_prediction(n_clicks, edad, genero, glucosa, bmi, hipertension, 
                     enfermedad, casado, trabajo, residencia, fumador):
    """
    Maneja la predicción de stroke:
    1. Valida los datos del formulario
    2. Envía los datos al backend
    3. Procesa la respuesta
    4. Muestra los resultados
    """
    
    if n_clicks == 0:
        return html.Div(), {}
    
    # Validar campos obligatorios
    is_valid, error_message = validate_form_data(
        edad, genero, glucosa, bmi, hipertension, 
        enfermedad, casado, trabajo, residencia, fumador
    )
    
    if not is_valid:
        return create_error_message(error_message), {}
    
    # Preparar datos para enviar al backend
    form_data = prepare_form_data(
        edad, genero, glucosa, bmi, hipertension,
        enfermedad, casado, trabajo, residencia, fumador
    )
    
    # Enviar predicción al backend
    result = api_client.predict_stroke(form_data)
    
    if 'error' in result:
        return create_error_message(f"❌ {result['error']}"), {}
    
    # Extraer resultados del backend
    prediction = result.get('prediction', 0)
    probability = result.get('probability', 0.0)
    risk_level = result.get('risk_level', 'Bajo')
    
    # Crear tarjeta de resultados
    result_card = create_result_card(prediction, probability, risk_level, show_image_button=True)
    
    return result_card, result

# Callbacks para página de historial
@callback(
    [Output('history-stats-container', 'children'),
     Output('combined-history-container', 'children'),
     Output('stroke-history-store', 'data'),
     Output('image-history-store', 'data'),
     Output('combined-history-store', 'data')],
    [Input('url', 'pathname'),
     Input('refresh-history-button', 'n_clicks')],
    prevent_initial_call=False
)
def load_history_data(pathname, refresh_clicks):
    """Cargar datos del historial combinado"""
    
    if pathname != '/history':
        return "", "", [], [], {}
    
    try:
        print("📊 Cargando datos del historial combinado...")
        
        # Obtener datos combinados
        combined_data = api_client.get_combined_predictions_history()
        
        if not combined_data.get('success', False):
            error_msg = combined_data.get('error', 'Error desconocido')
            return (
                html.Div([
                    html.H3("❌ Error al cargar historial"),
                    html.P(f"Error: {error_msg}")
                ], className="history-error"),
                "",
                [],
                [],
                {}
            )
        
        stroke_data = combined_data['stroke_data']
        image_data = combined_data['image_data']
        stats = combined_data['stats']
        
        print(f"✅ Datos cargados - Stroke: {len(stroke_data)}, Imágenes: {len(image_data)}")
        
        # Crear componentes
        stats_component = create_history_stats_summary(stroke_data, image_data)
        table_component = create_combined_history_table(stroke_data, image_data)
        
        return (
            stats_component,
            table_component,
            stroke_data,
            image_data,
            combined_data
        )
        
    except Exception as e:
        print(f"❌ Error cargando historial: {e}")
        return (
            html.Div([
                html.H3("❌ Error al cargar historial"),
                html.P(f"Error: {str(e)}")
            ], className="history-error"),
            "",
            [],
            [],
            {}
        )

@callback(
    Output('combined-history-container', 'children', allow_duplicate=True),
    [Input('risk-filter-dropdown', 'value'),
     Input('image-status-filter-dropdown', 'value')],
    [State('stroke-history-store', 'data'),
     State('image-history-store', 'data'),
     State('url', 'pathname')],
    prevent_initial_call=True
)
def filter_history_table(risk_filter, image_status_filter, stroke_data, image_data, pathname):
    """Filtrar tabla de historial según criterios seleccionados"""
    
    if pathname != '/history' or not stroke_data:
        return dash.no_update
    
    try:
        # Aplicar filtros y recrear tabla
        filtered_table = create_combined_history_table(stroke_data, image_data)
        return filtered_table
        
    except Exception as e:
        print(f"❌ Error filtrando historial: {e}")
        return html.Div([
            html.P(f"Error aplicando filtros: {str(e)}")
        ], className="filter-error")

@callback(
    Output('stroke-id-dropdown', 'options'),
    Output('stroke-id-dropdown', 'disabled'),
    Output('stroke-id-dropdown', 'value'),
    Output('stroke-id-info', 'children'),
    [Input('url', 'pathname'),
     Input('url', 'search')],
    prevent_initial_call=False
)
def load_stroke_predictions_for_dropdown(pathname, search):
    """Cargar predicciones de stroke disponibles para el dropdown"""
    
    if pathname != '/image-prediction':
        return [], True, None, ""
    
    # Obtener predicciones de stroke del backend
    stroke_data = api_client.get_predictions_history()
    
    if not stroke_data:
        return [], True, None, html.Div([
            html.I(className="fas fa-exclamation-triangle"),
            html.Span("No hay predicciones de stroke disponibles. "),
            html.A("Crear nueva predicción", href="/", className="link-primary")
        ], className="no-predictions-warning")
    
    # Verificar si viene con LATEST y resolverlo
    if search and 'stroke_id=LATEST' in search:
        latest_id = resolve_latest_stroke_id()
        if latest_id:
            options = create_stroke_id_options(stroke_data)
            selected_pred = next((p for p in stroke_data if p.get('id') == latest_id), None)
            
            if selected_pred:
                info_msg = html.Div([
                    html.I(className="fas fa-link"),
                    html.Span(f"Vinculado a predicción recién realizada (ID #{latest_id})")
                ], className="stroke-id-preselected")
                
                return options, True, latest_id, info_msg
    
    # Verificar si viene de una predicción específica (ID numérico)
    if search and 'stroke_id=' in search and 'LATEST' not in search:
        try:
            stroke_id = int(search.split('stroke_id=')[1].split('&')[0])
            options = create_stroke_id_options(stroke_data)
            selected_pred = next((p for p in stroke_data if p.get('id') == stroke_id), None)
            
            if selected_pred:
                info_msg = html.Div([
                    html.I(className="fas fa-link"),
                    html.Span(f"Vinculado a predicción específica (ID #{stroke_id})")
                ], className="stroke-id-preselected")
                
                return options, True, stroke_id, info_msg
        except ValueError:
            pass  # Si no es un número válido, continuar
    
    # Dropdown habilitado para selección libre
    options = create_stroke_id_options(stroke_data)
    info_msg = html.Div([
        html.I(className="fas fa-info-circle"),
        html.Span(f"Seleccione una de las {len(stroke_data)} predicciones disponibles")
    ], className="stroke-id-selection-info")
    
    return options, False, None, info_msg

# Callback para manejar clicks en botones de "Añadir Imagen" desde la tabla de historial  
@callback(
    Output('url', 'href', allow_duplicate=True),
    [Input('combined-history-table', 'active_cell')],
    [State('combined-history-table', 'data'),
     State('url', 'pathname')],
    prevent_initial_call=True
)
def handle_add_image_from_history(active_cell, table_data, current_pathname):
    """
    Manejar click en botón 'Añadir Tomografía' desde tabla de historial
    """
    print(f"🔍 DEBUG handle_add_image_from_history:")
    print(f"  - current_pathname: {current_pathname}")
    print(f"  - active_cell: {active_cell}")
    print(f"  - table_data length: {len(table_data) if table_data else 0}")
    
    # Solo funcionar en la página de historial
    if current_pathname != '/history':
        print("❌ No está en página de historial")
        return dash.no_update
    
    # Verificar que tenemos los datos necesarios
    if not active_cell or not table_data:
        print("❌ No hay active_cell o table_data")
        return dash.no_update
    
    try:
        row_index = active_cell.get('row')
        column_id = active_cell.get('column_id')
        
        print(f"  - row_index: {row_index}")
        print(f"  - column_id: {column_id}")
        
        # Verificar si se hizo click en la columna de imagen
        if column_id == 'Imagen' and row_index is not None:
            row_data = table_data[row_index]
            imagen_status = row_data.get('Imagen', '')
            stroke_id = row_data.get('ID')
            
            print(f"  - imagen_status: {imagen_status}")
            print(f"  - stroke_id: {stroke_id}")
            
            # Verificar si es un caso sin imagen (contiene "Añadir")
            if 'Añadir' in str(imagen_status) and stroke_id:
                navigation_url = f"/image-prediction?stroke_id={stroke_id}&origin=history"
                print(f"✅ Navegando a: {navigation_url}")
                return navigation_url
            else:
                print("❌ No es una celda de 'Añadir' o no hay stroke_id")
        else:
            print("❌ No es la columna 'Imagen'")
    
    except (IndexError, KeyError, TypeError) as e:
        print(f"❌ Error procesando click: {e}")
        return dash.no_update
    
    # No redirigir si no cumple condiciones
    print("❌ No cumple condiciones para navegación")
    return dash.no_update

@callback(
    Output('image-results-container', 'children'),
    [Input('analyze-image-button', 'n_clicks')],
    [State('image-upload', 'contents'),
     State('image-upload', 'filename'),
     State('stroke-id-dropdown', 'value'),
     State('url', 'pathname')],
    prevent_initial_call=True
)
def analyze_image_real(n_clicks, image_contents, filename, stroke_id, pathname):
    """
    Procesa imagen real y la envía al backend
    """
    
    print(f"🔍 DEBUG analyze_image_real llamado:")
    print(f"  - n_clicks: {n_clicks}")
    print(f"  - pathname: {pathname}")
    print(f"  - stroke_id: {stroke_id}")
    print(f"  - filename: {filename}")
    print(f"  - image_contents: {'SÍ' if image_contents else 'NO'}")
    
    # Solo funcionar en la página de imagen
    if pathname != '/image-prediction':
        return ""
    
    # Solo procesar si se hizo click
    if n_clicks == 0 or not n_clicks:
        return ""
    
    # Validar que tenemos todos los datos necesarios
    if not image_contents:
        return create_upload_error_message("No hay imagen seleccionada. Por favor suba una imagen primero.")
    
    if not stroke_id:
        return create_upload_error_message("No hay predicción de stroke seleccionada. Por favor seleccione una predicción primero.")
    
    if not filename:
        filename = "imagen_tomografia.jpg"  # Nombre por defecto
    
    print(f"✅ Iniciando análisis de imagen real...")
    print(f"✅ Stroke ID: {stroke_id}")
    print(f"✅ Filename: {filename}")
    
    # Mostrar animación de procesamiento primero
    processing_msg = create_processing_animation()
    
    try:
        # Enviar imagen real al backend
        result = api_client.predict_image(
            image_contents=image_contents,
            stroke_prediction_id=stroke_id,
            filename=filename
        )
        
        print(f"📊 Resultado del backend: {result}")
        
        # Manejar errores del backend
        if 'error' in result:
            error_msg = result['error']
            print(f"❌ Error del backend: {error_msg}")
            return create_upload_error_message(f"Error del servidor: {error_msg}")
        
        # Procesar respuesta exitosa
        prediction = result.get('prediction', 0)
        probability = result.get('probability', 0.0)
        risk_level = result.get('risk_level', 'Bajo')
        processing_time = result.get('processing_time_ms', 0)
        model_confidence = result.get('model_confidence', 0.0)
        message = result.get('message', 'Imagen procesada correctamente')
        
        print(f"✅ Predicción exitosa:")
        print(f"   - Prediction: {prediction}")
        print(f"   - Probability: {probability}")
        print(f"   - Risk Level: {risk_level}")
        print(f"   - Processing Time: {processing_time} ms")
        print(f"   - Model Confidence: {model_confidence}")
        print(f"   - Message: {message}")
        
        # Crear tarjeta de resultados
        result_card = create_image_result_card(
            prediction=prediction,
            probability=probability,
            risk_level=risk_level,
            processing_time=processing_time,
            stroke_id=stroke_id,
            model_confidence=model_confidence,
            message=message
        )
        
        print(f"✅ Tarjeta de resultados creada exitosamente")
        return result_card
        
    except Exception as e:
        print(f"❌ Error inesperado procesando imagen: {e}")
        return create_upload_error_message(f"Error inesperado: {str(e)}")

@callback(
    [Output('image-upload-status', 'children'),
     Output('image-upload', 'children'),
     Output('analyze-image-button', 'disabled')],
    [Input('image-upload', 'contents')],
    [State('image-upload', 'filename'),
     State('stroke-id-dropdown', 'value'),
     State('url', 'pathname')],
    prevent_initial_call=False
)
def handle_image_upload(contents, filename, stroke_id, pathname):
    """Manejar upload de imagen y mostrar preview dentro del área de upload"""
    
    print(f"🔍 DEBUG handle_image_upload llamado:")
    print(f"  - pathname: {pathname}")
    print(f"  - contents: {'SÍ' if contents else 'NO'}")
    print(f"  - filename: {filename}")
    print(f"  - stroke_id: {stroke_id}")
    
    # Área de upload por defecto
    upload_area_default = html.Div([
        html.I(className="fas fa-cloud-upload-alt upload-icon"),
        html.P("Arrastra y suelta una imagen aquí o"),
        html.Button("Seleccionar Archivo", className="btn-upload-select"),
        html.P("Formatos soportados: JPEG, PNG, WEBP, BMP", className="upload-hint")
    ])
    
    # Solo funcionar en la página de imagen
    if pathname != '/image-prediction':
        print("❌ No es página de imagen, retornando default")
        return "", upload_area_default, True
    
    # Si no hay contenido, mostrar área por defecto
    if not contents:
        print("❌ No hay contenido, retornando default")
        return "", upload_area_default, True
    
    # Si no hay stroke_id seleccionado
    if not stroke_id:
        print("❌ No hay stroke_id, retornando warning")
        warning_msg = html.Div([
            html.I(className="fas fa-exclamation-triangle"),
            html.Span("Primero seleccione una predicción de stroke")
        ], className="upload-warning")
        return warning_msg, upload_area_default, True
    
    print(f"✅ Procesando archivo: {filename}")
    print(f"✅ Stroke ID: {stroke_id}")
    print(f"✅ Contenido length: {len(contents) if contents else 0}")
    
    # Validar archivo
    validation = validate_image_file(filename, contents)
    print(f"✅ Validación resultado: {validation}")
    
    if not validation['valid']:
        print(f"❌ Validación falló: {validation['error']}")
        error_msg = create_upload_error_message(validation['error'])
        return error_msg, upload_area_default, True
    
    # Crear preview dentro del área de upload
    try:
        print("✅ Creando preview de imagen...")
        
        # Contenido con imagen para el área de upload
        upload_area_with_image = html.Div([
            html.Div([
                html.Img(
                    src=contents,
                    className="uploaded-image-preview",
                    style={
                        'max-width': '250px',
                        'max-height': '250px',
                        'border-radius': '12px',
                        'border': '3px solid #3B82F6',
                        'box-shadow': '0 8px 25px rgba(37, 99, 235, 0.3)',
                        'object-fit': 'cover',
                        'display': 'block',
                        'margin': '0 auto'
                    }
                ),
                html.Div([
                    html.Button(
                        "Remover",
                        id='remove-image-button',
                        className="btn-remove-uploaded",
                        n_clicks=0,
                        style={
                            'background': '#EF4444',
                            'color': 'white',
                            'border': 'none',
                            'padding': '10px 20px',
                            'border-radius': '8px',
                            'font-weight': '600',
                            'cursor': 'pointer',
                            'margin': '5px'
                        }
                    ),
                    html.Button(
                        "Cambiar",
                        className="btn-change-image",
                        style={
                            'background': 'rgba(255, 255, 255, 0.1)',
                            'color': 'white',
                            'border': '2px solid #6B7280',
                            'padding': '8px 16px',
                            'border-radius': '8px',
                            'font-weight': '600',
                            'cursor': 'pointer',
                            'margin': '5px'
                        }
                    )
                ], className="image-actions", style={
                    'display': 'flex',
                    'gap': '15px',
                    'justify-content': 'center',
                    'margin-top': '20px'
                })
            ], className="image-preview-wrapper", style={
                'display': 'flex',
                'flex-direction': 'column',
                'align-items': 'center',
                'gap': '20px',
                'padding': '20px'
            })
        ])
        
        # Mensaje de éxito
        success_msg = html.Div([
            html.I(className="fas fa-check-circle"),
            html.Span(f"✅ Imagen cargada: {filename} ({validation['formatted_size']})")
        ], className="upload-success")
        
        print(f"✅ Preview creado exitosamente")
        print(f"✅ Retornando: success_msg, upload_area_with_image, False")
        
        return success_msg, upload_area_with_image, False
        
    except Exception as e:
        print(f"❌ Error creando preview: {e}")
        error_msg = create_upload_error_message(f"Error procesando imagen: {str(e)}")
        return error_msg, upload_area_default, True

@callback(
    [Output('image-upload', 'children', allow_duplicate=True),
     Output('image-upload-status', 'children', allow_duplicate=True),
     Output('analyze-image-button', 'disabled', allow_duplicate=True),
     Output('image-upload', 'contents', allow_duplicate=True)],
    [Input('remove-image-button', 'n_clicks')],
    [State('url', 'pathname')],
    prevent_initial_call=True
)
def remove_image_preview(n_clicks, pathname):
    """Remover preview de imagen y restaurar área de upload"""
    
    print(f"🔍 DEBUG remove_image_preview llamado:")
    print(f"  - n_clicks: {n_clicks}")
    print(f"  - pathname: {pathname}")
    
    upload_area_default = html.Div([
        html.I(className="fas fa-cloud-upload-alt upload-icon"),
        html.P("Arrastra y suelta una imagen aquí o"),
        html.Button("Seleccionar Archivo", className="btn-upload-select"),
        html.P("Formatos soportados: JPEG, PNG, WEBP, BMP", className="upload-hint")
    ])
    
    if pathname != '/image-prediction':
        print("❌ No es página de imagen")
        return upload_area_default, "", True, None
        
    if n_clicks and n_clicks > 0:
        print("✅ Removiendo imagen y restaurando área de upload")
        return upload_area_default, "", True, None
    
    print("❌ No hay clicks, manteniendo estado")
    return dash.no_update, dash.no_update, dash.no_update, dash.no_update

# Callback para refrescar datos automáticamente
@callback(
    Output('stroke-predictions-store', 'data'),
    [Input('url', 'pathname')],
    prevent_initial_call=False
)
def load_stroke_predictions_data(pathname):
    """Cargar datos de predicciones de stroke al entrar a la página"""
    
    if pathname == '/image-prediction':
        try:
            # Obtener predicciones más recientes
            predictions = api_client.get_predictions_history()
            return predictions
        except Exception as e:
            print(f"Error cargando predicciones: {e}")
            return []
    
    return []

# Callback para navegación rápida desde acciones
@callback(
    Output('url', 'href', allow_duplicate=True),
    [Input('new-prediction-quick', 'n_clicks'),
     Input('upload-image-quick', 'n_clicks'),
     Input('view-stats-quick', 'n_clicks')],
    prevent_initial_call=True
)
def handle_quick_actions(new_pred_clicks, upload_img_clicks, stats_clicks):
    """Manejar acciones rápidas de navegación"""
    
    ctx = dash.callback_context
    if not ctx.triggered:
        return dash.no_update
    
    button_id = ctx.triggered[0]['prop_id'].split('.')[0]
    
    if button_id == 'new-prediction-quick':
        return "/"
    elif button_id == 'upload-image-quick':
        return "/image-prediction"
    elif button_id == 'view-stats-quick':
        return "/history"
    
    return dash.no_update

# Callback para manejo de errores globales
@callback(
    Output('page-content', 'children', allow_duplicate=True),
    [Input('url', 'pathname')],
    prevent_initial_call=True
)
def handle_page_errors(pathname):
    """Manejar errores de carga de página"""
    
    try:
        if pathname == '/about':
            return get_about_layout()
        elif pathname == '/image-prediction':
            return get_image_prediction_layout()
        elif pathname == '/history':
            return get_history_layout()
        else:
            return get_home_layout()
            
    except Exception as e:
        print(f"Error cargando página {pathname}: {e}")
        
        # Página de error
        return html.Div([
            html.Div([
                html.Video(
                    src='assets/background-video.mp4',
                    autoPlay=True,
                    muted=True,
                    loop=True
                )
            ], className="video-background"),
            
            html.Div(className="video-overlay"),
            create_navbar(),
            
            html.Div([
                html.H1("❌ Error de Carga"),
                html.P(f"No se pudo cargar la página: {pathname}"),
                html.P(f"Error: {str(e)}"),
                html.A("🏠 Volver al Inicio", href="/", className="btn-primary")
            ], className="error-page-content")
        ])

if __name__ == '__main__':
    app.run(debug=True, host='127.0.0.1', port=FRONTEND_PORT)