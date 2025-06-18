import dash
from dash import dcc, html, Input, Output, State, callback
from config.settings import FRONTEND_PORT
from pages.about import get_about_layout
from services.api_client import api_client
from components.form_components import (create_form_layout, validate_form_data, prepare_form_data)
from components.results_components import (create_result_card, create_error_message, create_disclaimer)
from components.history_components import create_history_table
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
    external_stylesheets=['assets/style.css', 'assets/about.css', 'assets/image_prediction.css'] # Sin estilos externos que puedan mostrar debugging            
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
        
        html.Div(id='history-container'),
        
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

# Callback para mostrar historial
@callback(
    Output('history-container', 'children'),
    [Input('history-button', 'n_clicks')]
)
def show_history(n_clicks):
    """
    Maneja la visualización del historial:
    1. Obtiene datos del backend
    2. Crea la tabla de historial
    """
    
    if n_clicks == 0:
        return html.Div()
    
    # Obtener historial del backend
    history_data = api_client.get_predictions_history()
    
    # Crear y retornar tabla de historial
    return create_history_table(history_data)

# ✅ AGREGAR ESTE CALLBACK NUEVO
@callback(
    [Output('stroke-id-dropdown', 'options'),
     Output('stroke-id-dropdown', 'disabled'),
     Output('stroke-id-dropdown', 'value'),
     Output('stroke-id-info', 'children')],
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
    
    # ✅ VERIFICAR SI VIENE CON LATEST Y RESOLVERLO
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
    
    # Forma 2: Dropdown habilitado para selección libre
    options = create_stroke_id_options(stroke_data)
    info_msg = html.Div([
        html.I(className="fas fa-info-circle"),
        html.Span(f"Seleccione una de las {len(stroke_data)} predicciones disponibles")
    ], className="stroke-id-selection-info")
    
    return options, False, None, info_msg

# Callback para manejar clicks en botones de "Añadir Imagen" desde la tabla de historial  
@callback(
    Output('url', 'href', allow_duplicate=True),
    [Input('history-table', 'active_cell')],
    [State('history-table', 'data')],
    prevent_initial_call=True
)
def handle_add_image_from_history(active_cell, table_data):
    """Manejar click en botón 'Añadir Imagen' desde tabla de historial"""
    if active_cell and table_data:
        row_index = active_cell['row']
        column_id = active_cell['column_id']
        
        # Verificar si se hizo click en columna "Estado Imagen" 
        if column_id == 'Estado Imagen':
            row_data = table_data[row_index]
            
            # Verificar si es un caso sin imagen (contiene "Añadir Imagen")
            if 'Añadir Imagen' in str(row_data.get('Estado Imagen', '')):
                stroke_id = row_data.get('ID')
                if stroke_id:
                    return f"/image-prediction?stroke_id={stroke_id}&origin=history"
    
    # No redirigir si no cumple condiciones
    return dash.no_update

# Callback para mostrar notificación de éxito después de análisis
@callback(
    Output('image-results-container', 'children', allow_duplicate=True),
    [Input('analyze-image-button', 'n_clicks')],
    [State('stroke-id-dropdown', 'value'),
     State('url', 'pathname')],
    prevent_initial_call=True
)
def analyze_image_simple_test(n_clicks, stroke_id, pathname):
    """Test simple de análisis de imagen"""
    
    if pathname != '/image-prediction':
        return ""
    
    if n_clicks == 0 or not stroke_id:
        return ""
    
    # ✅ USAR EL NUEVO MÉTODO DE TEST
    try:
        # Test simple sin imagen real
        result = api_client.predict_image_simple_test(stroke_id)
        
        if 'error' in result:
            return html.Div([
                html.H3("❌ Error de Conexión"),
                html.P(f"Error: {result['error']}"),
                html.P("Verifica que el backend esté ejecutándose en puerto 8000")
            ], className="error-result")
        
        # Mostrar resultado exitoso
        return html.Div([
            html.H3("Test de Conexión Exitoso"),
            html.P(f"Backend responde correctamente"),
            html.P(f"Stroke ID: {stroke_id}"),
            html.P(f"Predicción test: {result['prediction']}"),
            html.P(f"Probabilidad test: {result['probability']:.1f}%"),
            html.P(f"Tiempo: {result['processing_time_ms']} ms"),
            html.Button("Probar con imagen real", className="btn-primary")
        ], className="test-result")
        
    except Exception as e:
        return html.Div([
            html.H3("❌ Error Inesperado"),
            html.P(f"Error: {str(e)}")
        ], className="error-result")

# ✅ CALLBACK PRINCIPAL ARREGLADO Y CON DEBUG
@callback(
    [Output('image-upload-status', 'children'),
     Output('image-upload', 'children'),
     Output('analyze-image-button', 'disabled')],
    [Input('image-upload', 'contents')],
    [State('image-upload', 'filename'),
     State('stroke-id-dropdown', 'value'),
     State('url', 'pathname')],
    prevent_initial_call=False  # ✅ CAMBIO: Permitir llamada inicial
)
def handle_image_upload(contents, filename, stroke_id, pathname):
    """Manejar upload de imagen y mostrar preview dentro del área de upload"""
    
    print(f"🔍 DEBUG handle_image_upload llamado:")
    print(f"  - pathname: {pathname}")
    print(f"  - contents: {'SÍ' if contents else 'NO'}")
    print(f"  - filename: {filename}")
    print(f"  - stroke_id: {stroke_id}")
    
    # ✅ ÁREA DE UPLOAD POR DEFECTO
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
    
    # ✅ CREAR PREVIEW DENTRO DEL ÁREA DE UPLOAD
    try:
        print("✅ Creando preview de imagen...")
        
        # ✅ CONTENIDO CON IMAGEN PARA EL ÁREA DE UPLOAD
        upload_area_with_image = html.Div([
            html.Div([
                html.Img(
                    src=contents,  # ✅ USAR TODO EL CONTENIDO
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

# ✅ CALLBACK PARA REMOVER IMAGEN ARREGLADO
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
        return "/#history"
    
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
    app.run(debug=True, host='0.0.0.0', port=FRONTEND_PORT)