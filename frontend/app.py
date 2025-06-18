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
    prevent_initial_call=True
)
def show_analysis_notification(n_clicks):
    """Mostrar notificación mientras se procesa la imagen"""
    if n_clicks > 0:
        return create_processing_animation()
    return ""

# Callback para validar formulario de imagen en tiempo real
@callback(
    [Output('analyze-image-button', 'style'),
     Output('image-upload-area', 'className')],
    [Input('stroke-id-dropdown', 'value'),
     Input('image-upload', 'contents')],
    prevent_initial_call=True
)
def validate_image_form_realtime(stroke_id, image_contents):
    """Validar formulario de imagen en tiempo real"""
    
    # Clases base
    upload_area_class = "image-upload-area"
    button_style = {}
    
    # Validar si ambos campos están completos
    if stroke_id and image_contents:
        upload_area_class += " valid"
        button_style = {
            'background': 'linear-gradient(135deg, var(--color-accent) 0%, var(--color-primary) 100%)',
            'transform': 'scale(1.02)',
            'box-shadow': '0 0 25px rgba(139, 92, 246, 0.4)'
        }
    elif image_contents and not stroke_id:
        upload_area_class += " invalid"
    
    return button_style, upload_area_class

# Callback para cargar datos del stroke seleccionado
@callback(
    Output('stroke-id-info', 'children', allow_duplicate=True),
    [Input('stroke-id-dropdown', 'value')],
    [State('stroke-predictions-store', 'data')],
    prevent_initial_call=True
)
def show_selected_stroke_info(selected_id, stroke_predictions):
    """Mostrar información del stroke seleccionado"""
    
    if not selected_id or not stroke_predictions:
        return ""
    
    # Buscar la predicción seleccionada
    selected_pred = next((p for p in stroke_predictions if p.get('id') == selected_id), None)
    
    if not selected_pred:
        return ""
    
    # Crear info card
    return html.Div([
        html.H5(f"📋 Predicción #{selected_id}"),
        html.Div([
            html.Span(f"Riesgo: {selected_pred.get('risk_level', 'N/A')}", className="info-item"),
            html.Span(f"Probabilidad: {selected_pred.get('probability', 0):.1f}%", className="info-item"),
            html.Span(f"Fecha: {selected_pred.get('created_at', 'N/A')[:10] if selected_pred.get('created_at') else 'N/A'}", className="info-item")
        ], className="stroke-info-details")
    ], className="selected-stroke-info")

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