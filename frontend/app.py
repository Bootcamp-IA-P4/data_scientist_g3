import dash
from dash import dcc, html, Input, Output, State, callback
from config.settings import FRONTEND_PORT
from pages.about import get_about_layout
from services.api_client import api_client
from components.form_components import (
    create_form_layout, 
    validate_form_data, 
    prepare_form_data
)
from components.results_components import (
    create_result_card, 
    create_error_message, 
    create_disclaimer
)
from components.history_components import create_history_table
from components.navbar_components import create_navbar

# Inicializar la aplicación Dash
app = dash.Dash(
    __name__,
    suppress_callback_exceptions=True,  # Oculta errores de callbacks
    show_undo_redo=False,              # Oculta botones undo/redo
    external_stylesheets=['assets/style.css', 'assets/about.css']            # Sin estilos externos que puedan mostrar debugging            
)

# Título de la página
app.title = "Predictor de Riesgo de Stroke"

# Layout principal
def get_home_layout():
    return html.Div([
        
        # Video de fondo
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
        
        # Navbar futurista
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
    result_card = create_result_card(prediction, probability, risk_level)
    
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

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=FRONTEND_PORT)