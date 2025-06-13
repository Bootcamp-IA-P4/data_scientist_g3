# app.py - Aplicación Principal
import dash
from dash import dcc, html, Input, Output, State, callback
from config.settings import FRONTEND_PORT

# Importar componentes y servicios
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

# Inicializar la aplicación Dash
app = dash.Dash(__name__)

# Configurar el título de la página
app.title = "Predictor de Riesgo de Stroke"

# Layout principal de la aplicación
app.layout = html.Div([
    # Título principal
    html.H1("🧠 Predictor de Riesgo de Stroke"),
    
    # Formulario de entrada
    create_form_layout(),
    
    # Área de resultados
    html.Div(id='results-container'),
    
    # Disclaimer médico
    create_disclaimer(),
    
    # Área de historial
    html.Div(id='history-container'),
    
    # Store para guardar datos temporalmente
    dcc.Store(id='prediction-store')
])

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
    
    # Si no se ha clickeado el botón, no hacer nada
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
    
    # Verificar si hubo error en la respuesta
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
    
    # Si no se ha clickeado el botón, no mostrar nada
    if n_clicks == 0:
        return html.Div()
    
    # Obtener historial del backend
    history_data = api_client.get_predictions_history()
    
    # Crear y retornar tabla de historial
    return create_history_table(history_data)

# Ejecutar la aplicación
if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=FRONTEND_PORT)