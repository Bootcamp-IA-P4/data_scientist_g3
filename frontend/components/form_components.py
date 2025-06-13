from dash import dcc, html
from config.settings import DROPDOWN_VALUES

def create_form_layout():
    """Crea el layout completo del formulario con los 10 campos - CENTRADO Y VERTICAL"""
    
    return html.Div([
        html.H2("Datos del Paciente"),
        
        # Campo 1: Edad
        html.Div([
            html.Label("Edad (0-120 años):", className="form-label"),
            dcc.Input(
                id='edad-input',
                type='number',
                min=0, max=120,
                placeholder="Ej: 45"
            )
        ], className="form-group"),
        
        # Campo 2: Género
        html.Div([
            html.Label("Género:", className="form-label"),
            dcc.Dropdown(
                id='genero-dropdown',
                options=DROPDOWN_VALUES['genero'],
                placeholder="Seleccione género"
            )
        ], className="form-group"),
        
        # Campo 3: Glucosa
        html.Div([
            html.Label("Glucosa (50-500 mg/dL):", className="form-label"),
            dcc.Input(
                id='glucosa-input',
                type='number',
                min=50, max=500,
                placeholder="Ej: 120"
            )
        ], className="form-group"),
        
        # Campo 4: BMI (Opcional)
        html.Div([
            html.Label("BMI (10-60) - Opcional:", className="form-label"),
            dcc.Input(
                id='bmi-input',
                type='number',
                min=10, max=60,
                placeholder="Ej: 25.5"
            )
        ], className="form-group"),
        
        # Campo 5: Hipertensión
        html.Div([
            html.Label("Hipertensión:", className="form-label"),
            dcc.Dropdown(
                id='hipertension-dropdown',
                options=DROPDOWN_VALUES['hipertension'],
                placeholder="Seleccione opción"
            )
        ], className="form-group"),
        
        # Campo 6: Enfermedad cardíaca
        html.Div([
            html.Label("Enfermedad cardíaca:", className="form-label"),
            dcc.Dropdown(
                id='enfermedad-dropdown',
                options=DROPDOWN_VALUES['enfermedad'],
                placeholder="Seleccione opción"
            )
        ], className="form-group"),
        
        # Campo 7: Casado alguna vez
        html.Div([
            html.Label("Casado alguna vez:", className="form-label"),
            dcc.Dropdown(
                id='casado-dropdown',
                options=DROPDOWN_VALUES['casado'],
                placeholder="Seleccione opción"
            )
        ], className="form-group"),
        
        # Campo 8: Trabajo
        html.Div([
            html.Label("Trabajo:", className="form-label"),
            dcc.Dropdown(
                id='trabajo-dropdown',
                options=DROPDOWN_VALUES['trabajo'],
                placeholder="Seleccione tipo de trabajo"
            )
        ], className="form-group"),
        
        # Campo 9: Residencia
        html.Div([
            html.Label("Residencia:", className="form-label"),
            dcc.Dropdown(
                id='residencia-dropdown',
                options=DROPDOWN_VALUES['residencia'],
                placeholder="Seleccione tipo de residencia"
            )
        ], className="form-group"),
        
        # Campo 10: Estado de fumador
        html.Div([
            html.Label("Estado de fumador:", className="form-label"),
            dcc.Dropdown(
                id='fumador-dropdown',
                options=DROPDOWN_VALUES['fumador'],
                placeholder="Seleccione estado de fumador"
            )
        ], className="form-group"),
        
        # Botones
        html.Div([
            html.Button('Predecir Riesgo', 
                       id='predict-button', 
                       n_clicks=0, 
                       className='btn-primary'),
            html.Button('Ver Historial', 
                       id='history-button', 
                       n_clicks=0, 
                       className='btn-primary'),
        ])
        
    ], className="form-container")

def validate_form_data(edad, genero, glucosa, bmi, hipertension, enfermedad, 
                      casado, trabajo, residencia, fumador):
    """Valida que todos los campos obligatorios estén llenos"""
    
    # Campos obligatorios (todos excepto BMI)
    required_fields = [edad, genero, glucosa, hipertension, enfermedad, 
                      casado, trabajo, residencia, fumador]
    
    # Verificar si algún campo obligatorio está vacío
    if any(field is None or field == '' for field in required_fields):
        return False, "❌ Error: Todos los campos son obligatorios excepto BMI"
    
    return True, ""

def prepare_form_data(edad, genero, glucosa, bmi, hipertension, enfermedad, 
                     casado, trabajo, residencia, fumador):
    """Prepara los datos del formulario para enviar al backend"""
    
    return {
        'age': edad,
        'gender': genero,
        'avg_glucose_level': glucosa,
        'bmi': bmi,  # Puede ser None
        'hypertension': hipertension,
        'heart_disease': enfermedad,
        'ever_married': casado,
        'work_type': trabajo,
        'residence_type': residencia,
        'smoking_status': fumador
    }