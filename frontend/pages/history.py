from dash import html, dcc
from components.navbar_components import create_navbar

def get_history_layout():
    """Layout de la página de historial combinado de predicciones - SIN FILTROS"""
    return html.Div([
        # Navbar
        create_navbar(),
        
        # Contenido principal
        html.Div([
            # Título principal
            html.H1([
                "Historial de ",
                html.Span("Predicciones", className="title-accent")
            ], className="page-title"),
            
            # Subtítulo explicativo
            html.P([
                "Registro completo de predicciones de stroke y análisis de tomografías computarizadas."
            ], className="page-subtitle"),
            
            # Estadísticas generales
            html.Div(id='history-stats-container'),
            
            # Tabla de historial combinado
            html.Div(id='combined-history-container'),
            
            # Stores para manejar datos
            dcc.Store(id='stroke-history-store'),
            dcc.Store(id='image-history-store'),
            dcc.Store(id='combined-history-store')
            
        ], className="main-content")
    ], className="history-page")