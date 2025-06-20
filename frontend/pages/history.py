from dash import html, dcc
from components.navbar_components import create_navbar

def get_history_layout():
    """Layout de la página de historial combinado de predicciones"""
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
            
            # Filtros y controles
            html.Div([
                html.Div([
                    html.Label("Filtrar por riesgo:", className="filter-label"),
                    dcc.Dropdown(
                        id='risk-filter-dropdown',
                        options=[
                            {'label': 'Todos los niveles', 'value': 'all'},
                            {'label': 'Bajo', 'value': 'Bajo'},
                            {'label': 'Medio', 'value': 'Medio'},
                            {'label': 'Alto', 'value': 'Alto'},
                            {'label': 'Crítico', 'value': 'Crítico'}
                        ],
                        value='all',
                        className="filter-dropdown"
                    )
                ], className="filter-group"),
                
                html.Div([
                    html.Label("Estado de imagen:", className="filter-label"),
                    dcc.Dropdown(
                        id='image-status-filter-dropdown',
                        options=[
                            {'label': 'Todos', 'value': 'all'},
                            {'label': 'Con imagen', 'value': 'with_image'},
                            {'label': 'Sin imagen', 'value': 'without_image'}
                        ],
                        value='all',
                        className="filter-dropdown"
                    )
                ], className="filter-group"),
                
                html.Div([
                    html.Button(
                        "Actualizar",
                        id='refresh-history-button',
                        className="btn-secondary",
                        n_clicks=0
                    ),
                    html.Button(
                        "Exportar CSV",
                        id='export-csv-button',
                        className="btn-secondary",
                        n_clicks=0
                    )
                ], className="filter-actions")
            ], className="history-filters"),
            
            # Tabla de historial combinado
            html.Div(id='combined-history-container'),
            
            # Stores para manejar datos
            dcc.Store(id='stroke-history-store'),
            dcc.Store(id='image-history-store'),
            dcc.Store(id='combined-history-store')
            
        ], className="main-content")
    ], className="history-page")