from dash import html, dcc
from components.navbar_components import create_navbar
from components.image_components import (
    create_image_upload_form,
    create_upload_restrictions_info,
    create_image_result_card
)

def get_image_prediction_layout():
    """Layout de la página de predicción de imágenes"""
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
        
        # Navbar
        create_navbar(),
        
        # Contenido principal
        html.Div([
            # Título principal
            html.H1([
                "Predicción de Stroke por ",
                html.Span("Tomografía", className="title-accent")
            ], className="page-title"),
            
            # Subtítulo explicativo
            html.P([
                "Analiza imágenes de tomografía computarizada del cerebro para detectar ",
                "indicadores de stroke usando redes neuronales convolucionales."
            ], className="page-subtitle"),
            
            # Información de restricciones
            create_upload_restrictions_info(),
            
            # Formulario de upload
            create_image_upload_form(),
            
            # Contenedor de resultados
            html.Div(id='image-results-container'),
                       
            # Stores para manejar estado
            dcc.Store(id='image-prediction-store'),
            dcc.Store(id='upload-restrictions-store'),
            dcc.Store(id='stroke-predictions-store')  # Para cargar IDs disponibles
            
        ], className="main-content")
    ], className="image-prediction-page")