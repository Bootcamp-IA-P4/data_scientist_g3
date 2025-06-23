from dash import html
from components.navbar_components import create_navbar

def get_about_layout():
    """Retorna el layout de la página About"""
    return html.Div([
        
        create_navbar(),
        
        # Contenido principal
        html.Div([
            
            html.Div([
                html.H1("Nuestro Equipo"),
                html.P("Los desarrolladores detrás del Predictor de Riesgo de Stroke", 
                       className="team-subtitle")
            ], className="team-hero"),
            
            # Container
            html.Div([
                
                html.Div([
                    html.H2("Sobre el Proyecto"),
                    html.P([
                        "Somos un equipo multidisciplinario dedicado a desarrollar soluciones ",
                        "innovadoras en el área de la salud digital. Nuestro Predictor de Riesgo ",
                        "de Stroke utiliza técnicas avanzadas de machine learning para ayudar en ",
                        "la detección temprana de factores de riesgo cardiovascular."
                    ]),
                    
                ], className="project-info"),
                
                # Equipo
                html.Div([
                    
                    html.Div([
                        html.Div([
                            html.Div("", className="avatar avatar-pepe")
                        ], className="avatar-frame"),
                        html.H3("Pepe", className="member-name"),
                        html.P("Scrum manager", className="member-role"),
                        html.P("Especialista en machine learning y arquitectura de software.", 
                               className="member-description")
                    ], className="team-member"),
                    
                    html.Div([
                        html.Div([
                            html.Div("", className="avatar avatar-maryna")
                        ], className="avatar-frame"),
                        html.H3("Maryna", className="member-name"),
                        html.P("developer", className="member-role"),
                        html.P("Desarrolladora de modelos de machine learning y redes neuronales", 
                               className="member-description")
                    ], className="team-member"),
                    
                    html.Div([
                        html.Div([
                            html.Div("", className="avatar avatar-jorge")
                        ], className="avatar-frame"),
                        html.H3("Jorge", className="member-name"),
                        html.P("Developer", className="member-role"),
                        html.P("Creador de modelos de machine learning", 
                               className="member-description")
                    ], className="team-member"),
                    
                    html.Div([
                        html.Div([
                            html.Div("", className="avatar avatar-mariela")
                        ], className="avatar-frame"),
                        html.H3("Marie", className="member-name"),
                        html.P("Developer", className="member-role"),
                        html.P("Diseñadora de experiencia de usuario. Frontend", 
                               className="member-description")
                    ], className="team-member"),
                    
                    html.Div([
                        html.Div([
                            html.Div("", className="avatar avatar-maxi")
                        ], className="avatar-frame"),
                        html.H3("Maxi", className="member-name"),
                        html.P("Data Scientist", className="member-role"),
                        html.P("Científico de datos especializado en análisis de modelos de redes neuronales.", 
                               className="member-description")
                    ], className="team-member"),
                    
                ], className="team-row")
                
            ], className="team-container")
            
        ])
    ])