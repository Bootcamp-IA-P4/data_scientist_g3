from dash import html, dcc

def create_navbar():
    
    return html.Nav([
        # Logo/Brand
        html.Div([
            html.H2("🧠 NeuroWise", className="navbar-brand"),
            html.Span("AI Prediction Platform", className="navbar-subtitle")
        ], className="navbar-brand-container"),
       
        # Navigation Links
        html.Div([
            # Stroke Prediction
            html.A([
                html.I(className="nav-icon"),
                "Predicción Stroke"
            ],
            href="/",
            className="nav-link active",
            id="stroke-nav-link"),

            # Image Prediction
            html.A([
                html.I(className="nav-icon"),
                "Predicción Imágenes"
            ],
            href="/image-prediction",
            className="nav-link",
            id="image-nav-link"),
           
            # About
             html.A([
                html.I(className="nav-icon"),
                "Acerca de"
            ],
            href="/about",
            className="nav-link",
            id="about-nav-link"),
        ], className="navbar-nav"),
       
    ], className="navbar-container")