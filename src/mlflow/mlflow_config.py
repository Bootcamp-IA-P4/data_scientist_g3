import mlflow
import os
from pathlib import Path

def setup_mlflow():
    """Configuración inicial de MLflow"""
    # Obtener la ruta base del proyecto
    project_root = Path(__file__).parent.parent.parent
    
    # Configurar el directorio para almacenar los experimentos
    # Primero intentar usar la ruta definida en variable de entorno
    mlruns_path = os.getenv('MLFLOW_TRACKING_URI')
    
    if not mlruns_path:
        # Si no hay variable de entorno, usar ruta relativa al proyecto
        default_path = project_root / 'notebooks' / 'modeling' / 'mlruns'
        mlruns_path = f"file:///{str(default_path.absolute())}"
    
    mlflow.set_tracking_uri(mlruns_path)
    
    # Asegurar que el directorio existe
    os.makedirs(mlruns_path.replace('file:///', ''), exist_ok=True)
    
    # Crear un experimento si no existe
    experiment_name = "stroke_prediction"
    try:
        experiment_id = mlflow.create_experiment(experiment_name)
    except:
        experiment_id = mlflow.get_experiment_by_name(experiment_name).experiment_id
    
    return experiment_id

def log_model_metrics(model, metrics, params, model_name):
    """Función para registrar métricas y parámetros del modelo"""
    with mlflow.start_run():
        # Registrar parámetros
        for param_name, param_value in params.items():
            mlflow.log_param(param_name, param_value)
        
        # Registrar métricas
        for metric_name, metric_value in metrics.items():
            mlflow.log_metric(metric_name, metric_value)
        
        # Registrar el modelo
        mlflow.sklearn.log_model(model, model_name)