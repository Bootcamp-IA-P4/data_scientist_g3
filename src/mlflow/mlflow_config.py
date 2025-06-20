import mlflow
import os

def setup_mlflow():
    """Configuración inicial de MLflow"""
    # Configurar el directorio para almacenar los experimentos
    ruta_1 = "file:///C:/Users/jlmateos.ext/OneDrive/IA/Scripts/Repos/data_scientist_g3/notebooks/modeling/mlruns"
    ruta_2 = "file:///C:/Users/Usuario/OneDrive/IA/Scripts/Repos/data_scientist_g3/notebooks/modeling/mlruns"
    
    try:
        mlflow.set_tracking_uri(ruta_1)
        # Intentar crear un directorio de prueba para verificar permisos
        os.makedirs(ruta_1.replace('file:///', ''), exist_ok=True)
    except PermissionError:
        mlflow.set_tracking_uri(ruta_2)
    
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