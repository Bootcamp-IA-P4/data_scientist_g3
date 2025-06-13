# services/api_client.py
import requests
from config.settings import API_BASE_URL

class StrokeAPIClient:
    def __init__(self):
        self.base_url = API_BASE_URL
    
    def predict_stroke(self, form_data):
        """
        Envía datos al backend para predicción de stroke
        POST a /predict/stroke
        """
        try:
            response = requests.post(
                f"{self.base_url}/predict/stroke", 
                json=form_data, 
                timeout=30
            )
            if response.status_code == 200:
                return response.json()
            else:
                return {"error": f"Error del servidor: {response.status_code}"}
        except requests.exceptions.RequestException as e:
            return {"error": f"Error de conexión: {str(e)}"}
    
    def get_predictions_history(self):
        """
        Obtiene historial de predicciones
        GET a /predictions/stroke
        """
        try:
            response = requests.get(
                f"{self.base_url}/predictions/stroke", 
                timeout=30
            )
            if response.status_code == 200:
                return response.json()
            else:
                print(f"Error obteniendo historial: {response.status_code}")
                return []
        except requests.exceptions.RequestException as e:
            print(f"Error de conexión al obtener historial: {e}")
            return []

# Instancia global del cliente API
api_client = StrokeAPIClient()