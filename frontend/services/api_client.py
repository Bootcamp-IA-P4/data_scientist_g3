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
            print(f"Solicitando historial desde: {self.base_url}/predictions/stroke")
            response = requests.get(
                f"{self.base_url}/predictions/stroke", 
                timeout=30
            )
            
            print(f"Respuesta del servidor - Status: {response.status_code}")
            
            if response.status_code == 200:
                data = response.json()
                print(f"Datos recibidos del historial: {data}")
                print(f"Tipo de datos: {type(data)}")
                
                # Tu backend devuelve StrokePredictionsList con estructura específica
                if isinstance(data, dict) and 'predictions' in data:
                    predictions_list = data['predictions']
                    print(f"Extraída lista de predictions: {len(predictions_list)} elementos")
                    return predictions_list
                
                # Fallback: si es directamente una lista
                elif isinstance(data, list):
                    print("Datos recibidos directamente como lista")
                    return data
                
                # Si no coincide con el formato esperado
                else:
                    print(f"Estructura inesperada del backend: {data.keys() if isinstance(data, dict) else type(data)}")
                    return []
                    
            else:
                print(f"Error del servidor al obtener historial: {response.status_code}")
                try:
                    error_detail = response.json()
                    print(f"Detalle del error: {error_detail}")
                except:
                    print(f"Contenido de la respuesta: {response.text}")
                return []
                
        except requests.exceptions.ConnectionError as e:
            print(f"Error de conexión al obtener historial: {e}")
            return []
        except requests.exceptions.Timeout as e:
            print(f"Timeout al obtener historial: {e}")
            return []
        except requests.exceptions.RequestException as e:
            print(f"Error de request al obtener historial: {e}")
            return []
        except Exception as e:
            print(f"Error inesperado al obtener historial: {e}")
            return []

# Instancia global del cliente API
api_client = StrokeAPIClient()