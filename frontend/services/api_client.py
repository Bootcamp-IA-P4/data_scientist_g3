import requests
from config.settings import API_BASE_URL
from typing import Dict, List

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
        Obtiene historial de predicciones de stroke
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

    # nuevos metodos para imagenes
    def get_image_upload_info(self) -> Dict:
        """
        Obtiene información sobre restricciones de upload de imagen
        GET a /image/upload-info
        """
        try:
            response = requests.get(
                f"{self.base_url}/image/upload-info",
                timeout=10
            )
            
            if response.status_code == 200:
                return response.json()
            else:
                print(f"Error obteniendo info de upload: {response.status_code}")
                return {
                    "max_size_mb": 10,
                    "allowed_formats": ["JPEG", "PNG", "WEBP", "BMP"],
                    "min_dimensions": {"width": 32, "height": 32},
                    "max_dimensions": {"width": 4096, "height": 4096}
                }
                
        except requests.exceptions.RequestException as e:
            print(f"Error de conexión obteniendo upload info: {e}")
            return {
                "max_size_mb": 10,
                "allowed_formats": ["JPEG", "PNG", "WEBP", "BMP"],
                "min_dimensions": {"width": 32, "height": 32},
                "max_dimensions": {"width": 4096, "height": 4096}
            }

    def predict_image(self, image_data: bytes, stroke_prediction_id: int, 
                      filename: str) -> Dict:
        """
        Envía imagen al backend para predicción de stroke
        POST a /predict/image/{stroke_prediction_id}
        """
        try:
            # Preparar archivo para envío
            files = {
                'image': (filename, image_data, 'image/jpeg')
            }
            
            print(f"Enviando imagen {filename} para stroke ID {stroke_prediction_id}")
            
            response = requests.post(
                f"{self.base_url}/predict/image/{stroke_prediction_id}",
                files=files,
                timeout=60  # Más tiempo para procesamiento de imagen
            )
            
            print(f"Respuesta predicción imagen - Status: {response.status_code}")
            
            if response.status_code == 200:
                result = response.json()
                print(f"Predicción imagen exitosa: {result}")
                return result
            else:
                error_detail = "Error desconocido"
                try:
                    error_data = response.json()
                    error_detail = error_data.get('detail', f"Status {response.status_code}")
                except:
                    error_detail = f"Status {response.status_code}: {response.text}"
                
                print(f"Error en predicción imagen: {error_detail}")
                return {"error": f"Error del servidor: {error_detail}"}
                
        except requests.exceptions.ConnectionError as e:
            print(f"Error de conexión en predicción imagen: {e}")
            return {"error": f"Error de conexión: {str(e)}"}
        except requests.exceptions.Timeout as e:
            print(f"Timeout en predicción imagen: {e}")
            return {"error": "Timeout procesando imagen. Intente con una imagen más pequeña."}
        except requests.exceptions.RequestException as e:
            print(f"Error de request en predicción imagen: {e}")
            return {"error": f"Error de conexión: {str(e)}"}
        except Exception as e:
            print(f"Error inesperado en predicción imagen: {e}")
            return {"error": f"Error inesperado: {str(e)}"}

    def get_image_predictions_history(self, limit: int = 50) -> List[Dict]:
        """
        Obtiene historial de predicciones de imagen
        GET a /predictions/images
        """
        try:
            print(f"Solicitando historial de imágenes desde: {self.base_url}/predictions/images")
            
            response = requests.get(
                f"{self.base_url}/predictions/images",
                params={"limit": limit},
                timeout=30
            )
            
            print(f"Respuesta historial imágenes - Status: {response.status_code}")
            
            if response.status_code == 200:
                data = response.json()
                print(f"Datos recibidos historial imágenes: {len(data.get('predictions', []))} elementos")
                
                # El backend devuelve ImagePredictionsList
                if isinstance(data, dict) and 'predictions' in data:
                    return data['predictions']
                elif isinstance(data, list):
                    return data
                else:
                    print(f"Estructura inesperada historial imágenes: {type(data)}")
                    return []
                    
            else:
                print(f"Error obteniendo historial imágenes: {response.status_code}")
                try:
                    error_detail = response.json()
                    print(f"Detalle error historial imágenes: {error_detail}")
                except:
                    print(f"Respuesta historial imágenes: {response.text}")
                return []
                
        except requests.exceptions.RequestException as e:
            print(f"Error de conexión historial imágenes: {e}")
            return []
        except Exception as e:
            print(f"Error inesperado historial imágenes: {e}")
            return []

    def get_combined_predictions_history(self) -> Dict:
        """
        Obtiene historial combinado de stroke e imágenes para la tabla
        """
        try:
            # Obtener ambos historiales
            stroke_predictions = self.get_predictions_history()
            image_predictions = self.get_image_predictions_history()
            
            # Crear mapa de imágenes por stroke_prediction_id
            images_by_stroke_id = {}
            for img in image_predictions:
                stroke_id = img.get('stroke_prediction_id')
                if stroke_id:
                    images_by_stroke_id[stroke_id] = img
            
            # Combinar datos
            combined_data = []
            for stroke in stroke_predictions:
                stroke_id = stroke.get('id')
                image_data = images_by_stroke_id.get(stroke_id, None)
                
                combined_record = {
                    'id': stroke_id,
                    'fecha': stroke.get('created_at', 'N/A'),
                    'paciente': stroke.get('patient_name', f'Paciente #{stroke_id}'),
                    'hipertension': stroke.get('hypertension', 'N/A'),
                    'glucosa': stroke.get('avg_glucose_level', 'N/A'),
                    'stroke_probability': stroke.get('probability', 0),
                    'stroke_risk_level': stroke.get('risk_level', 'N/A'),
                    'estado_clinico': '✅ Completado',
                    'image_probability': image_data.get('probability', None) if image_data else None,
                    'image_risk_level': image_data.get('risk_level', None) if image_data else None,
                    'estado_imagen': '✅ Completado' if image_data else '[📸 Añadir Imagen]',
                    'has_image': bool(image_data)
                }
                combined_data.append(combined_record)
            
            return {
                'combined_data': combined_data,
                'total_stroke': len(stroke_predictions),
                'total_images': len(image_predictions),
                'completion_rate': len(image_predictions) / len(stroke_predictions) * 100 if stroke_predictions else 0
            }
            
        except Exception as e:
            print(f"Error combinando historiales: {e}")
            return {
                'combined_data': [],
                'total_stroke': 0,
                'total_images': 0,
                'completion_rate': 0
            }

# Instancia global del cliente API
api_client = StrokeAPIClient()