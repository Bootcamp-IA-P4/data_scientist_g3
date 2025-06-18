import requests
from config.settings import API_BASE_URL
from typing import Dict, List
import base64
import io

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

    def predict_image(self, image_contents: str, stroke_prediction_id: int, 
                      filename: str) -> Dict:
        """
        Envía imagen al backend para predicción de stroke
        POST a /predict/image/{stroke_prediction_id}
        
        Args:
            image_contents: Contenido de imagen en formato data:image/...;base64,xxxxx
            stroke_prediction_id: ID de la predicción de stroke
            filename: Nombre del archivo
        """
        try:
            print(f"🔍 Iniciando predicción de imagen para stroke ID {stroke_prediction_id}")
            print(f"📁 Archivo: {filename}")
            
            # Decodificar correctamente el base64
            if ',' in image_contents:
                # Formato: "data:image/jpeg;base64,/9j/4AAQSkZJRgABAQEA..."
                header, base64_data = image_contents.split(',', 1)
                print(f"📋 Header detectado: {header}")
            else:
                # Si ya es solo base64 sin header
                base64_data = image_contents
                print("📋 Datos ya en formato base64 puro")
            
            # Decodificar base64 a bytes
            try:
                image_bytes = base64.b64decode(base64_data)
                print(f"✅ Imagen decodificada: {len(image_bytes)} bytes")
            except Exception as decode_error:
                print(f"❌ Error decodificando base64: {decode_error}")
                return {"error": f"Error decodificando imagen: {str(decode_error)}"}
            
            # Detectar content type basado en filename
            if filename.lower().endswith(('.jpg', '.jpeg')):
                content_type = 'image/jpeg'
            elif filename.lower().endswith('.png'):
                content_type = 'image/png'
            elif filename.lower().endswith('.webp'):
                content_type = 'image/webp'
            elif filename.lower().endswith('.bmp'):
                content_type = 'image/bmp'
            else:
                content_type = 'image/jpeg'  # Default
            
            print(f"🎯 Content-Type detectado: {content_type}")
            
            # Preparar archivo para FastAPI
            files = {
                'image': (filename, image_bytes, content_type)
            }
            
            print(f"🚀 Enviando imagen a: {self.base_url}/predict/image/{stroke_prediction_id}")
            
            response = requests.post(
                f"{self.base_url}/predict/image/{stroke_prediction_id}",
                files=files,
                timeout=60  # Más tiempo para procesamiento de imagen
            )
            
            print(f"📊 Respuesta del servidor - Status: {response.status_code}")
            
            if response.status_code == 200:
                result = response.json()
                print(f"✅ Predicción imagen exitosa!")
                print(f"   - Predicción: {result.get('prediction')}")
                print(f"   - Probabilidad: {result.get('probability')}")
                print(f"   - Riesgo: {result.get('risk_level')}")
                print(f"   - Tiempo: {result.get('processing_time_ms')} ms")
                return result
            else:
                # Manejar errores del backend
                error_detail = "Error desconocido"
                try:
                    error_data = response.json()
                    error_detail = error_data.get('detail', f"Status {response.status_code}")
                    print(f"❌ Error del backend: {error_detail}")
                except:
                    error_detail = f"Status {response.status_code}: {response.text[:200]}"
                    print(f"❌ Error sin JSON: {error_detail}")
                
                return {"error": f"Error del servidor: {error_detail}"}
                
        except requests.exceptions.ConnectionError as e:
            print(f"❌ Error de conexión: {e}")
            return {"error": f"Error de conexión al backend. Verifique que esté ejecutándose en puerto 8000."}
        except requests.exceptions.Timeout as e:
            print(f"❌ Timeout: {e}")
            return {"error": "Timeout procesando imagen. La imagen puede ser muy grande o el servidor está ocupado."}
        except requests.exceptions.RequestException as e:
            print(f"❌ Error de request: {e}")
            return {"error": f"Error de conexión: {str(e)}"}
        except Exception as e:
            print(f"❌ Error inesperado: {e}")
            return {"error": f"Error inesperado: {str(e)}"}
    
    def predict_image_simple_test(self, stroke_prediction_id: int) -> Dict:
        """Test simple de predicción de imagen sin archivo real"""
        try:
            # Solo hacer un test de conectividad
            response = requests.get(
                f"{self.base_url}/health",
                timeout=10
            )
            
            if response.status_code == 200:
                return {
                    "prediction": 0,
                    "probability": 0.23,
                    "risk_level": "Bajo",
                    "processing_time_ms": 1500,
                    "stroke_prediction_id": stroke_prediction_id,
                    "model_confidence": 0.87,
                    "message": "Test exitoso - Backend funcionando"
                }
            else:
                return {"error": f"Backend no responde: {response.status_code}"}
                
        except Exception as e:
            return {"error": f"Error de conexión: {str(e)}"}

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
        Obtiene historial combinado de stroke e imágenes
        Combina datos en el frontend usando endpoints existentes
        """
        try:
            print("📊 Obteniendo historial combinado...")
            
            # Obtener ambos historiales
            stroke_predictions = self.get_predictions_history()
            image_predictions = self.get_image_predictions_history()
            
            print(f"✅ Datos obtenidos - Stroke: {len(stroke_predictions)}, Imágenes: {len(image_predictions)}")
            
            # Crear mapa de imágenes por stroke_prediction_id
            images_by_stroke_id = {}
            for img in image_predictions:
                stroke_id = img.get('stroke_prediction_id')
                if stroke_id:
                    images_by_stroke_id[stroke_id] = img
            
            print(f"📷 Imágenes mapeadas por stroke_id: {len(images_by_stroke_id)} asociaciones")
            
            # Estadísticas
            total_stroke = len(stroke_predictions)
            total_images = len(image_predictions)
            completion_rate = (total_images / total_stroke * 100) if total_stroke > 0 else 0
            
            # Contar casos de alto riesgo
            high_risk_stroke = sum(1 for s in stroke_predictions if s.get('risk_level') in ['Alto', 'Crítico'])
            high_risk_images = sum(1 for i in image_predictions if i.get('risk_level') in ['Alto', 'Crítico'])
            
            return {
                'stroke_data': stroke_predictions,
                'image_data': image_predictions,
                'images_by_stroke_id': images_by_stroke_id,
                'stats': {
                    'total_stroke': total_stroke,
                    'total_images': total_images,
                    'completion_rate': completion_rate,
                    'high_risk_stroke': high_risk_stroke,
                    'high_risk_images': high_risk_images,
                    'cases_with_both': len(images_by_stroke_id)
                },
                'success': True
            }
            
        except Exception as e:
            print(f"❌ Error combinando historiales: {e}")
            return {
                'stroke_data': [],
                'image_data': [],
                'images_by_stroke_id': {},
                'stats': {
                    'total_stroke': 0,
                    'total_images': 0,
                    'completion_rate': 0,
                    'high_risk_stroke': 0,
                    'high_risk_images': 0,
                    'cases_with_both': 0
                },
                'success': False,
                'error': str(e)
            }

# Instancia global del cliente API
api_client = StrokeAPIClient()