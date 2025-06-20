"""
Servicio de procesamiento de imágenes para predicción de stroke - CARGA LAZY
Evita segmentation fault en macOS cargando PyTorch solo cuando sea necesario
"""

import time
import uuid
import sys
from typing import Dict, List, Optional
from pathlib import Path

# IMPORTACIONES SEGURAS AL TOP - Sin PyTorch
try:
    # Agregar rutas necesarias al inicio
    current_dir = Path(__file__).resolve().parent
    backend_app_dir = current_dir.parent
    project_root = current_dir.parent.parent.parent
    
    if str(backend_app_dir) not in sys.path:
        sys.path.insert(0, str(backend_app_dir))
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    
    # Importaciones fijas de base de datos (seguras)
    from database.supabase_client import save_image_prediction, get_stroke_predictions, get_image_predictions
    
    print("✅ Importaciones de base de datos cargadas correctamente")
    
except ImportError as e:
    print(f"❌ Error importando dependencias de DB: {e}")
    # Definir funciones dummy para base de datos
    async def save_image_prediction(*args, **kwargs):
        return False
    
    async def get_stroke_predictions(*args, **kwargs):
        return []
    
    async def get_image_predictions(*args, **kwargs):
        return []


class LazyImagePipeline:
    """Pipeline CNN con carga lazy para evitar segmentation fault"""
    
    def __init__(self):
        self._pipeline_loaded = False
        self._pipeline_error = None
        self._predict_func = None
        self._validate_func = None
        self._status_func = None
    
    def _load_pipeline_lazy(self):
        """Cargar pipeline solo cuando sea necesario"""
        if self._pipeline_loaded or self._pipeline_error:
            return
        
        try:
            print("🔄 Cargando pipeline CNN de forma lazy...")
            
            # Importar SOLO cuando sea necesario
            from src.pipeline.image_pipeline import predict_stroke_from_image, validate_image, get_pipeline_status
            
            # Test rápido para verificar que funciona
            status = get_pipeline_status()
            if not status.get('is_loaded', False):
                raise Exception("Pipeline no se cargó correctamente")
            
            # Asignar funciones
            self._predict_func = predict_stroke_from_image
            self._validate_func = validate_image
            self._status_func = get_pipeline_status
            
            self._pipeline_loaded = True
            print(f"✅ Pipeline CNN cargado lazy - Dispositivo: {status.get('device', 'unknown')}")
            
        except Exception as e:
            self._pipeline_error = str(e)
            print(f"❌ Error cargando pipeline lazy: {e}")
            
            # Funciones dummy en caso de error
            self._predict_func = lambda x: {'error': f'Pipeline no disponible: {e}'}
            self._validate_func = lambda x: {'valid': False, 'errors': [f'Pipeline no disponible: {e}']}
            self._status_func = lambda: {'is_loaded': False, 'error': str(e)}
    
    def predict(self, image_data: bytes) -> Dict:
        """Predicción con carga lazy"""
        self._load_pipeline_lazy()
        return self._predict_func(image_data)
    
    def validate(self, image_data: bytes) -> Dict:
        """Validación con carga lazy"""
        self._load_pipeline_lazy()
        return self._validate_func(image_data)
    
    def get_status(self) -> Dict:
        """Estado con carga lazy"""
        self._load_pipeline_lazy()
        return self._status_func()
    
    # @property
    # def is_available(self) -> bool:
    #     """Check si el pipeline está disponible sin cargarlo"""
    #     if self._pipeline_loaded:
    #         return True
    #     if self._pipeline_error:
    #         return False
        
    #     # No cargar, solo verificar si existe el archivo
    #     try:
    #         current_dir = Path(__file__).resolve().parent
    #         project_root = current_dir.parent.parent.parent
    #         model_path = project_root / "models" / "CNN_PyTorch" / "modelo_cnn_stroke_pytorch.zip"
    #         return model_path.exists()
    #     except:
    #         return False
    
    @property
    def is_available(self) -> bool:
        """Check si el pipeline está disponible sin cargarlo"""
        if self._pipeline_loaded:
            return True
        if self._pipeline_error:
            return False

        try:
            model_path = Path("/backend/models/CNN_PyTorch/modelo_cnn_stroke_pytorch.zip")
            return model_path.exists()
        except Exception:
            return False


class ImageService:
    """Servicio para predicciones de imagen - Con carga lazy"""
    
    def __init__(self):
        self.pipeline = LazyImagePipeline()
        self.service_ready = True  # Siempre listo, carga bajo demanda
    
    async def process_image(self, image_data: bytes, stroke_prediction_id: int, 
                        filename: str) -> Dict:
        """Procesar imagen para predicción - Con carga lazy"""
        
        try:
            # 1. Verificar disponibilidad sin cargar
            if not self.pipeline.is_available:
                return {'error': 'Pipeline de imagen no disponible'}
            
            # 2. Validar imagen (esto carga el pipeline si es necesario)
            print(f"🔍 Validando imagen: {filename}")
            validation = self.pipeline.validate(image_data)
            if not validation['valid']:
                return {'error': 'Imagen no válida', 'details': validation['errors']}
            
            # 3. Verificar que stroke_prediction existe
            print(f"🔍 Verificando stroke prediction ID: {stroke_prediction_id}")
            stroke_predictions = await get_stroke_predictions(limit=1000)
            if not any(p.get('id') == stroke_prediction_id for p in stroke_predictions):
                return {'error': f'Predicción de stroke {stroke_prediction_id} no encontrada'}
            
            # 4. Verificar que no hay imagen duplicada
            print(f"🔍 Verificando imagen duplicada para stroke ID: {stroke_prediction_id}")
            existing_images = await get_image_predictions(stroke_prediction_id=stroke_prediction_id)
            if existing_images:
                return {'error': f'Ya existe imagen para predicción {stroke_prediction_id}'}
            
            # 5. Procesar con CNN (carga lazy automática)
            print(f"🧠 Procesando imagen con CNN...")
            cnn_result = self.pipeline.predict(image_data)
            
            if 'error' in cnn_result:
                return cnn_result
            
            # 6. Preparar datos para base de datos
            image_data_db = {
                'stroke_prediction_id': stroke_prediction_id,
                'image_filename': filename,
                'image_url': f"temp://processed_{uuid.uuid4().hex}",
                'image_size': len(image_data),
                'image_format': validation['metadata'].get('format', 'Unknown'),
                'prediction': cnn_result['prediction'],
                'probability': cnn_result['probability'],
                'risk_level': cnn_result['risk_level'],
                'model_confidence': cnn_result['model_confidence'],
                'processing_time_ms': cnn_result['processing_time_ms']
            }
            
            # 7. Guardar en base de datos
            print(f"💾 Guardando resultado en base de datos...")
            success = await save_image_prediction(image_data_db)
            if not success:
                return {'error': 'Error guardando en base de datos'}
            
            # 8. Respuesta exitosa
            response = {
                'prediction': cnn_result['prediction'],
                'probability': cnn_result['probability'],
                'risk_level': cnn_result['risk_level'],
                'model_confidence': cnn_result['model_confidence'],
                'processing_time_ms': cnn_result['processing_time_ms'],
                'stroke_prediction_id': stroke_prediction_id,
                'message': 'Imagen procesada correctamente'
            }
            
            print(f"✅ Imagen procesada exitosamente para stroke ID {stroke_prediction_id}")
            return response
            
        except Exception as e:
            error_msg = f'Error procesando imagen: {str(e)}'
            print(f"❌ {error_msg}")
            return {'error': error_msg}
    
    async def get_images(self, stroke_prediction_id: Optional[int] = None, limit: int = 50) -> List[Dict]:
        """Obtener imágenes"""
        try:
            return await get_image_predictions(stroke_prediction_id=stroke_prediction_id, limit=limit)
        except Exception as e:
            print(f"❌ Error obteniendo imágenes: {e}")
            return []
    
    async def get_status(self) -> Dict:
        """Estado del servicio"""
        try:
            # Status básico sin cargar pipeline
            basic_status = {
                'service_ready': self.service_ready,
                'pipeline_available': self.pipeline.is_available,
                'pipeline_loaded': self.pipeline._pipeline_loaded,
                'pipeline_error': self.pipeline._pipeline_error
            }
            
            # Si el pipeline ya está cargado, obtener status completo
            if self.pipeline._pipeline_loaded:
                pipeline_status = self.pipeline.get_status()
                basic_status.update({
                    'pipeline_status': pipeline_status
                })
            
            return basic_status
            
        except Exception as e:
            return {
                'service_ready': False,
                'pipeline_available': False,
                'error': str(e)
            }


# SINGLETON SIMPLE
_service_instance = None

def get_service() -> ImageService:
    """Obtener instancia del servicio"""
    global _service_instance
    if _service_instance is None:
        _service_instance = ImageService()
    return _service_instance


# FUNCIONES PRINCIPALES PARA ENDPOINTS
async def process_image_prediction(image_data: bytes, stroke_prediction_id: int, 
                                filename: str, content_type: str = None) -> Dict:
    """Función principal para procesar imagen"""
    return await get_service().process_image(image_data, stroke_prediction_id, filename)

async def get_images_for_stroke(stroke_prediction_id: int) -> List[Dict]:
    """Obtener imágenes para stroke prediction"""
    return await get_service().get_images(stroke_prediction_id=stroke_prediction_id)

async def get_all_image_predictions(limit: int = 50) -> List[Dict]:
    """Obtener todas las imágenes"""
    return await get_service().get_images(limit=limit)

async def get_image_pipeline_status() -> Dict:
    """Estado del pipeline"""
    return await get_service().get_status()


# TEST
if __name__ == "__main__":
    import asyncio
    
    async def test():
        print("🧪 Test Image Service - CARGA LAZY")
        print("=" * 50)
        
        try:
            # Test sin cargar pipeline
            status = await get_image_pipeline_status()
            print(f"Status inicial: {status}")
            
            # Test disponibilidad
            service = get_service()
            available = service.pipeline.is_available
            print(f"Pipeline disponible: {'✅' if available else '❌'}")
            
            if available:
                print("✅ Image Service listo (carga lazy)")
                print("✅ No hay segmentation fault en startup")
                print("✅ Pipeline se cargará cuando sea necesario")
            else:
                print("⚠️ Pipeline no disponible pero servicio funcional")
                
        except Exception as e:
            print(f"❌ Error en test: {e}")
    
    asyncio.run(test())