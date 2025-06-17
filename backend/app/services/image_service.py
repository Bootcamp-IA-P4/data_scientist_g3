"""
Servicio de procesamiento de imágenes para predicción de stroke
"""

import time
import uuid
from typing import Dict, List, Optional
from pathlib import Path

class ImageService:
    """Servicio para predicciones de imagen"""
    
    def __init__(self):
        self.pipeline_loaded = False
        self._load_pipeline()
    
    def _load_pipeline(self):
        """Cargar pipeline CNN"""
        try:
            import sys
            current_dir = Path(__file__).resolve().parent
            project_root = current_dir.parent.parent.parent
            if str(project_root) not in sys.path:
                sys.path.insert(0, str(project_root))
            
            from src.pipeline.image_pipeline import predict_stroke_from_image, validate_image, get_pipeline_status
            
            status = get_pipeline_status()
            if not status['is_loaded']:
                raise Exception("Pipeline CNN no cargado")
            
            self.predict = predict_stroke_from_image
            self.validate = validate_image
            self.pipeline_loaded = True
            print(f"✅ Pipeline CNN cargado - {status['device']}")
            
        except Exception as e:
            print(f"❌ Error cargando pipeline: {e}")
            self.pipeline_loaded = False
    
    async def process_image(self, image_data: bytes, stroke_prediction_id: int, 
                          filename: str) -> Dict:
        """Procesar imagen para predicción"""
        if not self.pipeline_loaded:
            return {'error': 'Pipeline CNN no disponible'}
        
        try:
            # Import dinámico de DB
            from database.supabase_client import save_image_prediction, get_stroke_predictions, get_image_predictions
            
            # 1. Validar imagen
            validation = self.validate(image_data)
            if not validation['valid']:
                return {'error': 'Imagen no válida', 'details': validation['errors']}
            
            # 2. Verificar stroke_prediction existe
            stroke_predictions = await get_stroke_predictions(limit=1000)
            if not any(p.get('id') == stroke_prediction_id for p in stroke_predictions):
                return {'error': f'Predicción {stroke_prediction_id} no encontrada'}
            
            # 3. Verificar no hay imagen duplicada
            existing_images = await get_image_predictions(stroke_prediction_id=stroke_prediction_id)
            if existing_images:
                return {'error': f'Ya existe imagen para predicción {stroke_prediction_id}'}
            
            # 4. Procesar con CNN
            cnn_result = self.predict(image_data)
            
            # 5. Guardar en DB
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
            
            success = await save_image_prediction(image_data_db)
            if not success:
                return {'error': 'Error guardando en base de datos'}
            
            # 6. Respuesta
            return {
                'prediction': cnn_result['prediction'],
                'probability': cnn_result['probability'],
                'risk_level': cnn_result['risk_level'],
                'model_confidence': cnn_result['model_confidence'],
                'processing_time_ms': cnn_result['processing_time_ms'],
                'stroke_prediction_id': stroke_prediction_id,
                'message': 'Imagen procesada correctamente'
            }
            
        except Exception as e:
            return {'error': f'Error procesando imagen: {str(e)}'}
    
    async def get_images(self, stroke_prediction_id: Optional[int] = None, limit: int = 50) -> List[Dict]:
        """Obtener imágenes"""
        try:
            from database.supabase_client import get_image_predictions
            return await get_image_predictions(stroke_prediction_id=stroke_prediction_id, limit=limit)
        except Exception as e:
            print(f"❌ Error obteniendo imágenes: {e}")
            return []
    
    async def get_status(self) -> Dict:
        """Estado del servicio"""
        return {
            'pipeline_loaded': self.pipeline_loaded,
            'service_ready': self.pipeline_loaded
        }


# Instancia global
_service = None

def get_service() -> ImageService:
    """Obtener instancia del servicio"""
    global _service
    if _service is None:
        _service = ImageService()
    return _service

# Funciones principales para endpoints
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


# Test
if __name__ == "__main__":
    import asyncio
    import sys
    from pathlib import Path
    
    # Agregar rutas
    current_dir = Path(__file__).resolve().parent
    backend_app_dir = current_dir.parent
    project_root = current_dir.parent.parent.parent
    sys.path.insert(0, str(backend_app_dir))
    sys.path.insert(0, str(project_root))
    
    async def test():
        print("🧪 Test Image Service")
        try:
            # Test pipeline
            status = await get_image_pipeline_status()
            print(f"Pipeline: {status}")
            
            # Test DB
            from database.supabase_client import test_db_connection
            db_ok = await test_db_connection()
            print(f"Database: {db_ok}")
            
            # Test imágenes existentes
            images = await get_all_image_predictions(limit=5)
            print(f"Imágenes existentes: {len(images)}")
            
            if status['pipeline_loaded'] and db_ok:
                print("✅ Image Service listo")
            else:
                print("❌ Image Service con problemas")
                
        except Exception as e:
            print(f"❌ Error en test: {e}")
    
    asyncio.run(test())