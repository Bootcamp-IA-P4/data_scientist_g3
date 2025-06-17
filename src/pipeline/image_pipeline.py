"""
Pipeline de predicción de imágenes CNN para stroke
Basado en modelo TorchScript con 98.13% accuracy
"""

import torch
from torchvision import transforms
from PIL import Image
import zipfile
import tempfile
import os
import time
from pathlib import Path
from typing import Dict, Union, Optional

class StrokeImagePipeline:
    """Pipeline CNN para predicción de stroke"""
    
    def __init__(self, model_filename: str = "modelo_cnn_stroke_pytorch.zip"):
        self.model = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model_loaded = False
        self.classes = ['Normal', 'Stroke']
        
        # Transformaciones exactas del entrenamiento
        self.transforms = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.Grayscale(num_output_channels=3),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5])
        ])
        
        self._load_model(model_filename)
    
    def _load_model(self, model_filename: str):
        """Cargar modelo TorchScript desde ZIP"""
        try:
            # Ruta al modelo
            current_dir = Path(__file__).resolve().parent
            project_root = current_dir.parent.parent
            model_path = project_root / "models" / "CNN_PyTorch" / model_filename
            
            if not model_path.exists():
                print(f"❌ Modelo no encontrado: {model_path}")
                return
            
            # Extraer y cargar TorchScript
            with zipfile.ZipFile(model_path, 'r') as zip_file:
                pytorch_files = [f for f in zip_file.namelist() if f.endswith(('.pt', '.pth'))]
                
                if not pytorch_files:
                    print(f"❌ No hay archivos .pt en el ZIP")
                    return
                
                with tempfile.NamedTemporaryFile(suffix='.pt', delete=False) as temp_file:
                    temp_file.write(zip_file.read(pytorch_files[0]))
                    temp_path = temp_file.name
                
                # Cargar como TorchScript
                self.model = torch.jit.load(temp_path, map_location=self.device)
                self.model.eval()
                
                # Limpiar archivo temporal
                os.unlink(temp_path)
                
                self.model_loaded = True
                print(f"✅ Modelo TorchScript cargado correctamente")
                
        except Exception as e:
            print(f"❌ Error cargando modelo: {e}")
            self.model_loaded = False
    
    def predict(self, image_data: Union[bytes, str, Image.Image]) -> Dict:
        """Predicción principal"""
        if not self.model_loaded:
            raise Exception("Modelo no cargado")
        
        start_time = time.time()
        
        # Preprocesar imagen
        image_tensor = self._preprocess_image(image_data)
        
        # Inferencia
        with torch.no_grad():
            outputs = self.model(image_tensor.unsqueeze(0).to(self.device))
            probabilities = torch.softmax(outputs, dim=1)
            prediction = torch.argmax(outputs, dim=1).item()
            stroke_probability = probabilities[0][1].item()
        
        # Calcular nivel de riesgo
        if stroke_probability < 0.3:
            risk_level = "Bajo"
        elif stroke_probability < 0.6:
            risk_level = "Medio"
        elif stroke_probability < 0.9:
            risk_level = "Alto"
        else:
            risk_level = "Crítico"
        
        processing_time = int((time.time() - start_time) * 1000)
        
        return {
            'prediction': prediction,
            'probability': float(stroke_probability),
            'risk_level': risk_level,
            'processing_time_ms': processing_time,
            'model_confidence': float(torch.max(probabilities).item())
        }
    
    def _preprocess_image(self, image_data: Union[bytes, str, Image.Image]) -> torch.Tensor:
        """Preprocesar imagen"""
        # Convertir a PIL Image
        if isinstance(image_data, bytes):
            import io
            image = Image.open(io.BytesIO(image_data))
        elif isinstance(image_data, str):
            image = Image.open(image_data)
        else:
            image = image_data
        
        # Convertir a RGB y aplicar transformaciones
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        return self.transforms(image)
    
    def validate_image(self, image_data: Union[bytes, str, Image.Image]) -> Dict:
        """Validación básica de imagen"""
        try:
            # Abrir imagen
            if isinstance(image_data, bytes):
                import io
                image = Image.open(io.BytesIO(image_data))
                file_size = len(image_data)
            elif isinstance(image_data, str):
                image = Image.open(image_data)
                file_size = Path(image_data).stat().st_size
            else:
                image = image_data
                file_size = None
            
            width, height = image.size
            format_name = image.format or "Unknown"
            
            # Validaciones básicas
            errors = []
            if file_size and file_size > 10 * 1024 * 1024:  # Max 10MB
                errors.append("Archivo muy grande (máximo 10MB)")
            if width < 32 or height < 32:
                errors.append("Imagen muy pequeña (mínimo 32x32)")
            if format_name not in ['JPEG', 'PNG', 'WEBP', 'BMP']:
                errors.append(f"Formato no soportado: {format_name}")
            
            return {
                "valid": len(errors) == 0,
                "errors": errors,
                "metadata": {
                    "width": width,
                    "height": height,
                    "format": format_name,
                    "file_size_mb": round(file_size / 1024 / 1024, 2) if file_size else None
                }
            }
            
        except Exception as e:
            return {"valid": False, "errors": [f"Error: {str(e)}"]}
    
    def get_status(self) -> Dict:
        """Estado del pipeline"""
        return {
            'is_loaded': self.model_loaded,
            'device': str(self.device),
            'model_type': 'TorchScript CNN' if self.model_loaded else None,
            'classes': self.classes,
            'accuracy': '98.13%'
        }


# Instancia global (Singleton)
_pipeline = None

def get_pipeline() -> StrokeImagePipeline:
    """Obtener instancia del pipeline"""
    global _pipeline
    if _pipeline is None:
        _pipeline = StrokeImagePipeline()
    return _pipeline

# Funciones principales para el backend
def predict_stroke_from_image(image_data: Union[bytes, str, Image.Image]) -> Dict:
    """Función principal de predicción"""
    return get_pipeline().predict(image_data)

def validate_image(image_data: Union[bytes, str, Image.Image]) -> Dict:
    """Función principal de validación"""
    return get_pipeline().validate_image(image_data)

def get_pipeline_status() -> Dict:
    """Estado del pipeline"""
    return get_pipeline().get_status()


# Test
if __name__ == "__main__":
    print("🧪 Test Pipeline CNN")
    print("=" * 40)
    
    status = get_pipeline_status()
    print(f"Estado: {status}")
    
    if status['is_loaded']:
        print("✅ Pipeline listo para usar en backend")
    else:
        print("❌ Error en pipeline")