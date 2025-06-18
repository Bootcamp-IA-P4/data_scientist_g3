"""Aplicación FastAPI para API de predicción de stroke - macOS Fix"""

import os
import sys

# ✅ FIX PARA macOS - Evitar segmentation fault con PyTorch
if sys.platform == "darwin":  # macOS
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1" 
    os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv

from api.endpoints import predictions

# Cargar variables de entorno
load_dotenv()

# Crear aplicación FastAPI
app = FastAPI(
    title="API de Predicción de Stroke",
    description="API para predicción de riesgo de stroke usando modelos de ML",
    version="1.0.0"
)

# Configurar CORS para el frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=os.getenv("CORS_ORIGINS", "http://localhost:8050").split(","),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Incluir rutas
app.include_router(predictions.router)

# Endpoint raíz
@app.get("/")
async def root():
    return {"message": "API de Predicción de Stroke funcionando correctamente"}

if __name__ == "__main__":
    import uvicorn
    
    # ✅ CONFIGURACIÓN SEGURA PARA macOS
    uvicorn_config = {
        "app": "main:app",
        "host": os.getenv("API_HOST", "0.0.0.0"), 
        "port": int(os.getenv("API_PORT", 8000)),
        "reload": False,  # Disable reload en macOS para evitar problemas
        "workers": 1,     # Solo 1 worker en macOS
    }
    
    print("🚀 Iniciando servidor con configuración segura para macOS...")
    uvicorn.run(**uvicorn_config)