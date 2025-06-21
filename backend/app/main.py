# """Aplicación FastAPI para API de predicción de stroke"""

# import os
# import sys
# import platform

# # FIX PARA - Evitar segmentation fault con PyTorch
# if sys.platform == "darwin":  # macOS
#     os.environ["OMP_NUM_THREADS"] = "1"
#     os.environ["MKL_NUM_THREADS"] = "1" 
#     os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
#     os.environ["TOKENIZERS_PARALLELISM"] = "false"
# elif sys.platform == "win32":  # Windows
#     os.environ["OMP_NUM_THREADS"] = "1"
#     os.environ["MKL_NUM_THREADS"] = "1"
#     os.environ["TOKENIZERS_PARALLELISM"] = "false"

# from fastapi import FastAPI
# from fastapi.middleware.cors import CORSMiddleware
# from dotenv import load_dotenv

# from api.endpoints import predictions

# # Cargar variables de entorno
# load_dotenv()

# # Crear aplicación FastAPI
# app = FastAPI(
#     title="API de Predicción de Stroke",
#     description="API para predicción de riesgo de stroke usando modelos de ML",
#     version="1.0.0"
# )

# # Configurar CORS para el frontend
# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=os.getenv("CORS_ORIGINS", "http://localhost:8050").split(","),
#     allow_credentials=True,
#     allow_methods=["*"],
#     allow_headers=["*"],
# )

# # Incluir rutas
# app.include_router(predictions.router)

# # Endpoint raíz
# @app.get("/")
# async def root():
#     return {"message": "API de Predicción de Stroke funcionando correctamente"}

# # Health check endpoint
# @app.get("/health")
# async def health_check():
#     return {"status": "healthy", "platform": platform.system()}

# if __name__ == "__main__":
#     import uvicorn
    
#     # CONFIGURACIÓN PARA DIFERENTES PLATAFORMAS
#     if sys.platform == "win32":
#         # Windows - usar asyncio.WindowsProactorEventLoopPolicy
#         import asyncio
#         asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())
        
#         uvicorn_config = {
#             "app": "main:app",
#             "host": os.getenv("API_HOST", "0.0.0.0"), # Para dockerizado
#             # "host": os.getenv("API_HOST", "127.0.0.1"), # Para desarrollo local
#             "port": int(os.getenv("API_PORT", 8000)),
#             "reload": False,
#             "workers": 1,
#             "loop": "asyncio",
#         }
#         print("🚀 Iniciando servidor con configuración para Windows...")
#     else:
#         # macOS/Linux
#         uvicorn_config = {
#             "app": "main:app",
#             "host": os.getenv("API_HOST", "0.0.0.0"),
#             "port": int(os.getenv("API_PORT", 8000)),
#             "reload": False,
#             "workers": 1,
#         }
#         print("🚀 Iniciando servidor con configuración para Unix...")
    
#     uvicorn.run(**uvicorn_config)




"""Aplicación FastAPI para API de predicción de stroke"""

import os
import sys
import platform

# FIX PARA - Evitar segmentation fault con PyTorch
if sys.platform == "darwin":  # macOS
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1" 
    os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
elif sys.platform == "win32":  # Windows
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
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

# Health check endpoint
@app.get("/health")
async def health_check():
    return {"status": "healthy", "platform": platform.system()}

if __name__ == "__main__":
    import uvicorn

    # ⬇️ CONFIGURACIÓN UNIVERSAL SEGÚN VARIABLES DE ENTORNO
    # Solo cambia API_HOST y API_PORT en tu .env según el entorno:
    # - Local: API_HOST=127.0.0.1
    # - Docker/Render: API_HOST=0.0.0.0

    # Si usas Windows, puedes mantener el bloque asyncio, pero no es obligatorio.
    # Si quieres máxima portabilidad, puedes dejar solo este bloque:
    uvicorn_config = {
        "app": "main:app",
        "host": os.getenv("API_HOST", "0.0.0.0"),  # ← Cambia en .env según entorno
        "port": int(os.getenv("API_PORT", 8000)),
        "reload": False,
        "workers": 1,
    }
    print(f"🚀 Iniciando servidor en {uvicorn_config['host']}:{uvicorn_config['port']} ...")
    uvicorn.run(**uvicorn_config)

    # ⬅️ Ahora solo cambias API_HOST y API_PORT en .env, no el código.
    # Ejemplo .env local:
    # API_HOST=127.0.0.1
    # API_PORT=8000
    # Ejemplo .env Docker/Render:
    # API_HOST=0.0.0.0
    # API_PORT=8000