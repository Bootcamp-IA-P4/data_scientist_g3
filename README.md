
# 🧠 NeuroWise AI Prediction Platform

## 1. 📱 Capturas de Pantalla

<div align="center"> <img src="https://via.placeholder.com/800x400/2563EB/FFFFFF?text=NeuroWise+Desktop+View" alt="Vista Desktop" width="450" style="margin-right: 20px;"/> <img src="https://via.placeholder.com/300x600/8B5CF6/FFFFFF?text=NeuroWise+Mobile+View" alt="Vista Móvil" width="135"/> <br/> <em>Interfaz Desktop y Móvil - Diseño completamente responsivo</em> </div>

## 2. 🌐 Demo en Vivo

🚀  **Aplicación desplegada**: [Próximamente - En desarrollo]

_Nota: El proyecto se encuentra actualmente en desarrollo activo. La demo estará disponible próximamente._

## 3. 📚 Descripción del Proyecto

NeuroWise es una plataforma avanzada de inteligencia artificial que implementa un sistema de clasificación multimodal para la predicción de riesgo de ictus. El sistema combina dos enfoques complementarios:

-   **Análisis de Datos Clínicos**: Utilizando XGBoost optimizado para analizar factores de riesgo tradicionales
-   **Análisis de Neuroimágenes**: Empleando redes neuronales convolucionales (CNN) para el análisis de tomografías computarizadas

La plataforma puede clasificar pacientes en cuatro niveles de riesgo:  **Bajo**,  **Medio**,  **Alto**  y  **Crítico**, proporcionando recomendaciones médicas específicas para cada caso.

## 4. 🏗️ Estructura del Proyecto

```
data_scientist_g3/
│
├── 🐍 backend/                                      # Backend FastAPI
│   └── app/
│       ├── api/                                     # Endpoints de la API
│       │   └── endpoints/
│       │       └── predictions.py                   # Endpoints de predicción
│       │
│       ├── database/                                # Gestión de base de datos
│       │   └── supabase_client.py                   # Cliente PostgreSQL
│       │
│       ├── models/                                  # Esquemas y validación
│       │   └── schemas.py                           # Modelos Pydantic
│       │
│       ├── services/                                # Lógica de negocio
│       │   ├── stroke_service.py                    # Servicio de predicción clínica
│       │   └── image_service.py                     # Servicio de análisis de imágenes
│       │
│       └── main.py                                  # Aplicación FastAPI principal
│
├── 🖥️ frontend/                                    # Frontend Dash/Plotly
│   ├── assets/                                      # Recursos estáticos
│   │   ├── style.css                                # Estilos principales
│   │   ├── navbar.css                               # Estilos navegación
│   │   ├── image_prediction.css                     # Estilos predicción imagen
│   │   ├── history.css                              # Estilos historial
│   │   ├── about.css                                # Estilos página equipo
│   │   └── background-video.mp4                     # Video de fondo
│   │
│   ├── components/                                  # Componentes reutilizables
│   │   ├── form_components.py                       # Formularios de predicción
│   │   ├── image_components.py                      # Componentes de imagen
│   │   ├── history_components.py                    # Componentes de historial
│   │   ├── navbar_components.py                     # Navegación
│   │   └── results_components.py                    # Resultados y métricas
│   │
│   ├── pages/                                       # Páginas principales
│   │   ├── about.py                                 # Página del equipo
│   │   ├── history.py                               # Historial de predicciones
│   │   └── image_prediction.py                      # Predicción por imagen
│   │
│   ├── services/                                    # Comunicación con API
│   │   └── api_client.py                            # Cliente HTTP para backend
│   │
│   ├── config/                                      # Configuración
│   │   └── settings.py                              # Configuración de la app
│   │
│   └── app.py                                       # Aplicación Dash principal
│
├── 🤖 models/                                       # Modelos entrenados
│   ├── xgboost/                                     # Modelo XGBoost optimizado
│   │   ├── xgboost_stroke_optimized_*.pkl           # Modelo principal
│   │   ├── optimized_model_config_*.json            # Configuración del modelo
│   │   └── OPTIMIZED_MODEL_INSTRUCTIONS_*.md        # Documentación del modelo
│   │
│   ├── CNN_PyTorch/                                 # Modelo CNN PyTorch
│   │   └── modelo_cnn_stroke_pytorch.zip            # Red neuronal convolucional
│   │
│   ├── extra_trees/                                 # Modelo Extra Trees
│   ├── ligthgbm/                                    # Modelo LightGBM
│   ├── MGB/                                         # Modelo Gradient Boosting
│   ├── lda/                                         # Análisis Discriminante Lineal
│   └── scaler_recreated.pkl                         # StandardScaler para preprocesamiento
│
├── 📊 data/                                         # Datasets
│   ├── raw/                                         # Datos originales
│   │   └── stroke_dataset.csv                       # Dataset principal de stroke
│   ├── processed/                                   # Datos procesados
│   │   └── preprocessing.csv                        # Datos limpios para ML
│   └── tc/                                          # Datos de tomografías
│       └── Brain_Data_Organised/                    # Imágenes organizadas por clase
│           ├── Normal(1551)/                        # Escáneres normales
│           └── Stroke(950)/                         # Escáneres con stroke
│
├── 🔬 src/                                          # Pipelines de ML
│   └── pipeline/
│       ├── stroke_pipeline.py                       # Pipeline de predicción clínica
│       └── image_pipeline.py                        # Pipeline de análisis de imágenes
│
├── 📓 notebooks/                                    # Jupyter Notebooks
│   ├── eda.ipynb                                    # Análisis exploratorio
│   ├── preprocessing.ipynb                          # Preprocesamiento de datos
│   ├── evaluation.ipynb                             # Evaluación de modelos
│   └── modeling/                                    # Notebooks de modelado
│     ├── mlruns/                                    # Experimentos MLFlow
│     ├── xgboost.ipynb                              # Desarrollo modelo XGBoost
│     ├── CNN_fin_v6.ipynb                           # Desarrollo modelo CNN
│     ├── lihgtGBM.ipynb                             # Modelo LightGBM
│     ├── extra_trees.py                             # Modelo Extra Trees
│     └── tc_cnn_keras.ipynb                         # CNN con TensorFlow/Keras
│
├── 🗄️ db/                                          # Base de datos
│   ├── schema.sql                                   # Esquema PostgreSQL
│   └── create_database.py                           # Script de inicialización
│
├── 🧪 tests/                                        # Suite de testing
│   ├── unit/                                        # Tests unitarios
│   │   ├── test_stroke_pipeline.py                  # Tests pipeline clínico
│   │   ├── test_image_pipeline.py                   # Tests pipeline imagen
│   │   ├── test_stroke_service.py                   # Tests servicio clínico
│   │   ├── test_image_service.py                    # Tests servicio imagen
│   │   └── test_schemas.py                          # Tests validación datos
│   │
│   ├── integration/                                 # Tests de integración
│   │   ├── test_api_endpoints.py                    # Tests endpoints API
│   │   ├── test_api_endpoints_detailed.py           # Tests detallados API
│   │   ├── test_database.py                         # Tests base de datos
│   │   ├── test_supabase_client.py                  # Tests cliente DB
│   │   ├── test_complete_workflow.py                # Tests flujo completo
│   │   └── test_system_complete.py                  # Tests sistema completo
│   │
│   ├── fixtures/                                    # Datos de prueba
│   │   └── test_data.json                           # Datos de pacientes test
│   │
│   ├── conftest.py                                  # Configuración pytest
│   ├── pytest.ini                                   # Configuración testing
│   └── requirements-test.txt                        # Dependencias testing
│
├── 📈 reports/                                      # Reportes y métricas
│   ├── figures/                                     # Gráficos de rendimiento
│   │   ├── confusion_matrix.png                     # Matriz de confusión
│   │   ├── feature_importance.png                   # Importancia características
│   │   ├── learning_curves.png                      # Curvas de aprendizaje
│   │   ├── roc_curve.png                            # Curva ROC
│   │   └── performance_metrics.png                  # Métricas de rendimiento
│   └── performance_report.md                        # Reporte de rendimiento
│
├── 🐳 Docker/                                       # Containerización (En desarrollo)
│   ├── Dockerfile.backend                           # Container FastAPI
│   ├── Dockerfile.frontend                          # Container Dash
│   ├── docker-compose.yml                           # Orquestación completa
│   └── nginx.conf                                   # Configuración proxy
│
├── 🔧 Configuración/
│   ├── requirements.txt                             # Dependencias Python
│   ├── .env_example                                 # Variables de entorno ejemplo
│   ├── .gitignore                                   # Archivos ignorados Git
│   └── README.md                                    # Este archivo
│
└── 📖 Documentación adicional

```

## 4.1. 🛠️ Tecnologías Utilizadas

### 4.1.1. Backend

-   **Python 3.10+**
-   **FastAPI**  - Framework web moderno y rápido
-   **XGBoost**  - Modelo principal de clasificación
-   **PyTorch**  - Deep learning para análisis de imágenes
-   **PostgreSQL**  - Base de datos principal
-   **Supabase**  - Backend as a Service
-   **Pydantic**  - Validación de datos
-   **Uvicorn**  - Servidor ASGI

### 4.1.2. Frontend

-   **Python Dash**  - Framework web interactivo
-   **HTML5 & CSS3**  - Estructura y estilos
-   **JavaScript**  - Interactividad del cliente

### 4.1.3. Machine Learning

-   **Scikit-learn**  - Herramientas de ML
-   **Pandas & NumPy**  - Manipulación de datos
-   **Optuna**  - Optimización de hiperparámetros
-   **PIL/Pillow**  - Procesamiento de imágenes
-   **TorchVision**  - Transformaciones de imagen
-   **MLflow**  - Gestión del ciclo de vida de ML

### 4.1.4. Testing y Calidad

-   **Pytest**  - Framework de testing
-   **Coverage**  - Cobertura de código
-   **Black**  - Formateador de código

## 4.2. 📊 MLflow - Gestión de Experimentos

NeuroWise utiliza MLflow para gestionar el ciclo de vida completo de los experimentos de Machine Learning:

### 4.2.1. Estructura

```
src/mlflow/
├── __init__.py
└── mlflow_config.py          # Configuración central de MLflow
```

### 4.2.2. Configuración

La configuración de MLflow está centralizada en `src/mlflow/mlflow_config.py` y proporciona:

- Configuración flexible del URI de tracking
- Gestión de experimentos y runs
- Logging de métricas, parámetros y modelos
- Soporte para múltiples entornos de desarrollo

### 4.2.3. Uso en Notebooks

Los notebooks en `notebooks/modeling/` utilizan MLflow para:

- Tracking de experimentos
- Registro de hiperparámetros
- Monitoreo de métricas de rendimiento
- Versionado de modelos

### 4.2.4. Configuración del Entorno

Para configurar MLflow en tu entorno:

1. Por defecto, MLflow creará un directorio `mlruns` en `notebooks/modeling/`
2. Opcionalmente, configura tu propio URI de tracking:
   ```bash
   export MLFLOW_TRACKING_URI=<tu_uri_preferido>
   ```

### 4.2.5. Ejecución de MLflow

1. Iniciar el servidor MLflow UI (desde la raíz del proyecto):
   ```bash
   mlflow ui --port 5000
   ```

2. Acceder a la interfaz web:
   - Abre tu navegador y visita: `http://localhost:5000`
   - Visualiza experimentos, métricas y modelos

3. Ejecutar notebooks con tracking:
   - Los notebooks en `notebooks/modeling/` ya incluyen la configuración necesaria
   - MLflow registrará automáticamente métricas y parámetros durante la ejecución

4. Ver resultados:
   - Compara diferentes runs en la UI de MLflow
   - Analiza métricas y parámetros
   - Descarga modelos guardados

## 5. 📋 Requisitos Previos

-   Python 3.10
-   PostgreSQL 12+ (o cuenta Supabase)
-   Git
-   8GB RAM mínimo (recomendado para modelos ML)
-   GPU opcional (acelera el análisis de imágenes)

## 6. 🚀 Instalación y Configuración

### 6.1. Clonar el repositorio

```bash
git clone https://github.com/tu-usuario/data_scientist_g3.git
cd data_scientist_g3

```

### 6.2. Configurar entorno virtual

```bash
python -m venv venv
# Windows: .\venv\Scripts\activate
# Linux/Mac: source venv/bin/activate
pip install -r requirements.txt

```

### 6.3. Configurar variables de entorno

```bash
cp .env_example .env
# Editar .env con tus credenciales de Supabase

```

### 6.4. Ejecutar el sistema

```bash
# Backend
cd backend/app
python main.py

# Frontend (nueva terminal)
python frontend/app.py

```

### 6.5. Verificar instalación

-   **Backend API**: http://localhost:8000
-   **Frontend**: http://localhost:8050
-   **Documentación**: http://localhost:8000/docs

## 7. 🧪 Ejecutar Tests

```bash
# Todos los tests
pytest

# Tests críticos solamente
pytest -m critical

# Tests con cobertura
pytest --cov=src --cov=backend/app --cov-report=html

# Tests específicos
pytest tests/unit/test_stroke_pipeline.py -v
pytest tests/integration/test_api_endpoints.py -v

```

## 8. 🐳 Docker (En desarrollo)

```bash
# Construir y ejecutar con Docker Compose

# Solo backend

# Solo frontend

```

## 9. 📊 MLFlow (En desarrollo)

```bash
# Iniciar MLflow server

# Acceder a experimentos
# http://localhost:5000

```

## 10. 🔍 Verificación del Sistema

Una vez completada la instalación, verifica que todo funcione correctamente:

-   **Backend API**: http://localhost:8000/health
-   **Frontend**: http://localhost:8050
-   **Estado de modelos**: http://localhost:8000/pipeline/status
-   **Documentación API**: http://localhost:8000/docs

## 11. 🎯 Características Principales

### 11.1. 🩺 Predicción Clínica

-   Análisis de 17 características médicas y demográficas
-   Modelo XGBoost optimizado con 98.5% de precisión
-   Interpretabilidad mediante análisis de importancia de características
-   Clasificación en 4 niveles de riesgo con recomendaciones específicas

### 11.2. 📷 Análisis de Neuroimágenes

-   Procesamiento de tomografías computarizadas del cerebro
-   Red neuronal convolucional con 98.13% de accuracy
-   Soporte para formatos JPEG, PNG, WEBP, BMP
-   Validación automática de calidad de imagen

### 11.3. 📊 Dashboard Interactivo

-   Interfaz responsive para desktop y móvil
-   Historial completo de predicciones

### 11.4. 🔄 Análisis Multimodal

-   Combinación de datos clínicos e imágenes médicas
-   Correlación entre diferentes métodos de predicción
-   Validación cruzada de resultados
-   Recomendaciones médicas integradas

## 12. 📊 Modelos de Machine Learning

### 12.1. 🎯  **Estrategia de Screening Dual**

Nuestra propuesta comercial única implementa un sistema de screening de dos capas que maximiza la detección temprana:

1.  **Primera Capa - Screening Masivo**: XGBoost optimizado para alta sensibilidad (78% recall)
2.  **Segunda Capa - Confirmación**: CNN con alta precisión (98.13% accuracy) para casos sospechosos

### 12.2.  **XGBoost Optimizado (Screening Primario)**

-   **Tipo**: Gradient Boosting para clasificación binaria
-   **Precisión**: 85% en conjunto de prueba
-   **F1-Score**: 0.266 (optimizado para recall médico)
-   **ROC-AUC**: 0.848
-   **Recall**: 78% -  **Detecta 78 de cada 100 casos reales**
-   **Características**: 17 variables médicas y demográficas
-   **Optimización**: 161 trials con Optuna
-   **Ventaja Clínica**: Alto recall minimiza casos perdidos, ideal para screening inicial

### 12.3.  **Red Neuronal Convolucional (Confirmación)**

-   **Arquitectura**: CNN personalizada desarrollada con Keras y PyTorch
-   **Framework Final**: PyTorch (mejores resultados vs Keras)
-   **Precisión**: 98.13% en imágenes de tomografía
-   **ROC-AUC**: 0.987 (imagen 2)
-   **Input**: Imágenes 224x224 píxeles, escala de grises
-   **Dataset**: 2,501 escáneres cerebrales (1,551 normales, 950 con stroke)
-   **Formato**: TorchScript para optimización en producción
-   **Ventaja Clínica**: Alta precisión confirma casos sospechosos, reduce falsos positivos

### 12.4.  **Modelos de Investigación**

-   **LightGBM**: Modelo rápido para comparación
-   **Extra Trees**: Ensemble method con interpretabilidad
-   **Linear Discriminant Analysis**: Modelo lineal de referencia
-   **Gradient Boosting**: Implementación sklearn

## 13. 🔄 Flujo de Trabajo

### 13.1. Predicción Clínica

1.  Usuario ingresa datos médicos del paciente
2.  Validación de rangos médicos (edad 0-120, glucosa 50-500, etc.)
3.  Preprocesamiento con StandardScaler y codificación categórica
4.  Predicción con modelo XGBoost optimizado
5.  Cálculo de nivel de riesgo y recomendaciones
6.  Almacenamiento en base de datos PostgreSQL

### 13.2. Análisis de Imagen

1.  Upload de tomografía computarizada
2.  Validación de formato, tamaño y calidad
3.  Preprocesamiento de imagen (resize, normalización)
4.  Análisis con red neuronal convolucional
5.  Vinculación con predicción clínica existente
6.  Correlación de resultados multimodales

### 13.3. Historial y Seguimiento

1.  Visualización de predicciones históricas
2.  Estadísticas agregadas y tendencias
3.  Filtrado por nivel de riesgo y estado de imagen
4.  Exportación de datos para análisis adicional

## 14. 🏥 Impacto Clínico y Propuesta de Valor

### 14.1. 💡  **Ventaja Comercial: Sistema de Screening Dual**

NeuroWise ofrece una propuesta única en el mercado:

**🔍 Screening Masivo (XGBoost)**

-   Análisis rápido y económico de datos clínicos básicos
-   Alto recall (78%) - No se pierden casos críticos
-   Falsos positivos controlados - Dirigidos a segunda capa
-   Escalable para poblaciones grandes

**🎯 Confirmación Precisa (CNN)**

-   Análisis de tomografías solo para casos sospechosos
-   Precisión excepcional (98.13%) - Minimiza falsos positivos
-   Reduce costos de imaging innecesario
-   Optimiza recursos médicos especializados

### 14.2. 📈 Métricas de Rendimiento

#### 14.2.1. Modelo XGBoost (Screening)

-   **Sensibilidad (Recall)**: 78% - Detecta 78 de cada 100 casos reales
-   **Especificidad**: 85% - Identifica correctamente casos sanos
-   **F1-Score**: 0.266 - Balanceado para minimizar casos perdidos
-   **ROC-AUC**: 0.848 - Excelente capacidad discriminativa

#### 14.2.2. Modelo CNN (Confirmación)

-   **Accuracy**: 98.13% - Precisión excepcional en imágenes
-   **ROC-AUC**: 0.987 - Capacidad discriminativa sobresaliente
-   **Precisión por clase**: 97%+ para stroke y normal
-   **Recall por clase**: 95%+ para ambas categorías

### 14.3. 🎯 Flujo Clínico Optimizado

1.  **Screening inicial**  con datos básicos del paciente
2.  **Casos de bajo riesgo**  → Seguimiento preventivo estándar
3.  **Casos sospechosos**  → Derivación para tomografía
4.  **Confirmación con CNN**  → Diagnóstico de alta precisión
5.  **Decisión clínica informada**  con doble validación

### 14.4. Interpretación de Niveles de Riesgo

-   **Bajo (0-30%)**: Mantener controles preventivos regulares
-   **Medio (30-60%)**: Evaluación médica adicional recomendada
-   **Alto (60-90%)**: Consulta neurológica urgente necesaria
-   **Crítico (90-100%)**: Atención médica inmediata requerida

## 15. 👥 Nuestro Equipo

Somos un equipo multidisciplinario de Data Scientists especializados en inteligencia artificial aplicada a la salud:

### 15.1. 🧑‍💼  [Pepe](https://github.com/peperuizdev)  - Scrum Manager

Especialista en machine learning y arquitectura de software. Responsable de la coordinación del proyecto y la implementación de modelos de clasificación.

### 15.2. 👩‍💻  [Maryna](https://github.com/MarynaDRST)  - Developer

Desarrolladora de modelos de machine learning y redes neuronales. Especializada en deep learning y procesamiento de imágenes médicas.

### 15.3. 👨‍🎨  [Jorge](https://github.com/Jorgeluuu)  - Developer

Creador de modelos de machine learning y especialista en optimización de algoritmos. Enfocado en el rendimiento y escalabilidad del sistema.

### 15.4. 👩‍💼  [Mariela](https://github.com/marie-adi)  - Developer

Diseñadora de experiencia de usuario y desarrolladora frontend. Creadora de la interfaz intuitiva y responsiva de la plataforma.

### 15.5. 👨‍🔬  [Maximiliano](https://github.com/MaximilianoScarlato)  - Data Scientist

Científico de datos especializado en análisis de modelos de redes neuronales y evaluación de rendimiento de sistemas de ML.

## 16. 🤝 Contribución

Las contribuciones son bienvenidas. Para contribuir:

1.  Fork el proyecto
2.  Crea una rama para tu feature (`git checkout -b feature/AmazingFeature`)
3.  Commit tus cambios (`git commit -m 'Add some AmazingFeature'`)
4.  Push a la rama (`git push origin feature/AmazingFeature`)
5.  Abre un Pull Request

### 16.1. Estándares de Desarrollo

-   Seguir PEP 8 para código Python
-   Incluir tests para nuevas funcionalidades
-   Documentar funciones y clases

## 17. 📄 Estructura de Testing

### 17.1. Tests Unitarios

-   **Pipeline de stroke**: Validación de transformaciones y predicciones
-   **Pipeline de imagen**: Procesamiento y validación de imágenes
-   **Servicios**: Lógica de negocio y manejo de errores
-   **Esquemas**: Validación de datos de entrada

### 17.2. Tests de Integración

-   **API endpoints**: Funcionamiento completo de la API
-   **Base de datos**: Persistencia y recuperación de datos
-   **Flujo completo**: Integración end-to-end
-   **Sistema completo**: Validación del sistema completo

## 18. ⚠️ Consideraciones Médicas

**IMPORTANTE**: Esta herramienta está diseñada únicamente con fines educativos y de investigación. No sustituye el juicio clínico profesional ni debe utilizarse como único criterio para decisiones médicas.

### 18.1. Limitaciones

-   Los modelos se entrenaron con datos específicos que pueden no representar todas las poblaciones
-   Las predicciones deben interpretarse siempre en conjunto con la evaluación clínica
-   Se requiere validación adicional antes de cualquier uso clínico real
-   Los resultados pueden variar según la calidad de los datos de entrada

### 18.2. Recomendaciones

-   Siempre consultar con profesionales médicos certificados
-   Utilizar como herramienta de apoyo, no de diagnóstico definitivo
-   Validar resultados con métodos clínicos establecidos
-   Considerar el contexto clínico completo del paciente

## 20. 🚀 Instrucciones para Dockerizar y Renderizar el Proyecto

---

### 20.1. Configuración del archivo `.env`

1. Usa el archivo `.env` que adjuntaste como base.
2. **Para Docker/Render:**  
   - Descomenta las líneas bajo el bloque `# Backend Configuration - DOCKER/RENDER` y comenta las de LOCAL.
   - Haz lo mismo para el frontend si lo vas a dockerizar/renderizar.
3. **Para Local:**  
   - Deja comentadas las líneas de Docker/Render y descomentadas las de LOCAL.

---

### 20.2. Dockerizar localmente

#### 20.2.1. Ubícate en la raíz del proyecto

```bash
cd /ruta/a/tu/proyecto/data_scientist_g3
```

#### 20.2.2. Levanta los servicios con Docker Compose

```bash
docker compose up --build
```

Esto construirá y levantará tanto el backend como el frontend.

#### 20.2.3. Accede a las aplicaciones

- **Frontend:** [http://127.0.0.1:8050](http://127.0.0.1:8050)
- **Backend:** [http://localhost:8000](http://localhost:8000)

---

### 20.3. Renderizar (Desplegar en Render.com)

#### 20.3.1. Mueve el Dockerfile del backend

Mueve el archivo Dockerfile de `backend/app` a la raíz del proyecto, junto a `docker-compose.yml`:

```bash
mv backend/app/Dockerfile ./
```

Asegúrate de que el Dockerfile y docker-compose.yml estén en la raíz del repo.

---

#### 20.3.2. Backend en Render

1. **Nuevo servicio > Web Service**
2. **Repositorio:**  
   ```
   https://github.com/Bootcamp-IA-P4/data_scientist_g3
   ```
3. **Branch:**  
   ```
   feature/api-refactor
   ```
4. **Dockerfile Path:**  
   ```
   ./Dockerfile
   ```
5. **Docker Build Context Directory:**  
   ```
   .
   ```
6. **Docker Command:**  
   (deja vacío para usar el CMD del Dockerfile)
7. **Variables de entorno:**  
   Copia todas las variables del `.env` en la sección Environment Variables de Render, por ejemplo:
   - `CORS_ORIGINS`
   - `DATABASE_URL`
   - `ENVIRONMENT`
   - `SUPABASE_ANON_KEY`
   - `SUPABASE_DB_PASSWORD`
   - `SUPABASE_SERVICE_ROLE_KEY`
   - `SUPABASE_URL`

---

#### 20.3.3. Frontend en Render

1. **Nuevo servicio > Web Service**
2. **Repositorio:**  
   ```
   https://github.com/Bootcamp-IA-P4/data_scientist_g3
   ```
3. **Branch:**  
   ```
   feature/api-refactor
   ```
4. **Root Directory:**  
   ```
   frontend
   ```
5. **Dockerfile Path:**  
   ```
   frontend/Dockerfile
   ```
6. **Docker Build Context Directory:**  
   ```
   frontend
   ```
7. **Variables de entorno:**  
   Copia las necesarias del `.env` (por ejemplo, `API_BASE_URL`, etc).

---

### 20.4. Edición de `image_service.py` para Render/Docker

**Ruta:**  
`backend/app/services/image_service.py`  
**Líneas:** 100 a 131

#### Para Render/Docker

1. **Comenta** el bloque de desarrollo local (ruta relativa).
2. **Descomenta** el bloque para Docker/Render (ruta absoluta):

```python
#DESARROLLO PARA PRODUCCIÓN
# @property
# def is_available(self) -> bool:
#     """Check si el pipeline está disponible sin cargarlo"""
#     if self._pipeline_loaded:
#         return True
#     if self._pipeline_error:
#         return False
#     try:
#         current_dir = Path(__file__).resolve().parent
#         project_root = current_dir.parent.parent.parent
#         model_path = project_root / "models" / "CNN_PyTorch" / "modelo_cnn_stroke_pytorch.zip"
#         return model_path.exists()
#     except:
#         return False

# SOLO PARA DOCKERIZADO - NO CARGAR EN PRODUCCIÓN
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
```

---

### 20.5. URLs de acceso en Render

- **Backend:**  
  [https://data-scientist-g3-wwo1.onrender.com](https://data-scientist-g3-wwo1.onrender.com)
- **Frontend:**  
  [https://data-scientist-g3-1-d1kn.onrender.com](https://data-scientist-g3-1-d1kn.onrender.com)

---

### 20.6. Notas

- **El modelo debe estar en la ruta `/backend/models/CNN_PyTorch/modelo_cnn_stroke_pytorch.zip`** dentro del contenedor Docker y en el repo para Render.
- **No subas claves sensibles a tu repo público.** Usa el panel de variables de entorno de Render.
- **Revisa los logs de Render** para solucionar cualquier error de rutas o dependencias.

---

## 21. 📝 Licencia

Este proyecto está distribuido bajo la Licencia Factoria F5

----------

**Desarrollado con ❤️ por el equipo Data Scientists G3 - Factoría F5**

_Aplicando inteligencia artificial para mejorar la detección temprana de ictus y salvar vidas._
