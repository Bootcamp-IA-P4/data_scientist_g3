
# 🧠 NeuroWise AI Prediction Platform

## 📱 Capturas de Pantalla

<div align="center"> <img src="https://via.placeholder.com/800x400/2563EB/FFFFFF?text=NeuroWise+Desktop+View" alt="Vista Desktop" width="450" style="margin-right: 20px;"/> <img src="https://via.placeholder.com/300x600/8B5CF6/FFFFFF?text=NeuroWise+Mobile+View" alt="Vista Móvil" width="135"/> <br/> <em>Interfaz Desktop y Móvil - Diseño completamente responsivo</em> </div>

## 🌐 Demo en Vivo

🚀  **Aplicación desplegada**: [Próximamente - En desarrollo]

_Nota: El proyecto se encuentra actualmente en desarrollo activo. La demo estará disponible próximamente._

## 📚 Descripción del Proyecto

NeuroWise es una plataforma avanzada de inteligencia artificial que implementa un sistema de clasificación multimodal para la predicción de riesgo de ictus. El sistema combina dos enfoques complementarios:

-   **Análisis de Datos Clínicos**: Utilizando XGBoost optimizado para analizar factores de riesgo tradicionales
-   **Análisis de Neuroimágenes**: Empleando redes neuronales convolucionales (CNN) para el análisis de tomografías computarizadas

La plataforma puede clasificar pacientes en cuatro niveles de riesgo:  **Bajo**,  **Medio**,  **Alto**  y  **Crítico**, proporcionando recomendaciones médicas específicas para cada caso.

## 🏗️ Estructura del Proyecto

```
data_scientist_g3/
│
├── 🐍 backend/                                # Backend FastAPI
│   └── app/
│       ├── api/                               # Endpoints de la API
│       │   └── endpoints/
│       │       └── predictions.py             # Endpoints de predicción
│       │
│       ├── database/                          # Gestión de base de datos
│       │   └── supabase_client.py             # Cliente PostgreSQL
│       │
│       ├── models/                            # Esquemas y validación
│       │   └── schemas.py                     # Modelos Pydantic
│       │
│       ├── services/                          # Lógica de negocio
│       │   ├── stroke_service.py              # Servicio de predicción clínica
│       │   └── image_service.py               # Servicio de análisis de imágenes
│       │
│       └── main.py                            # Aplicación FastAPI principal
│
├── 🖥️ frontend/                              # Frontend Dash/Plotly
│   ├── assets/                                # Recursos estáticos
│   │   ├── style.css                          # Estilos principales
│   │   ├── navbar.css                         # Estilos navegación
│   │   ├── image_prediction.css               # Estilos predicción imagen
│   │   ├── history.css                        # Estilos historial
│   │   ├── about.css                          # Estilos página equipo
│   │   └── background-video.mp4               # Video de fondo
│   │
│   ├── components/                            # Componentes reutilizables
│   │   ├── form_components.py                 # Formularios de predicción
│   │   ├── image_components.py                # Componentes de imagen
│   │   ├── history_components.py              # Componentes de historial
│   │   ├── navbar_components.py               # Navegación
│   │   └── results_components.py              # Resultados y métricas
│   │
│   ├── pages/                                 # Páginas principales
│   │   ├── about.py                           # Página del equipo
│   │   ├── history.py                         # Historial de predicciones
│   │   └── image_prediction.py                # Predicción por imagen
│   │
│   ├── services/                              # Comunicación con API
│   │   └── api_client.py                      # Cliente HTTP para backend
│   │
│   ├── config/                                # Configuración
│   │   └── settings.py                        # Configuración de la app
│   │
│   └── app.py                                 # Aplicación Dash principal
│
├── 🤖 models/                                 # Modelos entrenados
│   ├── xgboost/                               # Modelo XGBoost optimizado
│   │   ├── xgboost_stroke_optimized_*.pkl     # Modelo principal
│   │   ├── optimized_model_config_*.json      # Configuración del modelo
│   │   └── OPTIMIZED_MODEL_INSTRUCTIONS_*.md  # Documentación del modelo
│   │
│   ├── CNN_PyTorch/                           # Modelo CNN PyTorch
│   │   └── modelo_cnn_stroke_pytorch.zip      # Red neuronal convolucional
│   │
│   ├── extra_trees/                           # Modelo Extra Trees
│   ├── ligthgbm/                              # Modelo LightGBM
│   ├── MGB/                                   # Modelo Gradient Boosting
│   ├── lda/                                   # Análisis Discriminante Lineal
│   └── scaler_recreated.pkl                   # StandardScaler para preprocesamiento
│
├── 📊 data/                                   # Datasets
│   ├── raw/                                   # Datos originales
│   │   └── stroke_dataset.csv                # Dataset principal de stroke
│   ├── processed/                             # Datos procesados
│   │   └── preprocessing.csv                  # Datos limpios para ML
│   └── tc/                                    # Datos de tomografías
│       └── Brain_Data_Organised/              # Imágenes organizadas por clase
│           ├── Normal(1551)/                  # Escáneres normales
│           └── Stroke(950)/                   # Escáneres con stroke
│
├── 🔬 src/                                    # Pipelines de ML
│   └── pipeline/
│       ├── stroke_pipeline.py                # Pipeline de predicción clínica
│       └── image_pipeline.py                 # Pipeline de análisis de imágenes
│
├── 📓 notebooks/                              # Jupyter Notebooks
│   ├── eda.ipynb                              # Análisis exploratorio
│   ├── preprocessing.ipynb                    # Preprocesamiento de datos
│   ├── evaluation.ipynb                       # Evaluación de modelos
│   └── modeling/                              # Notebooks de modelado
│       ├── xgboost.ipynb                      # Desarrollo modelo XGBoost
│       ├── CNN_fin_v6.ipynb                   # Desarrollo modelo CNN
│       ├── lihgtGBM.ipynb                     # Modelo LightGBM
│       ├── extra_trees.py                     # Modelo Extra Trees
│       └── tc_cnn_keras.ipynb                 # CNN con TensorFlow/Keras
│
├── 🗄️ db/                                     # Base de datos
│   ├── schema.sql                             # Esquema PostgreSQL
│   └── create_database.py                     # Script de inicialización
│
├── 🧪 tests/                                  # Suite de testing
│   ├── unit/                                  # Tests unitarios
│   │   ├── test_stroke_pipeline.py            # Tests pipeline clínico
│   │   ├── test_image_pipeline.py             # Tests pipeline imagen
│   │   ├── test_stroke_service.py             # Tests servicio clínico
│   │   ├── test_image_service.py              # Tests servicio imagen
│   │   └── test_schemas.py                    # Tests validación datos
│   │
│   ├── integration/                           # Tests de integración
│   │   ├── test_api_endpoints.py              # Tests endpoints API
│   │   ├── test_api_endpoints_detailed.py     # Tests detallados API
│   │   ├── test_database.py                   # Tests base de datos
│   │   ├── test_supabase_client.py            # Tests cliente DB
│   │   ├── test_complete_workflow.py          # Tests flujo completo
│   │   └── test_system_complete.py            # Tests sistema completo
│   │
│   ├── fixtures/                              # Datos de prueba
│   │   └── test_data.json                     # Datos de pacientes test
│   │
│   ├── conftest.py                            # Configuración pytest
│   ├── pytest.ini                             # Configuración testing
│   └── requirements-test.txt                  # Dependencias testing
│
├── 📈 reports/                                # Reportes y métricas
│   ├── figures/                               # Gráficos de rendimiento
│   │   ├── confusion_matrix.png               # Matriz de confusión
│   │   ├── feature_importance.png             # Importancia características
│   │   ├── learning_curves.png                # Curvas de aprendizaje
│   │   ├── roc_curve.png                      # Curva ROC
│   │   └── performance_metrics.png            # Métricas de rendimiento
│   └── performance_report.md                  # Reporte de rendimiento
│
├── 🐳 Docker/                                 # Containerización (En desarrollo)
│   ├── Dockerfile.backend                     # Container FastAPI
│   ├── Dockerfile.frontend                    # Container Dash
│   ├── docker-compose.yml                     # Orquestación completa
│   └── nginx.conf                             # Configuración proxy
│
├── 📊 MLFlow/                                 # Gestión de experimentos (En desarrollo)
│   ├── mlruns/                                # Experimentos MLflow
│   └── artifacts/                             # Artefactos de modelos
│
├── 🔧 Configuración/
│   ├── requirements.txt                       # Dependencias Python
│   ├── .env_example                           # Variables de entorno ejemplo
│   ├── .gitignore                             # Archivos ignorados Git
│   └── README.md                              # Este archivo
│
└── 📖 Documentación adicional

```

## 🛠️ Tecnologías Utilizadas

### Backend

-   **Python 3.10+**
-   **FastAPI**  - Framework web moderno y rápido
-   **XGBoost**  - Modelo principal de clasificación
-   **PyTorch**  - Deep learning para análisis de imágenes
-   **PostgreSQL**  - Base de datos principal
-   **Supabase**  - Backend as a Service
-   **Pydantic**  - Validación de datos
-   **Uvicorn**  - Servidor ASGI

### Frontend

-   **Python Dash**  - Framework web interactivo
-   **HTML5 & CSS3**  - Estructura y estilos
-   **JavaScript**  - Interactividad del cliente

### Machine Learning

-   **Scikit-learn**  - Herramientas de ML
-   **Pandas & NumPy**  - Manipulación de datos
-   **Optuna**  - Optimización de hiperparámetros
-   **PIL/Pillow**  - Procesamiento de imágenes
-   **TorchVision**  - Transformaciones de imagen

### Testing y Calidad

-   **Pytest**  - Framework de testing
-   **Coverage**  - Cobertura de código
-   **Black**  - Formateador de código
-   **Flake8**  - Linter de código

## 📋 Requisitos Previos

-   Python 3.10
-   PostgreSQL 12+ (o cuenta Supabase)
-   Git
-   8GB RAM mínimo (recomendado para modelos ML)
-   GPU opcional (acelera el análisis de imágenes)

## 🚀 Instalación y Configuración

### 1. Clonar el repositorio

```bash
git clone https://github.com/tu-usuario/data_scientist_g3.git
cd data_scientist_g3

```

### 2. Configurar entorno virtual

```bash
python -m venv venv
# Windows: .\venv\Scripts\activate
# Linux/Mac: source venv/bin/activate
pip install -r requirements.txt

```

### 3. Configurar variables de entorno

```bash
cp .env_example .env
# Editar .env con tus credenciales de Supabase

```

### 4. Ejecutar el sistema

```bash
# Backend
cd backend/app
python main.py

# Frontend (nueva terminal)
python frontend/app.py

```

### 5. Verificar instalación

-   **Backend API**: http://localhost:8000
-   **Frontend**: http://localhost:8050
-   **Documentación**: http://localhost:8000/docs

## 🧪 Ejecutar Tests

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

## 🐳 Docker (En desarrollo)

```bash
# Construir y ejecutar con Docker Compose

# Solo backend

# Solo frontend

```

## 📊 MLFlow (En desarrollo)

```bash
# Iniciar MLflow server

# Acceder a experimentos
# http://localhost:5000

```

## 🔍 Verificación del Sistema

Una vez completada la instalación, verifica que todo funcione correctamente:

-   **Backend API**: http://localhost:8000/health
-   **Frontend**: http://localhost:8050
-   **Estado de modelos**: http://localhost:8000/pipeline/status
-   **Documentación API**: http://localhost:8000/docs

## 🎯 Características Principales

### 🩺 Predicción Clínica

-   Análisis de 17 características médicas y demográficas
-   Modelo XGBoost optimizado con 98.5% de precisión
-   Interpretabilidad mediante análisis de importancia de características
-   Clasificación en 4 niveles de riesgo con recomendaciones específicas

### 📷 Análisis de Neuroimágenes

-   Procesamiento de tomografías computarizadas del cerebro
-   Red neuronal convolucional con 98.13% de accuracy
-   Soporte para formatos JPEG, PNG, WEBP, BMP
-   Validación automática de calidad de imagen

### 📊 Dashboard Interactivo

-   Interfaz responsive para desktop y móvil
-   Historial completo de predicciones

### 🔄 Análisis Multimodal

-   Combinación de datos clínicos e imágenes médicas
-   Correlación entre diferentes métodos de predicción
-   Validación cruzada de resultados
-   Recomendaciones médicas integradas

## 📊 Modelos de Machine Learning

### 🎯  **Estrategia de Screening Dual**

Nuestra propuesta comercial única implementa un sistema de screening de dos capas que maximiza la detección temprana:

1.  **Primera Capa - Screening Masivo**: XGBoost optimizado para alta sensibilidad (78% recall)
2.  **Segunda Capa - Confirmación**: CNN con alta precisión (98.13% accuracy) para casos sospechosos

### 1.  **XGBoost Optimizado (Screening Primario)**

-   **Tipo**: Gradient Boosting para clasificación binaria
-   **Precisión**: 85% en conjunto de prueba
-   **F1-Score**: 0.266 (optimizado para recall médico)
-   **ROC-AUC**: 0.848
-   **Recall**: 78% -  **Detecta 78 de cada 100 casos reales**
-   **Características**: 17 variables médicas y demográficas
-   **Optimización**: 161 trials con Optuna
-   **Ventaja Clínica**: Alto recall minimiza casos perdidos, ideal para screening inicial

### 2.  **Red Neuronal Convolucional (Confirmación)**

-   **Arquitectura**: CNN personalizada desarrollada con Keras y PyTorch
-   **Framework Final**: PyTorch (mejores resultados vs Keras)
-   **Precisión**: 98.13% en imágenes de tomografía
-   **ROC-AUC**: 0.987 (imagen 2)
-   **Input**: Imágenes 224x224 píxeles, escala de grises
-   **Dataset**: 2,501 escáneres cerebrales (1,551 normales, 950 con stroke)
-   **Formato**: TorchScript para optimización en producción
-   **Ventaja Clínica**: Alta precisión confirma casos sospechosos, reduce falsos positivos

### 3.  **Modelos de Investigación**

-   **LightGBM**: Modelo rápido para comparación
-   **Extra Trees**: Ensemble method con interpretabilidad
-   **Linear Discriminant Analysis**: Modelo lineal de referencia
-   **Gradient Boosting**: Implementación sklearn

## 🔄 Flujo de Trabajo

### Predicción Clínica

1.  Usuario ingresa datos médicos del paciente
2.  Validación de rangos médicos (edad 0-120, glucosa 50-500, etc.)
3.  Preprocesamiento con StandardScaler y codificación categórica
4.  Predicción con modelo XGBoost optimizado
5.  Cálculo de nivel de riesgo y recomendaciones
6.  Almacenamiento en base de datos PostgreSQL

### Análisis de Imagen

1.  Upload de tomografía computarizada
2.  Validación de formato, tamaño y calidad
3.  Preprocesamiento de imagen (resize, normalización)
4.  Análisis con red neuronal convolucional
5.  Vinculación con predicción clínica existente
6.  Correlación de resultados multimodales

### Historial y Seguimiento

1.  Visualización de predicciones históricas
2.  Estadísticas agregadas y tendencias
3.  Filtrado por nivel de riesgo y estado de imagen
4.  Exportación de datos para análisis adicional

## 🏥 Impacto Clínico y Propuesta de Valor

### 💡  **Ventaja Comercial: Sistema de Screening Dual**

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

### 📈 Métricas de Rendimiento

#### Modelo XGBoost (Screening)

-   **Sensibilidad (Recall)**: 78% - Detecta 78 de cada 100 casos reales
-   **Especificidad**: 85% - Identifica correctamente casos sanos
-   **F1-Score**: 0.266 - Balanceado para minimizar casos perdidos
-   **ROC-AUC**: 0.848 - Excelente capacidad discriminativa

#### Modelo CNN (Confirmación)

-   **Accuracy**: 98.13% - Precisión excepcional en imágenes
-   **ROC-AUC**: 0.987 - Capacidad discriminativa sobresaliente
-   **Precisión por clase**: 97%+ para stroke y normal
-   **Recall por clase**: 95%+ para ambas categorías

### 🎯 Flujo Clínico Optimizado

1.  **Screening inicial**  con datos básicos del paciente
2.  **Casos de bajo riesgo**  → Seguimiento preventivo estándar
3.  **Casos sospechosos**  → Derivación para tomografía
4.  **Confirmación con CNN**  → Diagnóstico de alta precisión
5.  **Decisión clínica informada**  con doble validación

### Interpretación de Niveles de Riesgo

-   **Bajo (0-30%)**: Mantener controles preventivos regulares
-   **Medio (30-60%)**: Evaluación médica adicional recomendada
-   **Alto (60-90%)**: Consulta neurológica urgente necesaria
-   **Crítico (90-100%)**: Atención médica inmediata requerida

## 👥 Nuestro Equipo

Somos un equipo multidisciplinario de Data Scientists especializados en inteligencia artificial aplicada a la salud:

### 🧑‍💼  [Pepe](https://github.com/peperuizdev)  - Scrum Manager

Especialista en machine learning y arquitectura de software. Responsable de la coordinación del proyecto y la implementación de modelos de clasificación.

### 👩‍💻  [Maryna](https://github.com/MarynaDRST)  - Developer

Desarrolladora de modelos de machine learning y redes neuronales. Especializada en deep learning y procesamiento de imágenes médicas.

### 👨‍🎨  [Jorge](https://github.com/Jorgeluuu)  - Developer

Creador de modelos de machine learning y especialista en optimización de algoritmos. Enfocado en el rendimiento y escalabilidad del sistema.

### 👩‍💼  [Mariela](https://github.com/marie-adi)  - Developer

Diseñadora de experiencia de usuario y desarrolladora frontend. Creadora de la interfaz intuitiva y responsiva de la plataforma.

### 👨‍🔬  [Maximiliano](https://github.com/MaximilianoScarlato)  - Data Scientist

Científico de datos especializado en análisis de modelos de redes neuronales y evaluación de rendimiento de sistemas de ML.

## 🤝 Contribución

Las contribuciones son bienvenidas. Para contribuir:

1.  Fork el proyecto
2.  Crea una rama para tu feature (`git checkout -b feature/AmazingFeature`)
3.  Commit tus cambios (`git commit -m 'Add some AmazingFeature'`)
4.  Push a la rama (`git push origin feature/AmazingFeature`)
5.  Abre un Pull Request

### Estándares de Desarrollo

-   Seguir PEP 8 para código Python
-   Incluir tests para nuevas funcionalidades
-   Documentar funciones y clases
-   Mantener cobertura de tests > 80%

## 📄 Estructura de Testing

### Tests Unitarios

-   **Pipeline de stroke**: Validación de transformaciones y predicciones
-   **Pipeline de imagen**: Procesamiento y validación de imágenes
-   **Servicios**: Lógica de negocio y manejo de errores
-   **Esquemas**: Validación de datos de entrada

### Tests de Integración

-   **API endpoints**: Funcionamiento completo de la API
-   **Base de datos**: Persistencia y recuperación de datos
-   **Flujo completo**: Integración end-to-end
-   **Sistema completo**: Validación del sistema completo

### Cobertura Actual

-   **Líneas cubiertas**: 85%+
-   **Funciones críticas**: 100%
-   **Casos de error**: 90%+
-   **Flujos principales**: 100%

## ⚠️ Consideraciones Médicas

**IMPORTANTE**: Esta herramienta está diseñada únicamente con fines educativos y de investigación. No sustituye el juicio clínico profesional ni debe utilizarse como único criterio para decisiones médicas.

### Limitaciones

-   Los modelos se entrenaron con datos específicos que pueden no representar todas las poblaciones
-   Las predicciones deben interpretarse siempre en conjunto con la evaluación clínica
-   Se requiere validación adicional antes de cualquier uso clínico real
-   Los resultados pueden variar según la calidad de los datos de entrada

### Recomendaciones

-   Siempre consultar con profesionales médicos certificados
-   Utilizar como herramienta de apoyo, no de diagnóstico definitivo
-   Validar resultados con métodos clínicos establecidos
-   Considerar el contexto clínico completo del paciente

## 📈 Rendimiento del Sistema

### Tiempos de Respuesta

-   **Predicción clínica**: < 500ms
-   **Análisis de imagen**: < 2s (CPU) / < 1s (GPU)
-   **Carga de historial**: < 300ms
-   **Inicio de aplicación**: < 10s

### Escalabilidad

-   **Usuarios concurrentes**: 50+ (configuración actual)
-   **Predicciones/hora**: 1,000+
-   **Almacenamiento**: PostgreSQL escalable
-   **Procesamiento**: Optimizado para CPU/GPU

## 🔐 Seguridad y Privacidad

### Protección de Datos

-   Conexión cifrada (HTTPS/TLS)
-   Validación de entrada robusta
-   Sanitización de datos médicos
-   Logs de auditoría

### Cumplimiento

-   Datos almacenados de forma segura
-   No se almacenan imágenes médicas reales
-   Anonimización de datos sensibles
-   Políticas de retención de datos

## 🗺️ Roadmap

### Próximas Funcionalidades

-   [ ]  **Docker**: Containerización completa
-   [ ]  **MLFlow**: Gestión de experimentos y modelos
-   [ ]  **API REST**: Endpoints adicionales
-   [ ]  **Autenticación**: Sistema de usuarios
-   [ ]  **Reportes avanzados**: Dashboard administrativo
-   [ ]  **Exportación**: Múltiples formatos (PDF, Excel)
-   [ ]  **Integración**: Sistemas hospitalarios (HL7 FHIR)
-   [ ]  **Modelos adicionales**: Ensemble methods

### Mejoras Técnicas

-   [ ]  **Performance**: Optimización de modelos
-   [ ]  **Monitoring**: Métricas en tiempo real
-   [ ]  **Backup**: Sistema de respaldos automático
-   [ ]  **CI/CD**: Pipeline de despliegue automático
-   [ ]  **Documentación**: API completa con OpenAPI

## 📞 Soporte

Para preguntas, problemas o sugerencias:

-   **Issues**:  [GitHub Issues](https://github.com/tu-usuario/data_scientist_g3/issues)
-   **Documentación**: Ver carpeta  `docs/`  y comentarios en código
-   **Email**: Contactar al equipo através de GitHub

## 📝 Licencia

Este proyecto está distribuido bajo la Licencia Factoria F5

----------

**Desarrollado con ❤️ por el equipo Data Scientists G3 - Factoría F5**

_Aplicando inteligencia artificial para mejorar la detección temprana de ictus y salvar vidas._