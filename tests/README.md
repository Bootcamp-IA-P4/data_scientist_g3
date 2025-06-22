# Pruebas para el Proyecto de Detección de Ictus

## Instalación
```bash
pip install -r requirements-test.txt
```

## Ejecución de Pruebas
```bash
# Todas las pruebas
pytest

# Solo pruebas críticas
pytest -m critical

# Con informe de cobertura
pytest --cov=src --cov-report=html

# Pruebas unitarias
pytest tests/unit/

# Pruebas de integración
pytest tests/integration/

# Pruebas específicas
pytest tests/unit/test_stroke_pipeline.py -v
pytest tests/integration/test_api_endpoints.py -v
```

## Estructura de Pruebas

### 📁 Unit Tests (`tests/unit/`)
- **`test_stroke_pipeline.py`** - Pruebas del pipeline de predicción de ictus
- **`test_image_pipeline.py`** - Pruebas del pipeline de procesamiento de imágenes
- **`test_stroke_service.py`** - Pruebas del servicio de ictus (límites exactos)
- **`test_image_service.py`** - Pruebas del servicio de imágenes (manejo de errores)
- **`test_schemas.py`** - Pruebas de validación de esquemas (rangos médicos)

### 📁 Integration Tests (`tests/integration/`)
- **`test_api_endpoints.py`** - Pruebas básicas de endpoints API
- **`test_api_endpoints_detailed.py`** - Pruebas detalladas de API (flujos completos)
- **`test_database.py`** - Pruebas de base de datos
- **`test_supabase_client.py`** - Pruebas del cliente Supabase
- **`test_complete_workflow.py`** - Pruebas de flujo de trabajo básico
- **`test_system_complete.py`** - Pruebas de sistema completo

### 📁 Fixtures (`tests/fixtures/`)
- **`test_data.json`** - Datos de prueba para pacientes

## Pruebas Críticas Implementadas ✅

### 1. **Servicios Backend**
- ✅ `test_stroke_service.py::_calculate_risk_level()` - Límites exactos
- ✅ `test_image_service.py::process_image()` - Manejo de errores

### 2. **Esquemas**
- ✅ `test_schemas.py::StrokeRequest` - Validación de rangos médicos (edad 0-120, glucosa 50-500)

### 3. **Tests de Integración Esenciales**
- ✅ **CRÍTICO**: `test_api_endpoints_detailed.py::test_post_predict_stroke_complete_flow()` - Flujo completo API → Pipeline → BD
- ✅ **CRÍTICO**: `test_api_endpoints_detailed.py::test_post_predict_image_stroke_link()` - Vinculación stroke-imagen correcta
- ✅ `test_api_endpoints_detailed.py::test_get_predictions_stroke_data_retrieval()` - Recuperación de datos

### 4. **Base de Datos**
- ✅ `test_supabase_client.py::test_save_stroke_prediction_data_integrity()` - Integridad de datos
- ✅ **CRÍTICO**: `test_supabase_client.py::test_get_combined_predictions_with_without_image()` - Manejo casos con/sin imagen
- ✅ `test_supabase_client.py::test_connection_and_constraint_validations()` - Conexión y validaciones

### 5. **Test de Sistema**
- ✅ **CRÍTICO**: `test_system_complete.py::test_complete_system_flow()` - Flujo completo: Predicción → Imagen → Historial
- ✅ `test_system_complete.py::test_cross_system_validation_no_image_without_stroke()` - Validación cross-sistema

## Cobertura de Pruebas

### Funcionalidades Cubiertas:
1. **Transformación de datos** (17 características)
2. **Validación de imágenes** (formatos, tamaños, tipos)
3. **End-to-end workflow** (API completa)
4. **Sistema de vinculación** (stroke-imagen)
5. **Predicciones combinadas** (con y sin imágenes)
6. **Manejo de errores** (datos inválidos, archivos corruptos)
7. **Validaciones médicas** (rangos de edad, glucosa, BMI)
8. **Rendimiento básico** (tiempos de respuesta)
9. **Persistencia de datos** (base de datos)
10. **Operaciones concurrentes** (múltiples usuarios)

### Marcadores de Pruebas:
- `@pytest.mark.critical` - Pruebas críticas del sistema
- `@pytest.mark.unit` - Pruebas unitarias
- `@pytest.mark.integration` - Pruebas de integración

## Configuración

### Archivos de Configuración:
- **`pytest.ini`** - Configuración de pytest
- **`conftest.py`** - Fixtures globales y configuración
- **`requirements-test.txt`** - Dependencias para testing

### Variables de Entorno:
```bash
# Para pruebas con base de datos real 
export TEST_DB_URL="your_test_db_url"
export TEST_DB_KEY="your_test_db_key"
```

## Reportes

### Cobertura de Código:
```bash
# Generar reporte HTML
pytest --cov=src --cov-report=html

# Ver en navegador
open htmlcov/index.html
```

### Reporte de Pruebas:
```bash
# Reporte detallado
pytest -v --tb=short

# Reporte con tiempo
pytest --durations=10
```

## Troubleshooting

### Problemas Comunes:
1. **ModuleNotFoundError**: Ejecutar desde el directorio raíz del proyecto


### Debug:


# Ejecutar con más información
pytest --tb=long -v
```