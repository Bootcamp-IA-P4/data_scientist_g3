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
```

## Estructura de Pruebas
- `unit/` - pruebas unitarias
- `integration/` - pruebas de integración
- `fixtures/` - datos de prueba

## Pruebas Críticas
1. Transformación de datos (17 características)
2. Validación de imágenes
3. Flujo de trabajo completo
4. Relación imagen-ictus
5. Predicciones combinadas