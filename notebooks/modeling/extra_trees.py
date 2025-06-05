import pandas as pd
import numpy as np
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_recall_fscore_support
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline

# Cargar los datos
df = pd.read_csv('../../data/processed/preprocessing.csv')

# Separar features y target
X = df.drop('stroke', axis=1)
y = df['stroke']

# Mostrar distribución de clases original
print("\nDistribución de clases original:")
print(y.value_counts(normalize=True))

# Dividir los datos en conjuntos de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Crear pipeline con SMOTE y el modelo
pipeline = Pipeline([
    ('smote', SMOTE(random_state=42, sampling_strategy='auto')),
    ('classifier', ExtraTreesClassifier(
        n_estimators=150,
        max_depth=8,
        min_samples_split=8,
        min_samples_leaf=6,
        max_features='sqrt',
        bootstrap=True,
        random_state=42,
        class_weight='balanced',
        max_samples=0.8
    ))
])

# Realizar validación cruzada
cv_scores = cross_val_score(pipeline, X_train, y_train, cv=5)
print("\nResultados de Validación Cruzada:")
print(f"Precisión media CV: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")

# Entrenar el pipeline final
pipeline.fit(X_train, y_train)

# Evaluar en conjunto de entrenamiento
y_train_pred = pipeline.predict(X_train)
train_accuracy = accuracy_score(y_train, y_train_pred)

# Evaluar en conjunto de prueba
y_test_pred = pipeline.predict(X_test)
test_accuracy = accuracy_score(y_test, y_test_pred)

# Calcular métricas detalladas
precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_test_pred, average='weighted')

# Imprimir resultados
print("\nEvaluación de Overfitting:")
print(f"Precisión en entrenamiento: {train_accuracy:.4f}")
print(f"Precisión en prueba: {test_accuracy:.4f}")
print(f"Diferencia (train-test): {train_accuracy - test_accuracy:.4f}")

print("\nMétricas Detalladas:")
print(f"Precisión ponderada: {precision:.4f}")
print(f"Recall ponderado: {recall:.4f}")
print(f"F1-score ponderado: {f1:.4f}")

print("\nReporte de Clasificación Detallado:")
print(classification_report(y_test, y_test_pred, zero_division=0))

print("\nMatriz de Confusión:")
print(confusion_matrix(y_test, y_test_pred))

# Importancia de características
feature_importance = pd.DataFrame({
    'feature': X.columns,
    'importance': pipeline.named_steps['classifier'].feature_importances_
})
feature_importance = feature_importance.sort_values('importance', ascending=False)
print("\nImportancia de Características:")
print(feature_importance)