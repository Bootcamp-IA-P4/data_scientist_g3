import pandas as pd
import numpy as np
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.model_selection import GridSearchCV, train_test_split, StratifiedKFold
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_recall_fscore_support, roc_auc_score
from imblearn.over_sampling import SMOTE, ADASYN
from imblearn.combine import SMOTEENN, SMOTETomek
from imblearn.pipeline import Pipeline
from imblearn.metrics import specificity_score
from sklearn.inspection import permutation_importance  # Añadir para evaluación de importancia más robusta

# Cargar los datos
df = pd.read_csv('../../data/processed/preprocessing.csv')

# Separar features y target
X = df.drop('stroke', axis=1)
y = df['stroke']

# Mostrar distribución de clases original
print("\nDistribución de clases original:")
print(y.value_counts(normalize=True))

# Dividir los datos en conjuntos de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(
    X, y, 
    test_size=0.2,  # Reducido para tener más datos de entrenamiento
    random_state=42, 
    stratify=y
)

# Crear pipelines con diferentes técnicas de balanceo
pipelines = {
    'smote': Pipeline([
        ('sampler', SMOTE(random_state=42, k_neighbors=5)),
        ('classifier', ExtraTreesClassifier(random_state=42, class_weight='balanced', bootstrap=True))  # Añadido bootstrap=True
    ]),
    'adasyn': Pipeline([
        ('sampler', ADASYN(random_state=42, n_neighbors=5)),
        ('classifier', ExtraTreesClassifier(random_state=42, class_weight='balanced', bootstrap=True))  # Añadido bootstrap=True
    ]),
    'smoteenn': Pipeline([
        ('sampler', SMOTEENN(random_state=42, smote=SMOTE(k_neighbors=5))),
        ('classifier', ExtraTreesClassifier(random_state=42, class_weight='balanced', bootstrap=True))  # Añadido bootstrap=True
    ]),
    'smotetomek': Pipeline([
        ('sampler', SMOTETomek(random_state=42, smote=SMOTE(k_neighbors=5))),
        ('classifier', ExtraTreesClassifier(random_state=42, class_weight='balanced', bootstrap=True))  # Añadido bootstrap=True
    ])
}

# Definir grid de hiperparámetros
# Definir grid de hiperparámetros base para el clasificador
base_classifier_params = {
    'classifier__n_estimators': [200, 300],  # Reducir el número de árboles
    'classifier__max_depth': [6, 8],        # Limitar profundidad máxima
    'classifier__min_samples_split': [5, 10, 15],  # Reducir opciones
    'classifier__min_samples_leaf': [6, 8],     # Reducir opciones
    'classifier__max_features': ['sqrt', 0.5],   # Menos opciones de características
    'classifier__max_samples': [0.7],           # Un solo valor
    'classifier__ccp_alpha': [0.001, 0.01]     # Menos opciones de poda
}

# Parámetros específicos para cada técnica de balanceo
sampler_params = {
    'smote': {
        'sampler__k_neighbors': [3, 5],
        'sampler__sampling_strategy': ['auto']
    },
    'adasyn': {
        'sampler__n_neighbors': [3, 5],
        'sampler__sampling_strategy': ['auto']
    },
    'smoteenn': {
        'sampler__smote__k_neighbors': [3, 5],
        'sampler__sampling_strategy': ['auto']
    },
    'smotetomek': {
        'sampler__smote__k_neighbors': [3, 5],
        'sampler__sampling_strategy': ['auto']
    }
}

# Configurar validación cruzada estratificada
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Evaluar cada técnica de balanceo
best_scores = {}
best_models = {}

for name, pipeline in pipelines.items():
    print(f"\nEvaluando {name}...")
    
    # Combinar parámetros base del clasificador con parámetros específicos del sampler
    param_grid = base_classifier_params.copy()
    param_grid.update(sampler_params[name])
    
    # Crear y ejecutar GridSearchCV
    grid_search = GridSearchCV(
        pipeline,
        param_grid,
        cv=skf,
        scoring={
            'f1': 'f1_weighted',
            'auc': 'roc_auc',
            'precision': 'precision_weighted',
            'recall': 'recall_weighted',
            'balanced_accuracy': 'balanced_accuracy'
        },
        refit='f1',
        n_jobs=-1,
        verbose=1,
        return_train_score=True
    )
    
    # Ajustar el modelo
    print("\nBuscando mejores hiperparámetros...")
    grid_search.fit(X_train, y_train)
    
    # Guardar resultados
    best_scores[name] = grid_search.best_score_
    best_models[name] = grid_search.best_estimator_
    
    print(f"Mejor F1-score para {name}: {grid_search.best_score_:.4f}")
    print(f"Mejores parámetros para {name}:")
    print(grid_search.best_params_)

# Seleccionar el mejor método de balanceo
best_method = max(best_scores, key=best_scores.get)
best_model = best_models[best_method]

print(f"\nMejor método de balanceo: {best_method}")

# Evaluar el mejor modelo
y_train_pred = best_model.predict(X_train)
train_accuracy = accuracy_score(y_train, y_train_pred)
train_roc_auc = roc_auc_score(y_train, best_model.predict_proba(X_train)[:, 1])

y_test_pred = best_model.predict(X_test)
y_test_proba = best_model.predict_proba(X_test)[:, 1]
test_accuracy = accuracy_score(y_test, y_test_pred)
test_roc_auc = roc_auc_score(y_test, y_test_proba)
test_specificity = specificity_score(y_test, y_test_pred)

# Calcular métricas detalladas
precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_test_pred, average='weighted')

# Imprimir resultados detallados
print("\nResultados del Mejor Modelo:")
print(f"Método de balanceo: {best_method}")
print(f"F1-score: {best_scores[best_method]:.4f}")

print("\nEvaluación de Overfitting:")
print(f"Precisión en entrenamiento: {train_accuracy:.4f}")
print(f"Precisión en prueba: {test_accuracy:.4f}")
print(f"Diferencia (train-test): {train_accuracy - test_accuracy:.4f}")
print(f"ROC-AUC en entrenamiento: {train_roc_auc:.4f}")
print(f"ROC-AUC en prueba: {test_roc_auc:.4f}")

print("\nMétricas Detalladas del Conjunto de Prueba:")
print(classification_report(y_test, y_test_pred, zero_division=0))

print("\nMatriz de Confusión:")
print(confusion_matrix(y_test, y_test_pred))

# Importancia de características
feature_importance = pd.DataFrame({
    'feature': X.columns,
    'importance': best_model.named_steps['classifier'].feature_importances_
})
feature_importance['importance_percent'] = feature_importance['importance'] * 100
feature_importance = feature_importance.sort_values('importance', ascending=False)
print("\nImportancia de Características (%):")
print(feature_importance.to_string(float_format=lambda x: '{:.2f}'.format(x)))


import matplotlib.pyplot as plt
import seaborn as sns

# Después de obtener los resultados del modelo, agregar las siguientes visualizaciones:

# Configuración de estilo
plt.style.use('default')
sns.set_palette('husl')

# 1. Gráfica de Importancia de Características
plt.figure(figsize=(12, 6))
sns.barplot(x='importance_percent', y='feature', data=feature_importance)
plt.title('Importancia de Características (%)')
plt.xlabel('Importancia (%)')
plt.ylabel('Características')
plt.tight_layout()
plt.savefig('../../reports/figures/feature_importance.png')
plt.close()

# 2. Matriz de Confusión como Heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(confusion_matrix(y_test, y_test_pred), 
            annot=True, 
            fmt='d', 
            cmap='YlOrRd',
            xticklabels=['No Stroke', 'Stroke'],
            yticklabels=['No Stroke', 'Stroke'])
plt.title('Matriz de Confusión')
plt.xlabel('Predicción')
plt.ylabel('Valor Real')
plt.tight_layout()
plt.savefig('../../reports/figures/confusion_matrix.png')
plt.close()

# 3. Comparación de Métricas de Rendimiento
metricas = {
    'Precisión': precision,
    'Recall': recall,
    'F1-Score': f1,
    'Especificidad': test_specificity,
    'ROC-AUC': test_roc_auc
}

plt.figure(figsize=(10, 6))
sns.barplot(x=list(metricas.keys()), y=list(metricas.values()))
plt.title('Métricas de Rendimiento del Modelo')
plt.ylim(0, 1)
plt.xticks(rotation=45)
for i, v in enumerate(metricas.values()):
    plt.text(i, v + 0.01, f'{v:.3f}', ha='center')
plt.tight_layout()
plt.savefig('../../reports/figures/performance_metrics.png')
plt.close()

# 4. Curva ROC
from sklearn.metrics import roc_curve
fpr, tpr, _ = roc_curve(y_test, y_test_proba)

plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {test_roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('Tasa de Falsos Positivos')
plt.ylabel('Tasa de Verdaderos Positivos')
plt.title('Curva ROC')
plt.legend(loc='lower right')
plt.tight_layout()
plt.savefig('../../reports/figures/roc_curve.png')
plt.close()

# 5. Análisis de Overfitting
metricas_overfitting = {
    'Entrenamiento': {
        'Accuracy': train_accuracy,
        'ROC-AUC': train_roc_auc
    },
    'Prueba': {
        'Accuracy': test_accuracy,
        'ROC-AUC': test_roc_auc
    }
}

df_overfitting = pd.DataFrame({
    'Métrica': ['Accuracy', 'ROC-AUC'] * 2,
    'Conjunto': ['Entrenamiento'] * 2 + ['Prueba'] * 2,
    'Valor': [train_accuracy, train_roc_auc, test_accuracy, test_roc_auc]
})

plt.figure(figsize=(10, 6))
sns.barplot(x='Métrica', y='Valor', hue='Conjunto', data=df_overfitting)
plt.title('Comparación entre Entrenamiento y Prueba')
plt.ylim(0, 1)
for i, v in enumerate(df_overfitting['Valor']):
    plt.text(i // 2, v, f'{v:.3f}', ha='center')
plt.tight_layout()
plt.savefig('../../reports/figures/overfitting_analysis.png')
plt.close()

# Añadir curvas de aprendizaje para diagnosticar overfitting
from sklearn.model_selection import learning_curve

plt.figure(figsize=(10, 6))
train_sizes, train_scores, test_scores = learning_curve(
    best_model, X, y, cv=5, scoring='f1_weighted',
    train_sizes=np.linspace(0.1, 1.0, 10), n_jobs=-1
)

train_scores_mean = np.mean(train_scores, axis=1)
train_scores_std = np.std(train_scores, axis=1)
test_scores_mean = np.mean(test_scores, axis=1)
test_scores_std = np.std(test_scores, axis=1)

plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                 train_scores_mean + train_scores_std, alpha=0.1, color="r")
plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                 test_scores_mean + test_scores_std, alpha=0.1, color="g")
plt.plot(train_sizes, train_scores_mean, 'o-', color="r", label="F1 Entrenamiento")
plt.plot(train_sizes, test_scores_mean, 'o-', color="g", label="F1 Validación")
plt.title('Curvas de Aprendizaje')
plt.xlabel('Tamaño del conjunto de entrenamiento')
plt.ylabel('F1 Score')
plt.legend(loc="best")
plt.grid(True)
plt.tight_layout()
plt.savefig('../../reports/figures/learning_curves.png')
plt.close()

print("\nLas gráficas han sido guardadas en el directorio 'reports/figures/'")

# Después de seleccionar el mejor modelo y antes de las visualizaciones
print(f"\nMejor método de balanceo: {best_method}")

# Guardar el mejor modelo en formato pickle
import pickle

# Crear el directorio models/extra_trees si no existe
import os
model_dir = '../../models/extra_trees'
os.makedirs(model_dir, exist_ok=True)

# Guardar el modelo
model_path = os.path.join(model_dir, 'best_model.pkl')
with open(model_path, 'wb') as f:
    pickle.dump(best_model, f)

print(f"\nModelo guardado en: {model_path}")