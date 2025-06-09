import pandas as pd
import numpy as np
import optuna
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split, cross_validate, cross_val_score, StratifiedKFold
from sklearn.metrics import f1_score, roc_auc_score, make_scorer
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline
import joblib

df = pd.read_csv("../data/processed/preprocessing.csv")
X = df.drop(columns=["stroke"])
y = df["stroke"]

# 1. Dividir en entrenamiento y test (sin balancear aún)
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)

# 2. Modelo base
base_model = GradientBoostingClassifier(random_state=42)
base_model.fit(X_train, y_train)
y_pred_base = base_model.predict(X_test)
print("F1 Score:", f1_score(y_test, y_pred_base))
print("AUC-ROC:", roc_auc_score(y_test, base_model.predict_proba(X_test)[:, 1]))

# 3. Función objetivo (SMOTE solo en pipeline)
def objective(trial):
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 50, 200),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3),
        "max_depth": trial.suggest_int("max_depth", 3, 10),
        "min_samples_split": trial.suggest_int("min_samples_split", 2, 20),
        "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 20),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "max_features": trial.suggest_categorical("max_features", ["sqrt", "log2", None])
    }

    model = GradientBoostingClassifier(**params, random_state=42)

    pipeline = Pipeline([
        ('smote', SMOTE(random_state=42)),
        ('clf', model)
    ])

    scores = cross_val_score(pipeline, X_train, y_train,
                             scoring=make_scorer(f1_score),
                             cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42))
    return np.mean(scores)

# 4. Optimización
study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=50)

# 5. Entrenar modelo final con SMOTE dentro del pipeline
best_params = study.best_params
final_model = GradientBoostingClassifier(**best_params, random_state=42)
final_pipeline = Pipeline([
    ('smote', SMOTE(random_state=42)),
    ('clf', final_model)
])
final_pipeline.fit(X_train, y_train)

# 6. Evaluación en test set
y_pred_test = final_pipeline.predict(X_test)
y_proba_test = final_pipeline.predict_proba(X_test)[:, 1]
print("Test F1 Score:", f1_score(y_test, y_pred_test))
print("Test ROC-AUC:", roc_auc_score(y_test, y_proba_test))
print("Best params:", best_params)

# 7. Validación cruzada final
scoring = {'f1': 'f1', 'roc_auc': 'roc_auc'}
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = cross_validate(final_pipeline, X_train, y_train, scoring=scoring, cv=cv)
print("\nCross-validation (entrenamiento) metrics:")
print("CV F1 Score:", np.mean(cv_scores['test_f1']))
print("CV ROC-AUC:", np.mean(cv_scores['test_roc_auc']))

# 8. Diferencia para detectar overfitting
f1_diff = np.mean(cv_scores['test_f1']) - f1_score(y_test, y_pred_test)
roc_auc_diff = np.mean(cv_scores['test_roc_auc']) - roc_auc_score(y_test, y_proba_test)
print("\nDiferencias (CV - Test):")
print(f"F1 diff: {f1_diff:.4f}")
print(f"ROC-AUC diff: {roc_auc_diff:.4f}")

if f1_diff > 0.1 or roc_auc_diff > 0.1:
    print("⚠️ Posible overfitting: la métrica en entrenamiento es mucho mejor que en test.")
else:
    print("✅ No parece haber overfitting significativo.")

# 9. Guardar pipeline completo
joblib.dump(final_pipeline, "gradient_boosting_optuna_pipeline.pkl")
