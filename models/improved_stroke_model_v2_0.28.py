import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold, RepeatedStratifiedKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler, RobustScaler
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score, 
    precision_recall_curve, auc, f1_score, precision_score, 
    recall_score, balanced_accuracy_score, average_precision_score,
    matthews_corrcoef, make_scorer, roc_curve
)
from sklearn.utils.class_weight import compute_class_weight
from imblearn.over_sampling import SMOTE, ADASYN, BorderlineSMOTE
from imblearn.under_sampling import RandomUnderSampler, EditedNearestNeighbours
from imblearn.combine import SMOTEENN, SMOTETomek
from imblearn.ensemble import BalancedBaggingClassifier, BalancedRandomForestClassifier
import lightgbm as lgb
import optuna
from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.optimize import minimize_scalar
import warnings
warnings.filterwarnings('ignore')

class AdvancedStrokeLightGBM:
    def __init__(self, random_state=42):
        self.random_state = random_state
        self.best_params = None
        self.best_model = None
        self.label_encoders = {}
        self.feature_names = None
        self.scaler = None
        self.optimal_threshold = 0.5
        self.class_weights = None
        self.validation_scores = {}
        
    def calculate_class_weights(self, y, method='balanced'):
        """Вычисление весов классов различными методами"""
        if method == 'balanced':
            # Стандартный sklearn подход
            weights = compute_class_weight('balanced', classes=np.unique(y), y=y)
            weight_dict = dict(zip(np.unique(y), weights))
        elif method == 'sqrt':
            # Квадратный корень из обратного соотношения
            counts = Counter(y)
            total = len(y)
            weight_dict = {cls: np.sqrt(total / count) for cls, count in counts.items()}
        elif method == 'log':
            # Логарифмический подход
            counts = Counter(y)
            total = len(y)
            weight_dict = {cls: np.log(total / count) for cls, count in counts.items()}
        else:  # manual
            # Ручная настройка весов
            counts = Counter(y)
            minority_count = min(counts.values())
            weight_dict = {cls: minority_count / count for cls, count in counts.items()}
        
        return weight_dict
    
    def advanced_feature_engineering(self, df):
        """Расширенная инженерия признаков с дополнительными трансформациями"""
        df = df.copy()
        
        # Обработка BMI - более умная импутация
        if df['bmi'].isnull().any():
            for gender in df['gender'].unique():
                for age_group in ['young', 'middle', 'old']:
                    if age_group == 'young':
                        mask = (df['gender'] == gender) & (df['age'] < 40)
                    elif age_group == 'middle':
                        mask = (df['gender'] == gender) & (df['age'].between(40, 65))
                    else:
                        mask = (df['gender'] == gender) & (df['age'] > 65)
                    
                    if mask.sum() > 0:
                        median_bmi = df.loc[mask & df['bmi'].notna(), 'bmi'].median()
                        if pd.notna(median_bmi):
                            df.loc[mask & df['bmi'].isna(), 'bmi'] = median_bmi
            
            df['bmi'] = df['bmi'].fillna(df['bmi'].median())
        
        # Кодирование категориальных переменных
        categorical_features = ['gender', 'ever_married', 'work_type', 'Residence_type', 'smoking_status']
        for feature in categorical_features:
            if feature in df.columns:
                le = LabelEncoder()
                df[feature] = le.fit_transform(df[feature].astype(str))
                self.label_encoders[feature] = le
        
        # === РАСШИРЕННАЯ ИНЖЕНЕРИЯ ПРИЗНАКОВ ===
        
        # 1. Бинаризация числовых признаков
        df['age_high'] = (df['age'] > 65).astype(int)
        df['age_very_high'] = (df['age'] > 75).astype(int)
        df['bmi_high'] = (df['bmi'] > 30).astype(int)
        df['glucose_high'] = (df['avg_glucose_level'] > 140).astype(int)
        df['glucose_very_high'] = (df['avg_glucose_level'] > 180).astype(int)
        
        # 2. Комплексные медицинские индикаторы
        df['metabolic_syndrome'] = (
            (df['bmi_high'] == 1) & 
            (df['glucose_high'] == 1) & 
            (df['hypertension'] == 1)
        ).astype(int)
        
        df['elderly_risk'] = (
            (df['age_high'] == 1) & 
            ((df['hypertension'] == 1) | (df['heart_disease'] == 1))
        ).astype(int)
        
        df['severe_risk'] = (
            (df['age_very_high'] == 1) & 
            (df['glucose_very_high'] == 1)
        ).astype(int)
        
        # 3. Взаимодействия риск-факторов
        df['age_hypertension'] = df['age_high'] * df['hypertension']
        df['age_heart'] = df['age_high'] * df['heart_disease']
        df['bmi_hypertension'] = df['bmi_high'] * df['hypertension']
        df['glucose_hypertension'] = df['glucose_high'] * df['hypertension']
        
        # 4. Комплексные риск-профили
        df['high_risk_profile_1'] = (
            (df['age_high'] == 1) & 
            (df['hypertension'] == 1) & 
            (df['glucose_high'] == 1)
        ).astype(int)
        
        df['high_risk_profile_2'] = (
            (df['heart_disease'] == 1) & 
            (df['bmi_high'] == 1) & 
            (df['age_high'] == 1)
        ).astype(int)
        
        df['high_risk_profile_3'] = (
            (df['metabolic_syndrome'] == 1) & 
            (df['age_high'] == 1)
        ).astype(int)
        
        # 5. Статистические метрики
        df['risk_score'] = (
            df['age_high'] * 3 +
            df['hypertension'] * 2 +
            df['heart_disease'] * 2 +
            df['bmi_high'] * 1 +
            df['glucose_high'] * 2
        )
        
        # 6. Категориальные взаимодействия
        df['smoking_hypertension'] = (df['smoking_status'] >= 2) * df['hypertension']
        df['work_stress'] = (df['work_type'] == 4) * df['hypertension']  # private work
        
        # 7. Медицинские индикаторы
        df['diabetes_risk'] = (
            (df['glucose_high'] == 1) & 
            (df['bmi_high'] == 1)
        ).astype(int)
        
        df['hypertension_risk'] = (
            (df['age_high'] == 1) & 
            (df['bmi_high'] == 1) & 
            (df['hypertension'] == 1)
        ).astype(int)
        
        return df
    
    def advanced_sampling_strategy(self, X, y, strategy='adaptive'):
        """Адаптивная стратегия сэмплирования"""
        try:
            imbalance_ratio = Counter(y)[0] / Counter(y)[1]
            
            if strategy == 'adaptive':
                if imbalance_ratio > 50:
                    # Сильный дисбаланс - агрессивное сэмплирование
                    rus = RandomUnderSampler(sampling_strategy=0.7, random_state=self.random_state)
                    X_rus, y_rus = rus.fit_resample(X, y)
                    smote = SMOTE(random_state=self.random_state, k_neighbors=min(5, Counter(y_rus)[1]-1))
                    X_final, y_final = smote.fit_resample(X_rus, y_rus)
                elif imbalance_ratio > 20:
                    # Средний дисбаланс
                    rus = RandomUnderSampler(sampling_strategy=0.8, random_state=self.random_state)
                    X_rus, y_rus = rus.fit_resample(X, y)
                    smote = BorderlineSMOTE(random_state=self.random_state, k_neighbors=5)
                    X_final, y_final = smote.fit_resample(X_rus, y_rus)
                else:
                    # Небольшой дисбаланс
                    smote = SMOTE(random_state=self.random_state, k_neighbors=5)
                    X_final, y_final = smote.fit_resample(X, y)
                    
            elif strategy == 'hybrid':
                rus = RandomUnderSampler(sampling_strategy=0.8, random_state=self.random_state)
                X_rus, y_rus = rus.fit_resample(X, y)
                smote = SMOTE(random_state=self.random_state, k_neighbors=5)
                X_final, y_final = smote.fit_resample(X_rus, y_rus)
                
            elif strategy == 'tomek':
                smote_tomek = SMOTETomek(random_state=self.random_state)
                X_final, y_final = smote_tomek.fit_resample(X, y)
                
            else:  # smote
                smote = SMOTE(random_state=self.random_state, k_neighbors=5)
                X_final, y_final = smote.fit_resample(X, y)
            
            return X_final, y_final
            
        except Exception as e:
            print(f"Ошибка при сэмплировании: {e}")
            return X, y
    
    def optimize_threshold(self, y_true, y_pred_proba, metric='f1'):
        """Оптимизация порога классификации"""
        
        def threshold_metric(threshold):
            y_pred = (y_pred_proba >= threshold).astype(int)
            
            if metric == 'f1':
                return -f1_score(y_true, y_pred, pos_label=1, zero_division=0)
            elif metric == 'balanced_f1':
                f1_pos = f1_score(y_true, y_pred, pos_label=1, zero_division=0)
                f1_neg = f1_score(y_true, y_pred, pos_label=0, zero_division=0)
                return -(0.6 * f1_pos + 0.4 * f1_neg)  # Больший вес для положительного класса
            elif metric == 'geometric_mean':
                recall_pos = recall_score(y_true, y_pred, pos_label=1, zero_division=0)
                recall_neg = recall_score(y_true, y_pred, pos_label=0, zero_division=0)
                return -np.sqrt(recall_pos * recall_neg)
            elif metric == 'pr_f1':
                # Комбинация precision-recall и f1
                precision = precision_score(y_true, y_pred, pos_label=1, zero_division=0)
                recall = recall_score(y_true, y_pred, pos_label=1, zero_division=0)
                f1 = f1_score(y_true, y_pred, pos_label=1, zero_division=0)
                return -(0.4 * precision + 0.4 * recall + 0.2 * f1)
        
        # Поиск оптимального порога
        result = minimize_scalar(threshold_metric, bounds=(0.1, 0.9), method='bounded')
        optimal_threshold = result.x
        
        # Альтернативный метод через precision-recall кривую
        precisions, recalls, thresholds = precision_recall_curve(y_true, y_pred_proba)
        f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-8)
        best_idx = np.argmax(f1_scores)
        pr_optimal_threshold = thresholds[best_idx] if best_idx < len(thresholds) else 0.5
        
        # Выбираем лучший из двух методов
        y_pred_opt = (y_pred_proba >= optimal_threshold).astype(int)
        y_pred_pr = (y_pred_proba >= pr_optimal_threshold).astype(int)
        
        f1_opt = f1_score(y_true, y_pred_opt, pos_label=1, zero_division=0)
        f1_pr = f1_score(y_true, y_pred_pr, pos_label=1, zero_division=0)
        
        return optimal_threshold if f1_opt > f1_pr else pr_optimal_threshold
    
    def objective_with_validation(self, trial, X_train, y_train, X_val, y_val):
        """Целевая функция с отдельной валидационной выборкой"""
        
        # Параметры модели с более строгими ограничениями
        params = {
            'objective': 'binary',
            'metric': ['auc', 'binary_logloss'],
            'boosting_type': 'gbdt',
            'num_leaves': trial.suggest_int('num_leaves', 5, 30),  # Еще больше уменьшаем листья
            'learning_rate': trial.suggest_float('learning_rate', 0.001, 0.01),  # Уменьшаем скорость обучения
            'feature_fraction': trial.suggest_float('feature_fraction', 0.3, 0.5),  # Уменьшаем долю признаков
            'bagging_fraction': trial.suggest_float('bagging_fraction', 0.3, 0.5),  # Уменьшаем долю выборки
            'bagging_freq': trial.suggest_int('bagging_freq', 5, 7),  # Увеличиваем частоту бэггинга
            'min_child_samples': trial.suggest_int('min_child_samples', 50, 150),  # Увеличиваем минимальное количество сэмплов
            'min_child_weight': trial.suggest_float('min_child_weight', 1, 30),  # Увеличиваем минимальный вес
            'reg_alpha': trial.suggest_float('reg_alpha', 2, 15),  # Увеличиваем L1 регуляризацию
            'reg_lambda': trial.suggest_float('reg_lambda', 2, 15),  # Увеличиваем L2 регуляризацию
            'max_depth': trial.suggest_int('max_depth', 2, 4),  # Уменьшаем глубину
            'min_split_gain': trial.suggest_float('min_split_gain', 0.5, 1.5),  # Увеличиваем минимальный выигрыш
            'verbose': -1,
            'random_state': self.random_state,
            'n_jobs': -1,
        }
        
        # Стратегия сэмплирования
        sampling_strategy = 'adaptive'  # Фиксируем стратегию
        
        # Метод вычисления весов классов
        weight_method = 'balanced'  # Фиксируем метод
        
        # Применяем сэмплирование
        try:
            X_tr_balanced, y_tr_balanced = self.advanced_sampling_strategy(
                X_train, y_train, strategy=sampling_strategy
            )
        except:
            X_tr_balanced, y_tr_balanced = X_train, y_train
        
        # Вычисляем веса классов
        class_weights = self.calculate_class_weights(y_tr_balanced, method=weight_method)
        
        # Применяем веса через scale_pos_weight
        pos_weight = class_weights[1] / class_weights[0] if 0 in class_weights else 1.0
        params['scale_pos_weight'] = pos_weight
        
        # Создаем датасеты для обучения и валидации
        train_data = lgb.Dataset(X_tr_balanced, label=y_tr_balanced)
        valid_data = lgb.Dataset(X_val, label=y_val, reference=train_data)
        
        # Обучение модели с валидационным набором
        model = lgb.train(
            params,
            train_data,
            num_boost_round=100,  # Уменьшаем количество итераций
            valid_sets=[train_data, valid_data],
            valid_names=['train', 'valid'],
            callbacks=[lgb.early_stopping(10), lgb.log_evaluation(0)]
        )
        
        # Предсказания на валидации
        y_pred_proba = model.predict(X_val, num_iteration=model.best_iteration)
        
        # Оптимизация порога с учетом баланса precision/recall
        optimal_threshold = self.optimize_threshold(y_val, y_pred_proba, metric='balanced_f1')
        y_pred = (y_pred_proba >= optimal_threshold).astype(int)
        
        # Комплексная метрика с большим весом для precision
        minority_recall = recall_score(y_val, y_pred, pos_label=1, zero_division=0)
        minority_precision = precision_score(y_val, y_pred, pos_label=1, zero_division=0)
        minority_f1 = f1_score(y_val, y_pred, pos_label=1, zero_division=0)
        pr_auc = average_precision_score(y_val, y_pred_proba)
        roc_auc = roc_auc_score(y_val, y_pred_proba)
        
        # Увеличиваем вес precision в финальной метрике
        score = (0.05 * minority_recall + 0.7 * minority_precision + 
                0.15 * minority_f1 + 0.05 * pr_auc + 0.05 * roc_auc)
        
        return score
    
    def train_with_validation(self, X, y, n_trials=200, test_size=0.15, val_size=0.15):
        """Обучение с отдельной валидационной выборкой"""
        
        print("=== ПРОДВИНУТАЯ МОДЕЛЬ С ВАЛИДАЦИЕЙ ===")
        print(f"Общий размер данных: {X.shape}")
        
        # Вывод исходного распределения
        print("\n=== РАСПРЕДЕЛЕНИЕ КЛАССОВ ===")
        print("Исходные данные:")
        print(f"Всего примеров: {len(y)}")
        print(f"Класс 0 (без инсульта): {sum(y == 0)} ({sum(y == 0)/len(y)*100:.1f}%)")
        print(f"Класс 1 (инсульт): {sum(y == 1)} ({sum(y == 1)/len(y)*100:.1f}%)")
        
        # Сначала разделяем на train+val и test, сохраняя распределение классов
        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y, test_size=test_size, random_state=self.random_state, stratify=y
        )
        
        print("\nПосле разделения на train+val и test:")
        print(f"Train+Val (размер: {len(y_temp)}):")
        print(f"  Класс 0: {sum(y_temp == 0)} ({sum(y_temp == 0)/len(y_temp)*100:.1f}%)")
        print(f"  Класс 1: {sum(y_temp == 1)} ({sum(y_temp == 1)/len(y_temp)*100:.1f}%)")
        print(f"Test (размер: {len(y_test)}):")
        print(f"  Класс 0: {sum(y_test == 0)} ({sum(y_test == 0)/len(y_test)*100:.1f}%)")
        print(f"  Класс 1: {sum(y_test == 1)} ({sum(y_test == 1)/len(y_test)*100:.1f}%)")
        
        # Затем разделяем train+val на train и val
        val_size_adjusted = val_size / (1 - test_size)
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=val_size_adjusted, 
            random_state=self.random_state, stratify=y_temp
        )
        
        print("\nПосле разделения на train и validation:")
        print(f"Train (размер: {len(y_train)}):")
        print(f"  Класс 0: {sum(y_train == 0)} ({sum(y_train == 0)/len(y_train)*100:.1f}%)")
        print(f"  Класс 1: {sum(y_train == 1)} ({sum(y_train == 1)/len(y_train)*100:.1f}%)")
        print(f"Validation (размер: {len(y_val)}):")
        print(f"  Класс 0: {sum(y_val == 0)} ({sum(y_val == 0)/len(y_val)*100:.1f}%)")
        print(f"  Класс 1: {sum(y_val == 1)} ({sum(y_val == 1)/len(y_val)*100:.1f}%)")
        
        # Применяем сэмплирование только к тренировочной выборке
        print("\n=== БАЛАНСИРОВКА ВЫБОРОК ===")
        
        # Балансировка тренировочной выборки
        X_train_balanced, y_train_balanced = self.advanced_sampling_strategy(
            X_train, y_train, strategy='adaptive'
        )
        
        print(f"\nПосле балансировки тренировочной выборки:")
        print(f"Train (размер: {len(y_train_balanced)}):")
        print(f"  Класс 0: {sum(y_train_balanced == 0)} ({sum(y_train_balanced == 0)/len(y_train_balanced)*100:.1f}%)")
        print(f"  Класс 1: {sum(y_train_balanced == 1)} ({sum(y_train_balanced == 1)/len(y_train_balanced)*100:.1f}%)")
        
        # Анализ дисбаланса
        train_ratio = Counter(y_train_balanced)
        print(f"\nTrain class distribution: {train_ratio}")
        print(f"Imbalance ratio: {train_ratio[0]/train_ratio[1]:.1f}:1")
        
        # Оптимизация гиперпараметров
        print(f"\n=== Оптимизация гиперпараметров ({n_trials} trials) ===")
        study = optuna.create_study(
            direction='maximize',
            sampler=optuna.samplers.TPESampler(seed=self.random_state)
        )
        
        study.optimize(
            lambda trial: self.objective_with_validation(trial, X_train_balanced, y_train_balanced, 
                                                       X_val, y_val),
            n_trials=n_trials,
            show_progress_bar=True
        )
        
        self.best_params = study.best_params
        print(f"\nЛучший validation score: {study.best_value:.4f}")
        print(f"Лучшие параметры: {self.best_params}")
        
        # Обучение финальной модели
        print("\n=== Обучение финальной модели ===")
        
        # Объединяем train и validation для финального обучения
        X_train_final = pd.concat([X_train_balanced, X_val])
        y_train_final = pd.concat([y_train_balanced, y_val])
        
        # Применяем лучшую стратегию сэмплирования
        best_sampling = self.best_params.get('sampling_strategy', 'adaptive')
        X_train_final_balanced, y_train_final_balanced = self.advanced_sampling_strategy(
            X_train_final, y_train_final, strategy=best_sampling
        )
        
        print(f"После финальной балансировки: {X_train_final_balanced.shape}")
        print(f"Новое распределение: {Counter(y_train_final_balanced)}")
        
        # Вычисляем финальные веса
        best_weight_method = self.best_params.get('weight_method', 'balanced')
        self.class_weights = self.calculate_class_weights(y_train_final_balanced, method=best_weight_method)
        
        # Параметры модели
        model_params = {k: v for k, v in self.best_params.items() 
                       if k not in ['sampling_strategy', 'weight_method']}
        
        pos_weight = self.class_weights[1] / self.class_weights[0] if 0 in self.class_weights else 1.0
        
        model_params.update({
            'objective': 'binary',
            'metric': ['auc', 'binary_logloss'],
            'scale_pos_weight': pos_weight,
            'verbose': -1,
            'random_state': self.random_state,
            'n_jobs': -1
        })
        
        # Обучение финальной модели
        train_data = lgb.Dataset(X_train_final_balanced, label=y_train_final_balanced)
        self.best_model = lgb.train(
            model_params,
            train_data,
            num_boost_round=300,
            valid_sets=[train_data],
            callbacks=[lgb.early_stopping(30), lgb.log_evaluation(0)]
        )
        
        self.feature_names = X_train.columns.tolist()
        
        # Оптимизация порога на валидационных данных
        y_val_pred_proba = self.best_model.predict(X_val, num_iteration=self.best_model.best_iteration)
        self.optimal_threshold = self.optimize_threshold(y_val, y_val_pred_proba, metric='balanced_f1')
        y_val_pred = (y_val_pred_proba >= self.optimal_threshold).astype(int)
        
        # Вычисление метрик на валидационном наборе
        self.validation_scores = {
            'roc_auc': roc_auc_score(y_val, y_val_pred_proba),
            'pr_auc': average_precision_score(y_val, y_val_pred_proba),
            'f1_score': f1_score(y_val, y_val_pred, pos_label=1, zero_division=0),
            'precision': precision_score(y_val, y_val_pred, pos_label=1, zero_division=0),
            'recall': recall_score(y_val, y_val_pred, pos_label=1, zero_division=0),
            'balanced_accuracy': balanced_accuracy_score(y_val, y_val_pred)
        }
        
        print(f"\n=== ВАЛИДАЦИОННЫЕ МЕТРИКИ ===")
        print(f"Оптимальный порог: {self.optimal_threshold:.4f}")
        print(f"ROC-AUC: {self.validation_scores['roc_auc']:.4f}")
        print(f"PR-AUC: {self.validation_scores['pr_auc']:.4f}")
        print(f"Balanced Accuracy: {self.validation_scores['balanced_accuracy']:.4f}")
        print(f"Matthews Correlation: {matthews_corrcoef(y_val, y_val_pred):.4f}")
        print(f"Geometric Mean: {np.sqrt(recall_score(y_val, y_val_pred, pos_label=1, zero_division=0) * recall_score(y_val, y_val_pred, pos_label=0, zero_division=0)):.4f}")
        print(f"\nМиноритарный класс (инсульт):")
        print(f"  Precision: {self.validation_scores['precision']:.4f}")
        print(f"  Recall: {self.validation_scores['recall']:.4f}")
        print(f"  F1-Score: {self.validation_scores['f1_score']:.4f}")
        
        # Вычисление метрик на тестовом наборе
        y_test_pred_proba = self.best_model.predict(X_test, num_iteration=self.best_model.best_iteration)
        y_test_pred = (y_test_pred_proba >= self.optimal_threshold).astype(int)
        
        # Сравнение с валидационными метриками
        print(f"\n=== СРАВНЕНИЕ VALIDATION vs TEST ===")
        comparison_metrics = ['roc_auc', 'pr_auc', 'f1_score', 'precision', 'recall', 'balanced_accuracy']
        for metric in comparison_metrics:
            if metric in self.validation_scores:
                val_score = self.validation_scores[metric]
                if metric == 'roc_auc':
                    test_score = roc_auc_score(y_test, y_test_pred_proba)
                elif metric == 'pr_auc':
                    test_score = average_precision_score(y_test, y_test_pred_proba)
                elif metric == 'f1_score':
                    test_score = f1_score(y_test, y_test_pred, pos_label=1, zero_division=0)
                elif metric == 'precision':
                    test_score = precision_score(y_test, y_test_pred, pos_label=1, zero_division=0)
                elif metric == 'recall':
                    test_score = recall_score(y_test, y_test_pred, pos_label=1, zero_division=0)
                elif metric == 'balanced_accuracy':
                    test_score = balanced_accuracy_score(y_test, y_test_pred)
                
                print(f"{metric}: Val={val_score:.4f}, Test={test_score:.4f}, Diff={abs(val_score-test_score):.4f}")
        
        # Матрица ошибок и графики
        cm = confusion_matrix(y_test, y_test_pred)
        
        plt.figure(figsize=(20, 10))
        
        # Confusion Matrix
        plt.subplot(2, 4, 1)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=['Без инсульта', 'Инсульт'],
                   yticklabels=['Без инсульта', 'Инсульт'])
        plt.title('Confusion Matrix')
        
        # ROC Curve
        plt.subplot(2, 4, 2)
        fpr, tpr, _ = roc_curve(y_test, y_test_pred_proba)
        plt.plot(fpr, tpr, label=f'ROC AUC = {roc_auc_score(y_test, y_test_pred_proba):.3f}')
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # PR Curve
        plt.subplot(2, 4, 3)
        precisions, recalls, thresholds = precision_recall_curve(y_test, y_test_pred_proba)
        plt.plot(recalls, precisions, label=f'PR AUC = {average_precision_score(y_test, y_test_pred_proba):.3f}')
        plt.axvline(x=recall_score(y_test, y_test_pred, pos_label=1, zero_division=0), color='r', linestyle='--', alpha=0.7)
        plt.axhline(y=precision_score(y_test, y_test_pred, pos_label=1, zero_division=0), color='r', linestyle='--', alpha=0.7)
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curve')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Feature Importance
        plt.subplot(2, 4, 4)
        if self.best_model and self.feature_names:
            feature_imp = pd.DataFrame({
                'feature': self.feature_names,
                'importance': self.best_model.feature_importance(importance_type='gain')
            }).sort_values('importance', ascending=False)
            
            top_features = feature_imp.head(15)
            plt.barh(range(len(top_features)), top_features['importance'])
            plt.yticks(range(len(top_features)), top_features['feature'])
            plt.xlabel('Feature Importance')
            plt.title('Top 15 Feature Importance')
            plt.gca().invert_yaxis()
        
        # Threshold Analysis
        plt.subplot(2, 4, 5)
        thresholds_range = np.linspace(0.1, 0.9, 50)
        precision_scores = []
        recall_scores = []
        f1_scores = []
        
        for thresh in thresholds_range:
            y_pred_thresh = (y_test_pred_proba >= thresh).astype(int)
            precision_scores.append(precision_score(y_test, y_pred_thresh, pos_label=1, zero_division=0))
            recall_scores.append(recall_score(y_test, y_pred_thresh, pos_label=1, zero_division=0))
            f1_scores.append(f1_score(y_test, y_pred_thresh, pos_label=1, zero_division=0))
        
        plt.plot(thresholds_range, precision_scores, label='Precision', alpha=0.7)
        plt.plot(thresholds_range, recall_scores, label='Recall', alpha=0.7)
        plt.plot(thresholds_range, f1_scores, label='F1-Score', alpha=0.7)
        plt.axvline(x=self.optimal_threshold, color='red', linestyle='--', label=f'Optimal: {self.optimal_threshold:.3f}')
        plt.xlabel('Threshold')
        plt.ylabel('Score')
        plt.title('Threshold vs Metrics')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Prediction Distribution
        plt.subplot(2, 4, 6)
        plt.hist(y_test_pred_proba[y_test == 0], bins=30, alpha=0.7, label='No Stroke', density=True)
        plt.hist(y_test_pred_proba[y_test == 1], bins=30, alpha=0.7, label='Stroke', density=True)
        plt.axvline(x=self.optimal_threshold, color='red', linestyle='--', label=f'Threshold: {self.optimal_threshold:.3f}')
        plt.xlabel('Predicted Probability')
        plt.ylabel('Density')
        plt.title('Prediction Distribution')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Class Weight Impact
        plt.subplot(2, 4, 7)
        if self.class_weights:
            classes = list(self.class_weights.keys())
            weights = list(self.class_weights.values())
            colors = ['lightblue', 'lightcoral']
            plt.bar(classes, weights, color=colors)
            plt.xlabel('Class')
            plt.ylabel('Weight')
            plt.title('Class Weights')
            plt.xticks(classes, ['No Stroke', 'Stroke'])
            for i, v in enumerate(weights):
                plt.text(i, v + 0.01, f'{v:.2f}', ha='center')
        
        # Learning Curve
        plt.subplot(2, 4, 8)
        if hasattr(self.best_model, 'evals_result_') and self.best_model.evals_result_:
            eval_results = self.best_model.evals_result_
            if 'training' in eval_results:
                training_loss = eval_results['training']['binary_logloss']
                plt.plot(training_loss, label='Training Loss')
                plt.xlabel('Iteration')
                plt.ylabel('Loss')
                plt.title('Learning Curve')
                plt.legend()
                plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
        
        # Детальный отчет по классификации
        print(f"\n=== ДЕТАЛЬНЫЙ ОТЧЕТ ===")
        print(classification_report(y_test, y_test_pred, 
                                  target_names=['Без инсульта', 'Инсульт']))
        
        # Анализ ошибок
        print(f"\n=== АНАЛИЗ ОШИБОК ===")
        tn, fp, fn, tp = cm.ravel()
        print(f"True Negatives (правильно предсказанные отсутствия инсульта): {tn}")
        print(f"False Positives (ложные тревоги): {fp}")
        print(f"False Negatives (пропущенные инсульты): {fn}")
        print(f"True Positives (правильно предсказанные инсульты): {tp}")
        print(f"\nСтоимость ошибок:")
        print(f"  Пропущенные инсульты (FN): {fn} случаев")
        print(f"  Ложные тревоги (FP): {fp} случаев")
        if fp > 0:
            print(f"  Соотношение FN/FP: {fn/fp:.2f}")
        else:
            print("  FP = 0 (деление на ноль невозможно)")
        
        return X_test, y_test, study
    
    def predict_with_explanation(self, X, explain_top_features=10):
        """Предсказание с объяснением важности признаков"""
        if self.best_model is None:
            raise ValueError("Модель не обучена")
        
        # Предсказания
        y_pred_proba = self.best_model.predict(X, num_iteration=self.best_model.best_iteration)
        y_pred = (y_pred_proba >= self.optimal_threshold).astype(int)
        
        # Важность признаков для каждого предсказания
        if self.feature_names:
            feature_importance = pd.DataFrame({
                'feature': self.feature_names,
                'importance': self.best_model.feature_importance(importance_type='gain')
            }).sort_values('importance', ascending=False)
            
            print(f"Топ-{explain_top_features} наиболее важных признаков:")
            for i, (_, row) in enumerate(feature_importance.head(explain_top_features).iterrows()):
                print(f"{i+1:2d}. {row['feature']:30s}: {row['importance']:8.0f}")
        
        return y_pred, y_pred_proba, feature_importance if self.feature_names else None
    
    def save_model(self, filepath):
        """Сохранение модели"""
        import pickle
        
        model_data = {
            'model': self.best_model,
            'threshold': self.optimal_threshold,
            'feature_names': self.feature_names,
            'label_encoders': self.label_encoders,
            'class_weights': self.class_weights,
            'best_params': self.best_params,
            'validation_scores': self.validation_scores
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
        
        print(f"Модель сохранена: {filepath}")
    
    def load_model(self, filepath):
        """Загрузка модели"""
        import pickle
        
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
        
        self.best_model = model_data['model']
        self.optimal_threshold = model_data['threshold']
        self.feature_names = model_data['feature_names']
        self.label_encoders = model_data['label_encoders']
        self.class_weights = model_data['class_weights']
        self.best_params = model_data['best_params']
        self.validation_scores = model_data['validation_scores']
        
        print(f"Модель загружена: {filepath}")

    def comprehensive_evaluation(self, X_test, y_test):
        """Комплексная оценка модели"""
        if self.best_model is None:
            raise ValueError("Модель не обучена")
        
        # Предсказания
        y_pred_proba = self.best_model.predict(X_test, num_iteration=self.best_model.best_iteration)
        y_pred = (y_pred_proba >= self.optimal_threshold).astype(int)
        
        # Вычисление метрик
        metrics = {
            'roc_auc': roc_auc_score(y_test, y_pred_proba),
            'pr_auc': average_precision_score(y_test, y_pred_proba),
            'f1_score': f1_score(y_test, y_pred, pos_label=1, zero_division=0),
            'precision': precision_score(y_test, y_pred, pos_label=1, zero_division=0),
            'recall': recall_score(y_test, y_pred, pos_label=1, zero_division=0),
            'balanced_accuracy': balanced_accuracy_score(y_test, y_pred),
            'matthews_corrcoef': matthews_corrcoef(y_test, y_pred)
        }
        
        return metrics, y_pred, y_pred_proba

# Функция для запуска улучшенной модели
def run_advanced_model():
    """Запуск продвинутой модели с валидацией"""
    
    # Инициализация
    model = AdvancedStrokeLightGBM(random_state=42)
    
    # Загрузка данных (замените на свой путь)
    df = pd.read_csv('../data/raw/stroke_dataset.csv')
    
    print(f"Исходные данные: {df.shape}")
    print(f"Пропуски в BMI: {df['bmi'].isnull().sum()}")
    print(f"Распределение классов:")
    print(df['stroke'].value_counts())
    
    # Расширенная предобработка
    df_processed = model.advanced_feature_engineering(df)
    
    print(f"\nПосле feature engineering: {df_processed.shape}")
    print(f"Добавлено признаков: {df_processed.shape[1] - df.shape[1]}")
    
    # Подготовка данных
    X = df_processed.drop('stroke', axis=1)
    y = df_processed['stroke']
    
    print(f"\nФинальные данные для обучения:")
    print(f"Признаков: {X.shape[1]}")
    print(f"Образцов: {X.shape[0]}")
    print(f"Без инсульта: {(y==0).sum()}")
    print(f"Инсульт: {(y==1).sum()}")
    print(f"Дисбаланс: {(y==0).sum()/(y==1).sum():.1f}:1")
    
    # Обучение с валидацией
    X_test, y_test, study = model.train_with_validation(
        X, y, n_trials=150, test_size=0.15, val_size=0.15
    )
    
    # Комплексная оценка
    metrics, y_pred, y_pred_proba = model.comprehensive_evaluation(X_test, y_test)
    
    # Демонстрация предсказаний с объяснениями
    print(f"\n=== АНАЛИЗ ВАЖНОСТИ ПРИЗНАКОВ ===")
    _, _, feature_importance = model.predict_with_explanation(X_test.head(1))
    
    # Сохранение модели
    model.save_model('advanced_stroke_model.pkl')
    
    return model, metrics, study

# Дополнительная функция для кросс-валидации
def cross_validate_model(model, X, y, cv_folds=5):
    """Дополнительная кросс-валидация для оценки стабильности"""
    
    skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
    cv_scores = {
        'roc_auc': [],
        'pr_auc': [],
        'f1_score': [],
        'precision': [],
        'recall': [],
        'balanced_accuracy': []
    }
    
    print(f"\n=== КРОСС-ВАЛИДАЦИЯ ({cv_folds} фолдов) ===")
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        print(f"Фолд {fold + 1}/{cv_folds}...")
        
        X_train_cv, X_val_cv = X.iloc[train_idx], X.iloc[val_idx]
        y_train_cv, y_val_cv = y.iloc[train_idx], y.iloc[val_idx]
        
        # Применяем сэмплирование
        X_train_balanced, y_train_balanced = model.advanced_sampling_strategy(
            X_train_cv, y_train_cv, strategy='adaptive'
        )
        
        # Параметры модели
        params = model.best_params.copy() if model.best_params else {
            'objective': 'binary',
            'metric': 'None',
            'boosting_type': 'gbdt',
            'num_leaves': 50,
            'learning_rate': 0.05,
            'feature_fraction': 0.8,
            'bagging_fraction': 0.8,
            'bagging_freq': 5,
            'min_child_samples': 20,
            'verbose': -1,
            'random_state': 42
        }
        
        # Удаляем параметры, которые не нужны для lgb.train
        params = {k: v for k, v in params.items() 
                 if k not in ['sampling_strategy', 'weight_method']}
        
        # Обучение
        train_data = lgb.Dataset(X_train_balanced, label=y_train_balanced)
        fold_model = lgb.train(
            params,
            train_data,
            num_boost_round=500,
            valid_sets=[train_data],
            callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)]
        )
        
        # Предсказания
        y_pred_proba = fold_model.predict(X_val_cv, num_iteration=fold_model.best_iteration)
        
        # Оптимизация порога
        optimal_threshold = model.optimize_threshold(y_val_cv, y_pred_proba, metric='f1')
        y_pred = (y_pred_proba >= optimal_threshold).astype(int)
        
        # Метрики
        cv_scores['roc_auc'].append(roc_auc_score(y_val_cv, y_pred_proba))
        cv_scores['pr_auc'].append(average_precision_score(y_val_cv, y_pred_proba))
        cv_scores['f1_score'].append(f1_score(y_val_cv, y_pred, pos_label=1, zero_division=0))
        cv_scores['precision'].append(precision_score(y_val_cv, y_pred, pos_label=1, zero_division=0))
        cv_scores['recall'].append(recall_score(y_val_cv, y_pred, pos_label=1, zero_division=0))
        cv_scores['balanced_accuracy'].append(balanced_accuracy_score(y_val_cv, y_pred))
    
    # Результаты кросс-валидации
    print(f"\n=== РЕЗУЛЬТАТЫ КРОСС-ВАЛИДАЦИИ ===")
    for metric, scores in cv_scores.items():
        mean_score = np.mean(scores)
        std_score = np.std(scores)
        print(f"{metric:20s}: {mean_score:.4f} ± {std_score:.4f}")
    
    return cv_scores

if __name__ == "__main__":
    # Запуск основной модели
    model, metrics, study = run_advanced_model()
    
    # Дополнительная кросс-валидация
    df = pd.read_csv('../data/raw/stroke_dataset.csv')
    df_processed = model.advanced_feature_engineering(df)
    X = df_processed.drop('stroke', axis=1)
    y = df_processed['stroke']
    cv_results = cross_validate_model(model, X, y, cv_folds=5)
    print(f"Оптимальный порог: {model.optimal_threshold:.4f}")