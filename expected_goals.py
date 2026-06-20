"""
expected_goals.py - Modelo de Expected Goals (xG) estilo Opta
=============================================================

Este módulo implementa un sistema completo de Expected Goals que incluye:
- Logistic Regression como modelo base
- XGBoost como modelo avanzado
- Pipeline de entrenamiento y evaluación
- Predicción y análisis de resultados

Basado en factores clave:
- Posición del tiro (distancia y ángulo al arco)
- Tipo de asistencia previa
- Parte del cuerpo utilizada
- Contexto de juego

Autor: César Adrián Delgado Díaz
Portfolio: https://tu-portfolio.com
GitHub: https://github.com/cesar530
LinkedIn: https://www.linkedin.com/in/cesar-delgado-diaz

MIT License - 2026
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any, Union
import warnings
from pathlib import Path
import joblib

# Machine Learning
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    roc_auc_score, brier_score_loss, log_loss,
    precision_score, recall_score, f1_score,
    confusion_matrix, classification_report,
    roc_curve, precision_recall_curve
)
from sklearn.calibration import calibration_curve

try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    warnings.warn("XGBoost no está instalado. Algunas funcionalidades no estarán disponibles.")

# Importar utilidades locales
from utils import (
    calculate_distance_to_goal,
    calculate_angle_to_goal,
    extract_shot_features,
    create_shot_type_features,
    create_assist_features,
    generate_synthetic_shots
)


class ExpectedGoalsModel:
    """
    Clase principal para el modelo de Expected Goals (xG).
    
    Implementa tanto Logistic Regression como XGBoost para
    predecir la probabilidad de gol de un tiro.
    
    Attributes
    ----------
    model_type : str
        Tipo de modelo ('logistic' o 'xgboost')
    model : object
        Modelo entrenado
    scaler : StandardScaler
        Escalador para normalización de features
    feature_columns : list
        Lista de columnas usadas como features
    is_fitted : bool
        Indica si el modelo ha sido entrenado
    
    Examples
    --------
    >>> model = ExpectedGoalsModel(model_type='xgboost')
    >>> model.fit(X_train, y_train)
    >>> predictions = model.predict_proba(X_test)
    """
    
    # Features numéricas por defecto
    DEFAULT_NUMERIC_FEATURES = [
        'distance_to_goal',
        'angle_to_goal',
        'angle_distance_ratio'
    ]
    
    # Features categóricas por defecto
    DEFAULT_CATEGORICAL_FEATURES = [
        'shot_type',
        'body_part',
        'pass_type'
    ]
    
    def __init__(self, 
                 model_type: str = 'xgboost',
                 random_state: int = 42,
                 **model_params):
        """
        Inicializa el modelo xG.
        
        Parameters
        ----------
        model_type : str
            'logistic' para Logistic Regression o 'xgboost' para XGBoost
        random_state : int
            Semilla para reproducibilidad
        **model_params : dict
            Parámetros adicionales para el modelo
        """
        self.model_type = model_type
        self.random_state = random_state
        self.model_params = model_params
        self.model = None
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.feature_columns = []
        self.numeric_features = []
        self.categorical_features = []
        self.is_fitted = False
        self._training_history = {}
        
        self._initialize_model()
    
    def _initialize_model(self):
        """Inicializa el modelo base según el tipo especificado."""
        if self.model_type == 'logistic':
            default_params = {
                'random_state': self.random_state,
                'max_iter': 1000,
                'solver': 'lbfgs',
                'class_weight': 'balanced'
            }
            default_params.update(self.model_params)
            self.model = LogisticRegression(**default_params)
            
        elif self.model_type == 'xgboost':
            if not XGBOOST_AVAILABLE:
                raise ImportError("XGBoost no está instalado. Instala con: pip install xgboost")
            
            default_params = {
                'random_state': self.random_state,
                'n_estimators': 100,
                'max_depth': 5,
                'learning_rate': 0.1,
                'objective': 'binary:logistic',
                'eval_metric': 'logloss',
                'use_label_encoder': False,
                'verbosity': 0
            }
            default_params.update(self.model_params)
            self.model = xgb.XGBClassifier(**default_params)
        else:
            raise ValueError(f"Tipo de modelo no soportado: {self.model_type}")
    
    def prepare_features(self, 
                         df: pd.DataFrame,
                         numeric_features: Optional[List[str]] = None,
                         categorical_features: Optional[List[str]] = None,
                         fit_encoders: bool = False) -> np.ndarray:
        """
        Prepara las features para entrenamiento o predicción.
        
        Parameters
        ----------
        df : pd.DataFrame
            DataFrame con los datos
        numeric_features : list, optional
            Lista de features numéricas
        categorical_features : list, optional
            Lista de features categóricas
        fit_encoders : bool
            Si True, ajusta los encoders (solo durante entrenamiento)
        
        Returns
        -------
        np.ndarray
            Matriz de features preparadas
        """
        if numeric_features is None:
            numeric_features = [f for f in self.DEFAULT_NUMERIC_FEATURES if f in df.columns]
        if categorical_features is None:
            categorical_features = [f for f in self.DEFAULT_CATEGORICAL_FEATURES if f in df.columns]
        
        if fit_encoders:
            self.numeric_features = numeric_features
            self.categorical_features = categorical_features
        
        # Features numéricas
        X_numeric = df[numeric_features].values
        
        if fit_encoders:
            X_numeric = self.scaler.fit_transform(X_numeric)
        else:
            X_numeric = self.scaler.transform(X_numeric)
        
        # Features categóricas (one-hot encoding)
        X_categorical_list = []
        for col in categorical_features:
            if fit_encoders:
                # Crear one-hot encoding
                dummies = pd.get_dummies(df[col], prefix=col)
                self.label_encoders[col] = list(dummies.columns)
                X_categorical_list.append(dummies.values)
            else:
                # Usar encoding existente
                dummies = pd.get_dummies(df[col], prefix=col)
                # Asegurar que tenga las mismas columnas que durante entrenamiento
                for expected_col in self.label_encoders.get(col, []):
                    if expected_col not in dummies.columns:
                        dummies[expected_col] = 0
                dummies = dummies[self.label_encoders.get(col, dummies.columns)]
                X_categorical_list.append(dummies.values)
        
        # Combinar features
        if X_categorical_list:
            X_categorical = np.hstack(X_categorical_list)
            X = np.hstack([X_numeric, X_categorical])
        else:
            X = X_numeric
        
        # Guardar nombres de features
        if fit_encoders:
            self.feature_columns = list(numeric_features)
            for col, encoded_cols in self.label_encoders.items():
                self.feature_columns.extend(encoded_cols)
        
        return X
    
    def fit(self, 
            df: pd.DataFrame,
            target_col: str = 'goal',
            numeric_features: Optional[List[str]] = None,
            categorical_features: Optional[List[str]] = None,
            validation_split: float = 0.2,
            verbose: bool = True) -> 'ExpectedGoalsModel':
        """
        Entrena el modelo xG.
        
        Parameters
        ----------
        df : pd.DataFrame
            DataFrame con datos de entrenamiento
        target_col : str
            Nombre de la columna objetivo (gol/no gol)
        numeric_features : list, optional
            Features numéricas a usar
        categorical_features : list, optional
            Features categóricas a usar
        validation_split : float
            Proporción de datos para validación
        verbose : bool
            Mostrar información de entrenamiento
        
        Returns
        -------
        self
            El modelo entrenado
        """
        if verbose:
            print(f"Entrenando modelo {self.model_type}...")
            print(f"Datos de entrenamiento: {len(df)} tiros")
        
        # Preparar features
        X = self.prepare_features(
            df, 
            numeric_features, 
            categorical_features, 
            fit_encoders=True
        )
        y = df[target_col].values
        
        # Split para validación
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, 
            test_size=validation_split, 
            random_state=self.random_state,
            stratify=y
        )
        
        if verbose:
            print(f"Tamaño de entrenamiento: {len(X_train)}")
            print(f"Tamaño de validación: {len(X_val)}")
            print(f"Tasa de goles (train): {y_train.mean():.3f}")
            print(f"Features utilizadas: {len(self.feature_columns)}")
        
        # Entrenar modelo
        if self.model_type == 'xgboost':
            self.model.fit(
                X_train, y_train,
                eval_set=[(X_val, y_val)],
                verbose=False
            )
        else:
            self.model.fit(X_train, y_train)
        
        self.is_fitted = True
        
        # Evaluar en validación
        if verbose:
            y_pred_proba = self.model.predict_proba(X_val)[:, 1]
            auc = roc_auc_score(y_val, y_pred_proba)
            brier = brier_score_loss(y_val, y_pred_proba)
            print(f"\n=== Métricas de Validación ===")
            print(f"ROC-AUC: {auc:.4f}")
            print(f"Brier Score: {brier:.4f}")
        
        return self
    
    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        """
        Predice la probabilidad de gol (xG) para cada tiro.
        
        Parameters
        ----------
        df : pd.DataFrame
            DataFrame con datos de tiros
        
        Returns
        -------
        np.ndarray
            Array con probabilidades de gol
        """
        if not self.is_fitted:
            raise ValueError("El modelo debe ser entrenado antes de predecir.")
        
        X = self.prepare_features(df, fit_encoders=False)
        return self.model.predict_proba(X)[:, 1]
    
    def predict(self, df: pd.DataFrame, threshold: float = 0.5) -> np.ndarray:
        """
        Predice si cada tiro es gol o no (clasificación binaria).
        
        Parameters
        ----------
        df : pd.DataFrame
            DataFrame con datos de tiros
        threshold : float
            Umbral para clasificación
        
        Returns
        -------
        np.ndarray
            Array con predicciones (0 o 1)
        """
        proba = self.predict_proba(df)
        return (proba >= threshold).astype(int)
    
    def evaluate(self, df: pd.DataFrame, target_col: str = 'goal') -> Dict[str, float]:
        """
        Evalúa el modelo con múltiples métricas.
        
        Parameters
        ----------
        df : pd.DataFrame
            DataFrame con datos de evaluación
        target_col : str
            Nombre de la columna objetivo
        
        Returns
        -------
        dict
            Diccionario con métricas de evaluación
        """
        y_true = df[target_col].values
        y_pred_proba = self.predict_proba(df)
        y_pred = self.predict(df)
        
        metrics = {
            'roc_auc': roc_auc_score(y_true, y_pred_proba),
            'brier_score': brier_score_loss(y_true, y_pred_proba),
            'log_loss': log_loss(y_true, y_pred_proba),
            'precision': precision_score(y_true, y_pred, zero_division=0),
            'recall': recall_score(y_true, y_pred, zero_division=0),
            'f1_score': f1_score(y_true, y_pred, zero_division=0)
        }
        
        return metrics
    
    def cross_validate(self, 
                       df: pd.DataFrame,
                       target_col: str = 'goal',
                       n_folds: int = 5,
                       verbose: bool = True) -> Dict[str, List[float]]:
        """
        Realiza validación cruzada del modelo.
        
        Parameters
        ----------
        df : pd.DataFrame
            DataFrame con datos
        target_col : str
            Columna objetivo
        n_folds : int
            Número de folds
        verbose : bool
            Mostrar resultados
        
        Returns
        -------
        dict
            Métricas por fold
        """
        X = self.prepare_features(df, fit_encoders=True)
        y = df[target_col].values
        
        cv = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=self.random_state)
        
        results = {
            'roc_auc': [],
            'brier_score': [],
            'log_loss': []
        }
        
        for fold, (train_idx, val_idx) in enumerate(cv.split(X, y)):
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]
            
            # Reinicializar modelo
            self._initialize_model()
            
            if self.model_type == 'xgboost':
                self.model.fit(X_train, y_train, verbose=False)
            else:
                self.model.fit(X_train, y_train)
            
            y_pred_proba = self.model.predict_proba(X_val)[:, 1]
            
            results['roc_auc'].append(roc_auc_score(y_val, y_pred_proba))
            results['brier_score'].append(brier_score_loss(y_val, y_pred_proba))
            results['log_loss'].append(log_loss(y_val, y_pred_proba))
            
            if verbose:
                print(f"Fold {fold + 1}: AUC={results['roc_auc'][-1]:.4f}, "
                      f"Brier={results['brier_score'][-1]:.4f}")
        
        if verbose:
            print(f"\n=== Resultados CV ({n_folds} folds) ===")
            for metric, values in results.items():
                print(f"{metric}: {np.mean(values):.4f} (+/- {np.std(values):.4f})")
        
        return results
    
    def get_feature_importance(self) -> pd.DataFrame:
        """
        Obtiene la importancia de cada feature.
        
        Returns
        -------
        pd.DataFrame
            DataFrame con features ordenadas por importancia
        """
        if not self.is_fitted:
            raise ValueError("El modelo debe ser entrenado primero.")
        
        if self.model_type == 'xgboost':
            importance = self.model.feature_importances_
        elif self.model_type == 'logistic':
            importance = np.abs(self.model.coef_[0])
        else:
            raise ValueError(f"Importancia no disponible para {self.model_type}")
        
        df_importance = pd.DataFrame({
            'feature': self.feature_columns,
            'importance': importance
        }).sort_values('importance', ascending=False)
        
        return df_importance
    
    def save_model(self, filepath: Union[str, Path]):
        """
        Guarda el modelo entrenado.
        
        Parameters
        ----------
        filepath : str or Path
            Ruta donde guardar el modelo
        """
        if not self.is_fitted:
            raise ValueError("El modelo debe ser entrenado antes de guardar.")
        
        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'label_encoders': self.label_encoders,
            'feature_columns': self.feature_columns,
            'numeric_features': self.numeric_features,
            'categorical_features': self.categorical_features,
            'model_type': self.model_type,
            'random_state': self.random_state
        }
        
        joblib.dump(model_data, filepath)
        print(f"Modelo guardado en: {filepath}")
    
    @classmethod
    def load_model(cls, filepath: Union[str, Path]) -> 'ExpectedGoalsModel':
        """
        Carga un modelo guardado.
        
        Parameters
        ----------
        filepath : str or Path
            Ruta del modelo guardado
        
        Returns
        -------
        ExpectedGoalsModel
            Modelo cargado
        """
        model_data = joblib.load(filepath)
        
        instance = cls(
            model_type=model_data['model_type'],
            random_state=model_data['random_state']
        )
        instance.model = model_data['model']
        instance.scaler = model_data['scaler']
        instance.label_encoders = model_data['label_encoders']
        instance.feature_columns = model_data['feature_columns']
        instance.numeric_features = model_data['numeric_features']
        instance.categorical_features = model_data['categorical_features']
        instance.is_fitted = True
        
        return instance


class XGAnalyzer:
    """
    Clase para análisis y visualización de resultados xG.
    
    Proporciona métodos para:
    - Análisis de calibración
    - Comparación entre jugadores/equipos
    - Métricas agregadas (xG por partido, etc.)
    """
    
    def __init__(self, model: ExpectedGoalsModel):
        """
        Parameters
        ----------
        model : ExpectedGoalsModel
            Modelo xG entrenado
        """
        self.model = model
    
    def player_xg_summary(self, 
                          df: pd.DataFrame,
                          player_col: str = 'player',
                          target_col: str = 'goal') -> pd.DataFrame:
        """
        Calcula resumen de xG por jugador.
        
        Parameters
        ----------
        df : pd.DataFrame
            DataFrame con tiros y columna de jugador
        player_col : str
            Nombre de la columna de jugador
        target_col : str
            Columna de goles
        
        Returns
        -------
        pd.DataFrame
            Resumen por jugador
        """
        df = df.copy()
        df['xG'] = self.model.predict_proba(df)
        
        summary = df.groupby(player_col).agg({
            'xG': ['sum', 'mean', 'count'],
            target_col: 'sum'
        }).round(3)
        
        summary.columns = ['xG_total', 'xG_per_shot', 'shots', 'goals']
        summary['goals_minus_xG'] = summary['goals'] - summary['xG_total']
        summary['conversion_rate'] = summary['goals'] / summary['shots']
        
        return summary.sort_values('xG_total', ascending=False)
    
    def calibration_analysis(self,
                              df: pd.DataFrame,
                              target_col: str = 'goal',
                              n_bins: int = 10) -> Tuple[np.ndarray, np.ndarray]:
        """
        Analiza la calibración del modelo.
        
        Un modelo bien calibrado debería tener predicciones que
        coinciden con las frecuencias reales de goles.
        
        Parameters
        ----------
        df : pd.DataFrame
            DataFrame con datos de evaluación
        target_col : str
            Columna objetivo
        n_bins : int
            Número de bins para el análisis
        
        Returns
        -------
        tuple
            (fraction_of_positives, mean_predicted_value)
        """
        y_true = df[target_col].values
        y_pred = self.model.predict_proba(df)
        
        fraction_of_positives, mean_predicted_value = calibration_curve(
            y_true, y_pred, n_bins=n_bins, strategy='uniform'
        )
        
        return fraction_of_positives, mean_predicted_value
    
    def calculate_match_xg(self,
                           df: pd.DataFrame,
                           match_col: str = 'match_id',
                           team_col: str = 'team') -> pd.DataFrame:
        """
        Calcula xG total por partido y equipo.
        
        Parameters
        ----------
        df : pd.DataFrame
            DataFrame con tiros
        match_col : str
            Columna de identificador de partido
        team_col : str
            Columna de equipo
        
        Returns
        -------
        pd.DataFrame
            xG por partido y equipo
        """
        df = df.copy()
        df['xG'] = self.model.predict_proba(df)
        
        match_xg = df.groupby([match_col, team_col]).agg({
            'xG': 'sum',
            'goal': 'sum'
        }).reset_index()
        
        match_xg.columns = [match_col, team_col, 'xG', 'goals']
        
        return match_xg


def train_xg_pipeline(data_source: str = 'synthetic',
                      model_type: str = 'xgboost',
                      n_samples: int = 2000,
                      save_path: Optional[str] = None,
                      verbose: bool = True) -> Tuple[ExpectedGoalsModel, pd.DataFrame]:
    """
    Pipeline completo para entrenar un modelo xG.
    
    Parameters
    ----------
    data_source : str
        'synthetic' para datos sintéticos o ruta a archivo CSV
    model_type : str
        Tipo de modelo ('logistic' o 'xgboost')
    n_samples : int
        Número de muestras si se usan datos sintéticos
    save_path : str, optional
        Ruta para guardar el modelo entrenado
    verbose : bool
        Mostrar información
    
    Returns
    -------
    tuple
        (modelo_entrenado, dataframe_con_predicciones)
    """
    if verbose:
        print("="*60)
        print("PIPELINE DE ENTRENAMIENTO - Expected Goals (xG)")
        print("="*60)
    
    # 1. Cargar datos
    if data_source == 'synthetic':
        if verbose:
            print(f"\n[1/4] Generando {n_samples} tiros sintéticos...")
        df = generate_synthetic_shots(n_samples)
    else:
        if verbose:
            print(f"\n[1/4] Cargando datos de: {data_source}")
        df = pd.read_csv(data_source)
        df = extract_shot_features(df)
    
    # 2. Preparar features adicionales
    if verbose:
        print("[2/4] Preparando features...")
    df = create_shot_type_features(df)
    df = create_assist_features(df)
    
    # 3. Entrenar modelo
    if verbose:
        print(f"[3/4] Entrenando modelo {model_type}...")
    
    model = ExpectedGoalsModel(model_type=model_type)
    model.fit(df, verbose=verbose)
    
    # 4. Agregar predicciones al DataFrame
    if verbose:
        print("[4/4] Generando predicciones...")
    df['xG_predicted'] = model.predict_proba(df)
    
    # Guardar modelo si se especifica ruta
    if save_path:
        model.save_model(save_path)
    
    if verbose:
        print("\n" + "="*60)
        print("ENTRENAMIENTO COMPLETADO")
        print("="*60)
        
        # Mostrar importancia de features
        importance = model.get_feature_importance()
        print("\nTop 10 Features más importantes:")
        print(importance.head(10).to_string(index=False))
    
    return model, df


if __name__ == "__main__":
    # Demo del pipeline completo
    print("\n" + "="*70)
    print("  DEMO: Expected Goals (xG) Model - Estilo Opta")
    print("="*70)
    
    # Entrenar con datos sintéticos
    model, df = train_xg_pipeline(
        data_source='synthetic',
        model_type='xgboost',
        n_samples=2000,
        verbose=True
    )
    
    # Análisis adicional
    print("\n" + "-"*50)
    print("ANÁLISIS DE RESULTADOS")
    print("-"*50)
    
    # Métricas de evaluación
    metrics = model.evaluate(df)
    print("\nMétricas finales:")
    for metric, value in metrics.items():
        print(f"  {metric}: {value:.4f}")
    
    # Comparar xG predicho vs real
    print(f"\nxG total predicho: {df['xG_predicted'].sum():.2f}")
    print(f"Goles reales: {df['goal'].sum()}")
    print(f"Diferencia: {df['goal'].sum() - df['xG_predicted'].sum():.2f}")
    
    print("\n" + "="*70)
    print("  Demo completado exitosamente!")
    print("="*70)
