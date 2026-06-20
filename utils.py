"""
utils.py - Utilidades para el cálculo de Expected Goals (xG)
============================================================

Este módulo contiene funciones auxiliares para:
- Cálculo de distancia y ángulo al arco
- Feature engineering para modelos xG
- Visualización de mapas de tiro
- Procesamiento de datos de StatsBomb

Autor: César Adrián Delgado Díaz
GitHub: https://github.com/cesar530
LinkedIn: https://www.linkedin.com/in/cesar-delgado-diaz

MIT License - 2026
"""

import numpy as np
import pandas as pd
from typing import Tuple, List, Optional, Union
import warnings

# Constantes del campo de fútbol (dimensiones StatsBomb: 120x80 yards)
PITCH_LENGTH = 120  # yards
PITCH_WIDTH = 80    # yards
GOAL_WIDTH = 8      # yards (aproximado ~7.32m)
GOAL_CENTER_X = 120  # posición X del arco
GOAL_CENTER_Y = 40   # centro del arco en Y
GOAL_POST_LEFT = 36  # Y del poste izquierdo
GOAL_POST_RIGHT = 44  # Y del poste derecho


def calculate_distance_to_goal(x: Union[float, np.ndarray], 
                                y: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    """
    Calcula la distancia desde la posición del tiro hasta el centro del arco.
    
    Parameters
    ----------
    x : float or array-like
        Coordenada X de la posición del tiro (0-120 yards)
    y : float or array-like
        Coordenada Y de la posición del tiro (0-80 yards)
    
    Returns
    -------
    float or array-like
        Distancia en yards al centro del arco
    
    Examples
    --------
    >>> calculate_distance_to_goal(108, 40)  # Dentro del área, centro
    12.0
    >>> calculate_distance_to_goal(100, 40)  # Punto penal aproximado
    20.0
    """
    return np.sqrt((GOAL_CENTER_X - x)**2 + (GOAL_CENTER_Y - y)**2)


def calculate_angle_to_goal(x: Union[float, np.ndarray], 
                            y: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    """
    Calcula el ángulo visible del arco desde la posición del tiro.
    
    El ángulo se calcula usando la fórmula del ángulo entre dos vectores
    desde la posición del tiro hacia cada poste.
    
    Parameters
    ----------
    x : float or array-like
        Coordenada X de la posición del tiro
    y : float or array-like
        Coordenada Y de la posición del tiro
    
    Returns
    -------
    float or array-like
        Ángulo en grados del arco visible
    
    Notes
    -----
    Un ángulo mayor significa que el tirador ve más del arco,
    lo cual típicamente resulta en mayor probabilidad de gol.
    """
    # Vectores a cada poste
    vec_left = np.array([GOAL_CENTER_X - x, GOAL_POST_LEFT - y])
    vec_right = np.array([GOAL_CENTER_X - x, GOAL_POST_RIGHT - y])
    
    # Si son arrays, calcular elemento por elemento
    if isinstance(x, np.ndarray):
        angles = []
        for i in range(len(x)):
            v1 = np.array([GOAL_CENTER_X - x[i], GOAL_POST_LEFT - y[i]])
            v2 = np.array([GOAL_CENTER_X - x[i], GOAL_POST_RIGHT - y[i]])
            cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-10)
            cos_angle = np.clip(cos_angle, -1, 1)
            angles.append(np.degrees(np.arccos(cos_angle)))
        return np.array(angles)
    else:
        cos_angle = np.dot(vec_left, vec_right) / (np.linalg.norm(vec_left) * np.linalg.norm(vec_right) + 1e-10)
        cos_angle = np.clip(cos_angle, -1, 1)
        return np.degrees(np.arccos(cos_angle))


def calculate_gk_angle(x: float, y: float, 
                       gk_x: Optional[float] = None, 
                       gk_y: Optional[float] = None) -> float:
    """
    Calcula el ángulo entre el tirador y el portero.
    
    Parameters
    ----------
    x, y : float
        Posición del tirador
    gk_x, gk_y : float, optional
        Posición del portero. Si no se proporciona, 
        se asume portero en el centro del arco.
    
    Returns
    -------
    float
        Ángulo en grados
    """
    if gk_x is None:
        gk_x = GOAL_CENTER_X - 1
    if gk_y is None:
        gk_y = GOAL_CENTER_Y
    
    dx = gk_x - x
    dy = gk_y - y
    return np.degrees(np.arctan2(dy, dx))


def extract_shot_features(df: pd.DataFrame, 
                          x_col: str = 'x', 
                          y_col: str = 'y') -> pd.DataFrame:
    """
    Extrae features para el modelo xG a partir de un DataFrame de tiros.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame con datos de tiros
    x_col : str
        Nombre de la columna con coordenada X
    y_col : str
        Nombre de la columna con coordenada Y
    
    Returns
    -------
    pd.DataFrame
        DataFrame con features adicionales calculadas
    """
    df = df.copy()
    
    # Features geométricas básicas
    df['distance_to_goal'] = calculate_distance_to_goal(
        df[x_col].values, df[y_col].values
    )
    df['angle_to_goal'] = calculate_angle_to_goal(
        df[x_col].values, df[y_col].values
    )
    
    # Indicador de zona central
    df['is_central'] = (df[y_col] >= 30) & (df[y_col] <= 50)
    
    # Indicador dentro del área (aproximadamente 18 yards del arco)
    df['in_box'] = df['distance_to_goal'] <= 18
    
    # Feature de "ángulo ajustado por distancia"
    df['angle_distance_ratio'] = df['angle_to_goal'] / (df['distance_to_goal'] + 1)
    
    # Zona del campo (dividido en sectores)
    df['zone'] = pd.cut(
        df[y_col], 
        bins=[0, 25, 40, 55, 80],
        labels=['far_left', 'left', 'right', 'far_right']
    )
    
    return df


def encode_categorical_features(df: pd.DataFrame, 
                                 categorical_cols: List[str]) -> pd.DataFrame:
    """
    Codifica variables categóricas usando one-hot encoding.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame con features
    categorical_cols : List[str]
        Lista de columnas categóricas a codificar
    
    Returns
    -------
    pd.DataFrame
        DataFrame con variables dummy
    """
    return pd.get_dummies(df, columns=categorical_cols, drop_first=True)


def create_shot_type_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Crea features basadas en el tipo de tiro y situación de juego.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame que debe contener columnas como:
        - shot_type: tipo de tiro (Open Play, Free Kick, Corner, Penalty)
        - body_part: parte del cuerpo (Left Foot, Right Foot, Head)
        - technique: técnica del tiro
    
    Returns
    -------
    pd.DataFrame
        DataFrame con features adicionales
    """
    df = df.copy()
    
    # Features de tipo de jugada
    if 'shot_type' in df.columns:
        df['is_penalty'] = (df['shot_type'] == 'Penalty').astype(int)
        df['is_free_kick'] = (df['shot_type'] == 'Free Kick').astype(int)
        df['is_open_play'] = (df['shot_type'] == 'Open Play').astype(int)
        df['is_corner'] = (df['shot_type'] == 'Corner').astype(int)
    
    # Features de parte del cuerpo
    if 'body_part' in df.columns:
        df['is_header'] = (df['body_part'] == 'Head').astype(int)
        df['is_right_foot'] = (df['body_part'] == 'Right Foot').astype(int)
        df['is_left_foot'] = (df['body_part'] == 'Left Foot').astype(int)
    
    return df


def create_assist_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Crea features relacionadas con la asistencia previa al tiro.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame que puede contener columnas como:
        - pass_type: tipo de pase previo
        - through_ball: si fue pase filtrado
        - cross: si fue centro
    
    Returns
    -------
    pd.DataFrame
        DataFrame con features de asistencia
    """
    df = df.copy()
    
    if 'pass_type' in df.columns:
        df['assisted_throughball'] = (df['pass_type'] == 'Through Ball').astype(int)
        df['assisted_cross'] = (df['pass_type'] == 'Cross').astype(int)
        df['assisted_cutback'] = (df['pass_type'] == 'Cut Back').astype(int)
        df['assisted_pullback'] = (df['pass_type'] == 'Pull Back').astype(int)
    
    # Si hay información de si el tiro fue asistido o no
    if 'assisted' in df.columns:
        df['is_assisted'] = df['assisted'].astype(int)
    
    return df


def normalize_coordinates(x: Union[float, np.ndarray], 
                          y: Union[float, np.ndarray],
                          source_dims: Tuple[float, float] = (120, 80),
                          target_dims: Tuple[float, float] = (120, 80)) -> Tuple:
    """
    Normaliza coordenadas de un sistema a otro.
    
    Útil cuando los datos vienen de diferentes fuentes con 
    diferentes dimensiones de campo.
    
    Parameters
    ----------
    x, y : float or array-like
        Coordenadas originales
    source_dims : tuple
        (largo, ancho) del sistema original
    target_dims : tuple
        (largo, ancho) del sistema destino
    
    Returns
    -------
    tuple
        (x_normalized, y_normalized)
    """
    x_norm = (x / source_dims[0]) * target_dims[0]
    y_norm = (y / source_dims[1]) * target_dims[1]
    return x_norm, y_norm


def get_xg_baseline_by_zone(distance: float, angle: float) -> float:
    """
    Obtiene un xG baseline basado en reglas simples por zona.
    
    Útil para comparación con el modelo entrenado.
    
    Parameters
    ----------
    distance : float
        Distancia al arco en yards
    angle : float
        Ángulo visible del arco en grados
    
    Returns
    -------
    float
        xG baseline (0-1)
    """
    # Reglas heurísticas basadas en estadísticas históricas
    if distance <= 6:  # Muy cerca
        base_xg = 0.40
    elif distance <= 12:  # Dentro del área chica
        base_xg = 0.25
    elif distance <= 18:  # Dentro del área grande
        base_xg = 0.12
    elif distance <= 25:  # Borde del área
        base_xg = 0.06
    else:  # Fuera del área
        base_xg = 0.03
    
    # Ajuste por ángulo
    if angle > 30:
        angle_multiplier = 1.2
    elif angle > 20:
        angle_multiplier = 1.0
    elif angle > 10:
        angle_multiplier = 0.8
    else:
        angle_multiplier = 0.5
    
    return min(base_xg * angle_multiplier, 0.95)


def validate_coordinates(x: Union[float, np.ndarray], 
                         y: Union[float, np.ndarray]) -> bool:
    """
    Valida que las coordenadas estén dentro del campo.
    
    Parameters
    ----------
    x, y : float or array-like
        Coordenadas a validar
    
    Returns
    -------
    bool
        True si todas las coordenadas son válidas
    """
    x_valid = np.all((x >= 0) & (x <= PITCH_LENGTH))
    y_valid = np.all((y >= 0) & (y <= PITCH_WIDTH))
    return x_valid and y_valid


def calculate_expected_goals_simple(df: pd.DataFrame,
                                    x_col: str = 'x',
                                    y_col: str = 'y') -> pd.Series:
    """
    Calcula xG usando el modelo baseline (sin ML).
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame con coordenadas de tiros
    x_col, y_col : str
        Nombres de columnas de coordenadas
    
    Returns
    -------
    pd.Series
        Serie con valores xG para cada tiro
    """
    distances = calculate_distance_to_goal(df[x_col].values, df[y_col].values)
    angles = calculate_angle_to_goal(df[x_col].values, df[y_col].values)
    
    xg_values = [get_xg_baseline_by_zone(d, a) for d, a in zip(distances, angles)]
    return pd.Series(xg_values, index=df.index)


# Función para generar datos sintéticos de prueba
def generate_synthetic_shots(n_shots: int = 1000, 
                             random_state: int = 42) -> pd.DataFrame:
    """
    Genera datos sintéticos de tiros para pruebas y demos.
    
    Parameters
    ----------
    n_shots : int
        Número de tiros a generar
    random_state : int
        Semilla para reproducibilidad
    
    Returns
    -------
    pd.DataFrame
        DataFrame con tiros sintéticos y resultado (gol/no gol)
    """
    np.random.seed(random_state)
    
    # Generar posiciones (más concentradas en el área)
    x = np.random.beta(8, 2, n_shots) * 40 + 80  # Concentrado entre 80-120
    y = np.random.normal(40, 15, n_shots)  # Centrado en el medio
    y = np.clip(y, 5, 75)  # Mantener dentro del campo
    
    # Tipos de tiro
    shot_types = np.random.choice(
        ['Open Play', 'Free Kick', 'Corner', 'Penalty'],
        n_shots,
        p=[0.75, 0.10, 0.08, 0.07]
    )
    
    # Parte del cuerpo
    body_parts = np.random.choice(
        ['Right Foot', 'Left Foot', 'Head'],
        n_shots,
        p=[0.50, 0.30, 0.20]
    )
    
    # Tipo de pase previo
    pass_types = np.random.choice(
        ['Ground Pass', 'Through Ball', 'Cross', 'Cut Back', 'None'],
        n_shots,
        p=[0.40, 0.15, 0.25, 0.10, 0.10]
    )
    
    df = pd.DataFrame({
        'x': x,
        'y': y,
        'shot_type': shot_types,
        'body_part': body_parts,
        'pass_type': pass_types
    })
    
    # Calcular features
    df = extract_shot_features(df)
    
    # Generar resultado basado en features (probabilístico)
    # Penales tienen alta probabilidad
    base_prob = np.where(
        df['shot_type'] == 'Penalty',
        0.76,  # Tasa histórica de conversión de penales
        0.1 / (1 + df['distance_to_goal'] / 15) * (df['angle_to_goal'] / 30)
    )
    
    # Ajuste por cabezazo (generalmente menor conversión)
    base_prob = np.where(df['body_part'] == 'Head', base_prob * 0.8, base_prob)
    
    # Ajuste por pase filtrado (mejor oportunidad)
    base_prob = np.where(df['pass_type'] == 'Through Ball', base_prob * 1.3, base_prob)
    
    # Limitar probabilidades
    base_prob = np.clip(base_prob, 0.01, 0.95)
    
    # Generar resultado
    df['goal'] = (np.random.random(n_shots) < base_prob).astype(int)
    df['true_xg'] = base_prob  # Para evaluación del modelo
    
    return df


if __name__ == "__main__":
    # Demo de las funciones
    print("=== Demo de utils.py para Expected Goals ===\n")
    
    # Ejemplo de cálculo de distancia y ángulo
    x, y = 108, 40  # Centro del área
    print(f"Posición: ({x}, {y})")
    print(f"Distancia al arco: {calculate_distance_to_goal(x, y):.2f} yards")
    print(f"Ángulo visible: {calculate_angle_to_goal(x, y):.2f} grados")
    print(f"xG baseline: {get_xg_baseline_by_zone(calculate_distance_to_goal(x, y), calculate_angle_to_goal(x, y)):.3f}")
    
    print("\n" + "="*50 + "\n")
    
    # Generar datos sintéticos
    df_shots = generate_synthetic_shots(100)
    print(f"Datos sintéticos generados: {len(df_shots)} tiros")
    print(f"Goles: {df_shots['goal'].sum()} ({df_shots['goal'].mean()*100:.1f}%)")
    print(f"\nDistribución de tipos de tiro:")
    print(df_shots['shot_type'].value_counts())
