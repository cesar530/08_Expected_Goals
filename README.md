# 🎯 Expected Goals (xG) Model

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10+-blue.svg" alt="Python">
  <img src="https://img.shields.io/badge/License-MIT-green.svg" alt="License">
  <img src="https://img.shields.io/badge/Status-Active-brightgreen.svg" alt="Status">
  <img src="https://img.shields.io/badge/Sports-Analytics-orange.svg" alt="Sports Analytics">
</p>

## 📋 Descripción

Este proyecto implementa un **modelo de Expected Goals (xG)** estilo Opta/StatsBomb para análisis de fútbol. El modelo predice la probabilidad de que un tiro termine en gol basándose en múltiples factores contextuales y geométricos.

El Expected Goals (xG) es una métrica avanzada ampliamente utilizada en el análisis deportivo moderno para evaluar la calidad de las oportunidades de gol.

## ⚽ ¿Qué es Expected Goals (xG)?

El xG asigna un valor entre 0 y 1 a cada tiro, representando la probabilidad de que ese tiro termine en gol basándose en miles de tiros históricos similares. Por ejemplo:
- Un **penal** típicamente tiene un xG de ~0.76
- Un **tiro desde fuera del área** puede tener un xG de ~0.05
- Un **uno contra uno** frente al portero puede tener un xG de ~0.35

## 🎯 Factores del Modelo

El modelo considera los siguientes factores para calcular el xG:

| Factor | Descripción |
|--------|-------------|
| **Posición del tiro** | Coordenadas X, Y en el campo |
| **Distancia al arco** | Distancia euclidiana al centro de la portería |
| **Ángulo visible** | Ángulo del arco visible desde la posición del tiro |
| **Tipo de jugada** | Open Play, Penal, Tiro Libre, Corner |
| **Parte del cuerpo** | Pie derecho, pie izquierdo, cabeza |
| **Tipo de pase previo** | Pase filtrado, centro, recorte, etc. |

## 🔧 Tecnologías Utilizadas

- **Python 3.10+**
- **Scikit-learn** - Logistic Regression
- **XGBoost** - Gradient Boosting
- **Pandas & NumPy** - Procesamiento de datos
- **Matplotlib & Seaborn** - Visualización
- **mplsoccer** - Visualización de campos de fútbol

## 📁 Estructura del Proyecto

```
08_Expected_Goals/
├── 📓 xG_Expected_Goals_Model.ipynb   # Notebook principal con análisis completo
├── 🐍 expected_goals.py                # Clases del modelo xG
├── 🔧 utils.py                         # Funciones auxiliares
├── 📋 requirements.txt                 # Dependencias del proyecto
├── 📖 README.md                        # Este archivo
├── 🚫 .gitignore                       # Archivos ignorados por Git
├── 📁 models/                          # Modelos entrenados
│   └── xg_model_xgboost.joblib
├── 📁 figures/                         # Visualizaciones generadas
│   ├── mapa_tiros.png
│   ├── heatmap_conversion.png
│   └── ...
└── 📁 data/                            # Datos (opcional)
    └── sample_shots.csv
```

## 🚀 Instalación

### 1. Clonar el repositorio

```bash
git clone https://github.com/cesar530/expected-goals-model.git
cd expected-goals-model
```

### 2. Crear entorno virtual (recomendado)

```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows
```

### 3. Instalar dependencias

```bash
pip install -r requirements.txt
```

> ⚠️ **Nota:** Las versiones de las librerías están especificadas para evitar el error `ValueError: numpy.dtype size changed`.

## 💻 Uso Rápido

### Opción 1: Usar el Notebook

Abre `xG_Expected_Goals_Model.ipynb` en Jupyter y ejecuta las celdas secuencialmente.

### Opción 2: Usar como Script

```python
from expected_goals import ExpectedGoalsModel, train_xg_pipeline
from utils import generate_synthetic_shots, extract_shot_features

# Generar datos de prueba
df_shots = generate_synthetic_shots(n_shots=2000)

# Entrenar modelo
model = ExpectedGoalsModel(model_type='xgboost')
model.fit(df_shots, target_col='goal')

# Predecir xG
df_shots['xG'] = model.predict_proba(df_shots)

# Evaluar
metrics = model.evaluate(df_shots)
print(f"ROC-AUC: {metrics['roc_auc']:.4f}")
```

### Opción 3: Calcular xG de un tiro específico

```python
from utils import calculate_distance_to_goal, calculate_angle_to_goal

# Posición del tiro (coordenadas StatsBomb: 120x80)
x, y = 108, 40  # Dentro del área, centrado

# Calcular métricas geométricas
distance = calculate_distance_to_goal(x, y)
angle = calculate_angle_to_goal(x, y)

print(f"Distancia: {distance:.2f} yards")
print(f"Ángulo visible: {angle:.2f}°")
```

## 📊 Resultados del Modelo

### Métricas de Evaluación

| Modelo | ROC-AUC | Brier Score | Log Loss |
|--------|---------|-------------|----------|
| Logistic Regression | 0.82 | 0.08 | 0.28 |
| **XGBoost** | **0.85** | **0.07** | **0.25** |

### Visualizaciones Generadas

<table>
<tr>
<td><strong>Mapa de Tiros</strong></td>
<td><strong>Mapa de Calor xG</strong></td>
</tr>
<tr>
<td>Distribución de goles y no goles</td>
<td>Probabilidad de gol por zona</td>
</tr>
</table>

## 📈 Features Más Importantes

1. **distance_to_goal** - Distancia al arco (más importante)
2. **angle_to_goal** - Ángulo visible de la portería
3. **angle_distance_ratio** - Ratio ángulo/distancia
4. **is_penalty** - Indicador de penal
5. **body_part_Head** - Indicador de cabezazo

## 🔄 Pipeline de Entrenamiento

```python
from expected_goals import train_xg_pipeline

# Pipeline completo
model, df_with_predictions = train_xg_pipeline(
    data_source='synthetic',  # o ruta a CSV
    model_type='xgboost',
    n_samples=3000,
    save_path='models/my_model.joblib',
    verbose=True
)
```

## 📚 API Reference

### `ExpectedGoalsModel`

```python
class ExpectedGoalsModel:
    def __init__(self, model_type='xgboost', random_state=42, **model_params)
    def fit(self, df, target_col='goal', verbose=True)
    def predict_proba(self, df) -> np.ndarray
    def predict(self, df, threshold=0.5) -> np.ndarray
    def evaluate(self, df, target_col='goal') -> dict
    def get_feature_importance(self) -> pd.DataFrame
    def save_model(self, filepath)
    @classmethod
    def load_model(cls, filepath) -> 'ExpectedGoalsModel'
```

### Funciones Utilitarias (`utils.py`)

```python
calculate_distance_to_goal(x, y) -> float
calculate_angle_to_goal(x, y) -> float
extract_shot_features(df) -> pd.DataFrame
create_shot_type_features(df) -> pd.DataFrame
generate_synthetic_shots(n_shots=1000) -> pd.DataFrame
```

## 🗺️ Roadmap

- [x] Modelo base con Logistic Regression
- [x] Modelo XGBoost
- [x] Visualización de campos de fútbol
- [x] Feature engineering geométrico
- [ ] Integración con StatsBomb Open Data
- [ ] Dashboard interactivo (Streamlit/Plotly)
- [ ] Modelo específico para cabezazos
- [ ] Inclusión de features de presión defensiva
- [ ] API REST para predicciones

## 🤝 Contribuciones

Las contribuciones son bienvenidas. Por favor:

1. Fork el repositorio
2. Crea una rama para tu feature (`git checkout -b feature/AmazingFeature`)
3. Commit tus cambios (`git commit -m 'Add AmazingFeature'`)
4. Push a la rama (`git push origin feature/AmazingFeature`)
5. Abre un Pull Request

## 📄 Licencia

Este proyecto está bajo la Licencia MIT. Ver el archivo [LICENSE](LICENSE) para más detalles.

```
MIT License

Copyright (c) 2026 César Adrián Delgado Díaz

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

## 👤 Autor

**César Adrián Delgado Díaz**

- 🌐 Portfolio: [tu-portfolio.com](https://tu-portfolio.com)
- 💼 LinkedIn: [linkedin.com/in/cesar-delgado-diaz](https://www.linkedin.com/in/cesar-delgado-diaz)
- 🐙 GitHub: [github.com/cesar530](https://github.com/cesar530)

---

## 📚 Referencias

- [StatsBomb Open Data](https://github.com/statsbomb/open-data)
- [Opta Sports](https://www.optasports.com/)
- [mplsoccer Documentation](https://mplsoccer.readthedocs.io/)
- [Expected Goals Explained - The Analyst](https://theanalyst.com/eu/2021/07/what-are-expected-goals-xg/)

---

<p align="center">
  ⭐ Si este proyecto te fue útil, considera darle una estrella en GitHub ⭐
</p>
