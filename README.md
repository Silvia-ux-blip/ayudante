# 🛠 Ayudante

**Kit de herramientas reutilizables para Análisis Exploratorio de Datos (EDA) y Machine Learning en Python.**

Diseñado para acelerar el flujo de trabajo de ciencia de datos: carga un CSV y obtén desde el EDA completo hasta la comparación de múltiples modelos con una sola llamada.

---

## 📦 Estructura del proyecto

```
ayudante/
├── eda/                     # Módulo de análisis exploratorio
│   ├── __init__.py
│   └── explorar.py          # Funciones: resumen, describir, nulos, outliers, etc.
├── ml/                      # Módulo de machine learning
│   ├── __init__.py
│   └── experimentos.py      # Pipeline: preparar datos, entrenar, comparar modelos
├── visualizaciones/          # Módulo de gráficos especializados
│   ├── __init__.py
│   ├── arbol.py             # Visualización de árbol de decisión
│   ├── correlacion.py       # Heatmap de correlaciones
│   ├── importancias.py      # Importancia de características
│   └── regresiones.py       # Real vs predicho y diagnóstico de residuos
├── data/                    # Datos de ejemplo (hormigón)
├── venv/                    # Entorno virtual (no se sube a GitHub)
├── requirements.txt
└── README.md
```

---

## ⚙️ Instalación

### Local

```bash
# 1. Clonar el repositorio
git clone https://github.com/Silvia-ux-blip/ayudante.git
cd ayudante

# 2. Crear y activar entorno virtual
python -m venv venv
source venv/bin/activate       # Linux/Mac
# venv\Scripts\activate        # Windows

# 3. Instalar dependencias
pip install -r requirements.txt
```

### Google Colab

```python
# Instalar directamente desde GitHub (sin clonar)
!pip install git+https://github.com/Silvia-ux-blip/ayudante.git

# Descargar datos de ejemplo
!wget -q https://raw.githubusercontent.com/tu-usuario/Silvia-ux-blip/main/data/hormigon.csv -O hormigon.csv

# Importar y usar
import pandas as pd
from eda import reporte_eda
from ml import comparar_modelos

df = pd.read_csv("hormigon.csv")
reporte_eda(df, target="strength")
resultados = comparar_modelos(df, target="strength")
```

---

## 🔍 Módulo EDA — `eda.explorar`

Carga cualquier CSV y ejecuta un análisis exploratorio completo:

```python
import pandas as pd
from eda import reporte_eda

df = pd.read_csv("data/hormigon.csv")
reporte_eda(df)                        # Sin target
reporte_eda(df, target="strength")     # Con variable objetivo
```

### Funciones disponibles

| Función | Descripción |
|---|---|
| `resumen(df)` | Dimensiones, tipos de datos, memoria, duplicados, nulos |
| `describir(df)` | Estadísticas descriptivas + asimetría, curtosis, IQR, CV, outliers |
| `analizar_nulos(df)` | Tabla y gráfico de valores nulos por columna |
| `detectar_outliers(df)` | Detección de outliers (método IQR o Z-score) |
| `distribuciones(df)` | Grid de histogramas para variables numéricas |
| `boxplots(df)` | Grid de boxplots para variables numéricas |
| `categoricas(df)` | Frecuencias y gráficos para variables categóricas |
| `correlaciones(df)` | Mapa de calor triangular + pares más correlacionados |
| `reporte_eda(df, target)` | **Informe completo** que ejecuta todo lo anterior |

---

## 🤖 Módulo ML — `ml.experimentos`

Pipeline completo: preparación → entrenamiento múltiple → comparación → modelo listo para visualizar.

```python
from ml import comparar_modelos
import pandas as pd

df = pd.read_csv("data/hormigon.csv")

# Un solo llamado entrena 8 modelos y los compara
resultados = comparar_modelos(df, target="strength")
```

### ¿Qué hace `comparar_modelos`?

1. **Prepara** los datos (train/test, one-hot encoding, escalado)
2. **Entrena** todos los modelos de forma automática
3. **Evalúa** con métricas (R², RMSE, MAE, MAPE o Accuracy, F1, etc.)
4. **Grafica** la comparación visual
5. **Devuelve** los modelos entrenados listos para usar

### Modelos incluidos

| Regresión | Clasificación |
|---|---|
| Regresión Lineal | Regresión Logística |
| Ridge | Árbol de Decisión |
| Lasso | Random Forest |
| Árbol de Decisión | Gradient Boosting |
| Random Forest | KNN |
| Gradient Boosting | SVC |
| KNN | |
| SVR | |

### Integración con visualizaciones

```python
from visualizaciones.importancias import grafica_importancias
from visualizaciones.regresiones import par_real_predicho

# El mejor modelo
modelo = resultados["modelos"]["Random Forest"]

# Importancia de características
grafica_importancias(modelo, resultados["X_test"], save_path="importancias.png")

# Real vs predicho
par_real_predicho(resultados["y_test"], modelo.predict(resultados["X_test"]), magnitud="strength")
```

### Funciones disponibles

| Función | Descripción |
|---|---|
| `preparar_datos(df, target)` | Divide en train/test, codifica categóricas, escala |
| `obtener_modelos(tipo)` | Diccionario de modelos según regresión o clasificación |
| `entrenar_y_evaluar(X_train, ...)` | Entrena todos los modelos y devuelve métricas |
| `comparar_modelos(df, target)` | **Pipeline completo** en una sola llamada |
| `resaltar_mejores(resultados)` | Muestra el podio de los mejores modelos |

> 🔄 **Detección automática**: si el target tiene ≤15 valores únicos o es texto, se trata como clasificación. En caso contrario, como regresión.

---

## 📊 Módulo de Visualizaciones — `visualizaciones`

Funciones gráficas para trabajar con modelos ya entrenados:

| Función | Descripción |
|---|---|
| `grafica_arbol(model, X)` | Dibuja un árbol de decisión |
| `heatmap_corr(df)` | Mapa de calor triangular de correlaciones |
| `grafica_importancias(model, X)` | Importancia de características (barplot) |
| `par_real_predicho(y_test, y_pred, magnitud)` | Real vs predicho con bisectriz y KDE marginal |
| `par_real_predicho_res(y_test, y_pred, magnitud)` | Diagnóstico de residuos con métricas |

---

## 📁 Dataset de ejemplo

**`data/hormigon.csv`** — Resistencia a la compresión del hormigón (Concrete Compressive Strength).

- **1030 muestras**, 8 features + 1 target (`strength`)
- Variables: cemento, escoria, ceniza, agua, superplastificante, agregado grueso, agregado fino, edad

---

## ✨ Flujo de trabajo recomendado

```python
import pandas as pd
from eda import reporte_eda
from ml import comparar_modelos
from visualizaciones.importancias import grafica_importancias
from visualizaciones.regresiones import par_real_predicho

df = pd.read_csv("data/hormigon.csv")

# 1. EDA completo
reporte_eda(df, target="strength")

# 2. Probar modelos
resultados = comparar_modelos(df, target="strength")

# 3. Visualizar el mejor modelo
modelo = resultados["modelos"][resultados["mejor_nombre"]]
grafica_importancias(modelo, resultados["X_test"])
par_real_predicho(resultados["y_test"], modelo.predict(resultados["X_test"]), "strength")
```

---

## 📄 Requisitos

- Python ≥ 3.10
- pandas, numpy, matplotlib, seaborn, scikit-learn, scipy

Ver `requirements.txt` para versiones específicas.

---

## 📌 Notas para GitHub

- El directorio `venv/` está excluido vía `.gitignore`
- Los archivos `.png` generados al ejecutar los scripts no se suben (añádelos a `.gitignore` si quieres evitarlos)
- Los datos de `data/` se incluyen para que los ejemplos funcionen directamente

---

Hecho con ❤️ para facilitar el trabajo de ciencia de datos.
