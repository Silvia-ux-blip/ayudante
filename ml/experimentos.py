import warnings
import time

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    r2_score,
    mean_absolute_error,
    mean_squared_error,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
)

from sklearn.linear_model import LinearRegression, Ridge, Lasso, LogisticRegression
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.ensemble import (
    RandomForestRegressor,
    RandomForestClassifier,
    GradientBoostingRegressor,
    GradientBoostingClassifier,
)
from sklearn.svm import SVR, SVC
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier

warnings.filterwarnings("ignore")


# ---------------------------------------------------------------------------
# Detección automática del tipo de problema
# ---------------------------------------------------------------------------
def _detectar_tipo(y):
    unicos = y.nunique()
    if y.dtype in ("object", "category", "bool"):
        return "clasificacion"
    if unicos <= 15:
        return "clasificacion"
    return "regresion"


# ---------------------------------------------------------------------------
# División de datos
# ---------------------------------------------------------------------------
def preparar_datos(df, target, test_size=0.2, val_size=None, random_state=42, escalar=True):
    y = df[target]
    X = df.drop(columns=[target])

    cat_cols = X.select_dtypes(include=["object", "category"]).columns
    X = pd.get_dummies(X, columns=cat_cols, drop_first=True)

    num_cols = X.select_dtypes(include=[np.number]).columns
    X[num_cols] = X[num_cols].fillna(X[num_cols].median())

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    scaler = None
    if escalar:
        scaler = StandardScaler()
        X_train[num_cols] = scaler.fit_transform(X_train[num_cols])
        X_test[num_cols] = scaler.transform(X_test[num_cols])

    print("=" * 60)
    print("DATOS PREPARADOS")
    print("=" * 60)
    print(f"Features: {X.shape[1]}  |  Target: {target}")
    print(f"Tipo: {_detectar_tipo(y)}")
    print(f"Train: {len(X_train)}  |  Test: {len(X_test)}")
    print()

    return X_train, X_test, y_train, y_test, scaler


# ---------------------------------------------------------------------------
# Modelos
# ---------------------------------------------------------------------------
def obtener_modelos(tipo="regresion"):
    if tipo == "regresion":
        return {
            "Regresión Lineal": LinearRegression(),
            "Ridge": Ridge(random_state=42),
            "Lasso": Lasso(random_state=42),
            "Árbol Decisión": DecisionTreeRegressor(max_depth=5, random_state=42),
            "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42),
            "Gradient Boosting": GradientBoostingRegressor(n_estimators=100, random_state=42),
            "KNN": KNeighborsRegressor(n_neighbors=5),
            "SVR": SVR(kernel="rbf"),
        }
    else:
        return {
            "Regresión Logística": LogisticRegression(max_iter=1000, random_state=42),
            "Árbol Decisión": DecisionTreeClassifier(max_depth=5, random_state=42),
            "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
            "Gradient Boosting": GradientBoostingClassifier(n_estimators=100, random_state=42),
            "KNN": KNeighborsClassifier(n_neighbors=5),
            "SVC": SVC(kernel="rbf", probability=True, random_state=42),
        }


# ---------------------------------------------------------------------------
# Métricas
# ---------------------------------------------------------------------------
def _metricas_regresion(y_true, y_pred):
    return {
        "R²": round(r2_score(y_true, y_pred), 4),
        "RMSE": round(np.sqrt(mean_squared_error(y_true, y_pred)), 4),
        "MAE": round(mean_absolute_error(y_true, y_pred), 4),
        "MAPE": round(np.mean(np.abs((y_true - y_pred) / y_true)) * 100, 2),
    }


def _metricas_clasificacion(y_true, y_pred, y_prob=None):
    met = {
        "Accuracy": round(accuracy_score(y_true, y_pred), 4),
        "Precision": round(precision_score(y_true, y_pred, average="weighted", zero_division=0), 4),
        "Recall": round(recall_score(y_true, y_pred, average="weighted", zero_division=0), 4),
        "F1": round(f1_score(y_true, y_pred, average="weighted", zero_division=0), 4),
    }
    if y_prob is not None and y_prob.shape[1] == 2:
        try:
            met["ROC-AUC"] = round(roc_auc_score(y_true, y_prob[:, 1]), 4)
        except Exception:
            pass
    return met


# ---------------------------------------------------------------------------
# Entrenar y evaluar un único modelo
# ---------------------------------------------------------------------------
def _entrenar_y_evaluar(modelo, nombre, X_train, y_train, X_test, y_test, tipo):
    t0 = time.time()
    modelo.fit(X_train, y_train)
    t_train = round(time.time() - t0, 3)

    t0 = time.time()
    y_pred = modelo.predict(X_test)
    t_pred = round(time.time() - t0, 3)

    if tipo == "regresion":
        metricas = _metricas_regresion(y_test, y_pred)
    else:
        y_prob = getattr(modelo, "predict_proba", None)
        y_prob = y_prob(X_test) if y_prob else None
        metricas = _metricas_clasificacion(y_test, y_pred, y_prob)

    return {
        "modelo": modelo,
        "nombre": nombre,
        "y_pred": y_pred,
        "metricas": metricas,
        "tiempo_train": t_train,
        "tiempo_pred": t_pred,
    }


# ---------------------------------------------------------------------------
# Entrenar y evaluar todos los modelos
# ---------------------------------------------------------------------------
def entrenar_y_evaluar(X_train, y_train, X_test, y_test, modelos=None, tipo=None):
    if tipo is None:
        tipo = _detectar_tipo(y_train)

    if modelos is None:
        modelos = obtener_modelos(tipo)

    resultados = []
    print("=" * 60)
    print(f"ENTRENANDO MODELOS ({tipo})")
    print("=" * 60)

    for nombre, modelo in modelos.items():
        print(f"  → {nombre}...", end=" ")
        res = _entrenar_y_evaluar(modelo, nombre, X_train, y_train, X_test, y_test, tipo)
        resultados.append(res)
        print(f"OK ({res['tiempo_train']}s)")

    print()
    return resultados


# ---------------------------------------------------------------------------
# Comparar modelos: pipeline completo
# ---------------------------------------------------------------------------
def comparar_modelos(df, target, modelos=None, test_size=0.2, random_state=42, escalar=True):
    print("\n" + "#" * 70)
    print("# COMPARACIÓN DE MODELOS")
    print("#" * 70 + "\n")

    X_train, X_test, y_train, y_test, _ = preparar_datos(
        df, target, test_size=test_size, random_state=random_state, escalar=escalar
    )

    tipo = _detectar_tipo(y_train)
    resultados = entrenar_y_evaluar(X_train, y_train, X_test, y_test, modelos, tipo)

    resumen_df = pd.DataFrame(
        [r["metricas"] for r in resultados], index=[r["nombre"] for r in resultados]
    )
    resumen_df["Tiempo (s)"] = [r["tiempo_train"] + r["tiempo_pred"] for r in resultados]

    print("=" * 60)
    print("RESUMEN DE RESULTADOS")
    print("=" * 60)

    ordenar_por = "R²" if tipo == "regresion" else "Accuracy"
    resumen_df = resumen_df.sort_values(ordenar_por, ascending=False)

    if tipo == "regresion":
        print(resumen_df[["R²", "RMSE", "MAE", "MAPE", "Tiempo (s)"]].to_string())
    else:
        print(resumen_df[["Accuracy", "Precision", "Recall", "F1", "Tiempo (s)"]].to_string())
    print()

    _graficar_comparacion(resumen_df, tipo)

    mejor = resumen_df.index[0]
    mejor_r2 = resumen_df.iloc[0][ordenar_por]
    print(f"🏆 Mejor modelo: {mejor} ({ordenar_por} = {mejor_r2})")
    print()

    modelos_entrenados = {r["nombre"]: r["modelo"] for r in resultados}

    return {
        "resultados": resultados,
        "resumen": resumen_df,
        "mejor_nombre": mejor,
        "modelos": modelos_entrenados,
        "tipo": tipo,
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
    }


# ---------------------------------------------------------------------------
# Gráfico de comparación
# ---------------------------------------------------------------------------
def _graficar_comparacion(resumen_df, tipo):
    if tipo == "regresion":
        met_princ = "R²"
        met_sec = "RMSE"
    else:
        met_princ = "Accuracy"
        met_sec = "F1"

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    colores = ["#2ecc71" if v == resumen_df[met_princ].max() else "#3498db"
               for v in resumen_df[met_princ]]

    axes[0].barh(resumen_df.index, resumen_df[met_princ], color=colores, edgecolor="white")
    axes[0].axvline(resumen_df[met_princ].mean(), color="red", linestyle="--", alpha=0.5,
                    label=f"Media: {resumen_df[met_princ].mean():.3f}")
    axes[0].set_xlabel(met_princ)
    axes[0].set_title(f"Comparación de {met_princ}")
    axes[0].invert_yaxis()
    axes[0].legend(fontsize=9)

    axes[1].barh(resumen_df.index, resumen_df[met_sec], color="steelblue", edgecolor="white")
    axes[1].axvline(resumen_df[met_sec].mean(), color="red", linestyle="--", alpha=0.5,
                    label=f"Media: {resumen_df[met_sec].mean():.3f}")
    axes[1].set_xlabel(met_sec)
    axes[1].set_title(f"Comparación de {met_sec}")
    axes[1].invert_yaxis()
    axes[1].legend(fontsize=9)

    plt.tight_layout()
    plt.show()

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.barh(resumen_df.index, resumen_df["Tiempo (s)"], color="coral", edgecolor="white")
    ax.set_xlabel("Tiempo (s)")
    ax.set_title("Tiempo de entrenamiento + predicción")
    ax.invert_yaxis()
    plt.tight_layout()
    plt.show()


# ---------------------------------------------------------------------------
# Resaltar mejores modelos
# ---------------------------------------------------------------------------
def resaltar_mejores(resultados, top_n=3):
    print("=" * 60)
    print(f"TOP {top_n} MODELOS")
    print("=" * 60)

    tipo = "regresion" if "R²" in resultados["resumen"].columns else "clasificacion"
    ordenar_por = "R²" if tipo == "regresion" else "Accuracy"

    top = resultados["resumen"].sort_values(ordenar_por, ascending=False).head(top_n)

    for i, (nombre, row) in enumerate(top.iterrows(), 1):
        met = row.drop("Tiempo (s)")
        print(f"\n{i}. {nombre}")
        print(f"   {' | '.join(f'{k}={v}' for k, v in met.items())}")
        print(f"   ⏱ {row['Tiempo (s)']}s")

    mejor = resultados["mejor_nombre"]
    print(f"\n✓ Mejor modelo global: {mejor}")

    return top
