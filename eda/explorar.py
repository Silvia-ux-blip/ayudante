import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats


def resumen(df):
    print("=" * 60)
    print("RESUMEN GENERAL DEL DATAFRAME")
    print("=" * 60)
    print(f"Dimensiones: {df.shape[0]} filas x {df.shape[1]} columnas")
    print(f"Memoria aprox: {df.memory_usage(deep=True).sum() / 1024 ** 2:.2f} MB")
    print(f"Duplicados: {df.duplicated().sum()}")
    print()

    nulos = df.isnull().sum()
    if nulos.sum() > 0:
        print("Columnas con nulos:")
        print(nulos[nulos > 0].to_string())
    else:
        print("Sin valores nulos.")

    print("\nTIPOS DE DATOS:")
    print(df.dtypes.to_string())
    print()

    return {
        "filas": df.shape[0],
        "columnas": df.shape[1],
        "memoria_mb": df.memory_usage(deep=True).sum() / 1024 ** 2,
        "duplicados": df.duplicated().sum(),
    }


def describir(df):
    num = df.select_dtypes(include=[np.number])
    if num.empty:
        print("No hay columnas numéricas.")
        return

    desc = num.describe().T
    desc["skew"] = num.skew()
    desc["kurtosis"] = num.kurtosis()
    desc["iqr"] = desc["75%"] - desc["25%"]
    desc["cv"] = desc["std"] / desc["mean"]
    desc["outliers_iqr"] = (
        ((num < (desc["25%"] - 1.5 * desc["iqr"])) | (num > (desc["75%"] + 1.5 * desc["iqr"]))).sum()
    )

    print("=" * 60)
    print("ESTADÍSTICAS DESCRIPTIVAS (numéricas)")
    print("=" * 60)
    print(desc.round(4).to_string())
    print()

    return desc


def analizar_nulos(df, figsize=(10, 5)):
    nulos = df.isnull().sum()
    pct = (df.isnull().mean() * 100).round(2)

    tabla = pd.DataFrame({"Nulos": nulos, "%": pct})
    tabla = tabla[tabla["Nulos"] > 0].sort_values("%", ascending=False)

    if tabla.empty:
        print("No hay valores nulos en el DataFrame.")
        return tabla

    print("=" * 60)
    print("ANÁLISIS DE VALORES NULOS")
    print("=" * 60)
    print(tabla.to_string())
    print()

    fig, ax = plt.subplots(figsize=figsize)
    ax.barh(tabla.index, tabla["%"], color="coral", edgecolor="white")
    ax.set_xlabel("Porcentaje de nulos (%)")
    ax.set_title("Valores nulos por columna")
    ax.invert_yaxis()
    plt.tight_layout()
    plt.show()

    return tabla


def detectar_outliers(df, metodo="iqr", factor=1.5):
    num = df.select_dtypes(include=[np.number])
    if num.empty:
        print("No hay columnas numéricas.")
        return

    resultados = {}
    for col in num.columns:
        q1, q3 = num[col].quantile(0.25), num[col].quantile(0.75)
        iqr = q3 - q1

        if metodo == "iqr":
            lim_inf = q1 - factor * iqr
            lim_sup = q3 + factor * iqr
            outliers = num[(num[col] < lim_inf) | (num[col] > lim_sup)][col]
        elif metodo == "zscore":
            z = np.abs(stats.zscore(num[col], nan_policy="omit"))
            outliers = num[col][z > factor]
        else:
            raise ValueError("metodo debe ser 'iqr' o 'zscore'")

        resultados[col] = {
            "outliers": len(outliers),
            "%": round(len(outliers) / len(num) * 100, 2),
            "lim_inf": round(lim_inf, 4) if metodo == "iqr" else None,
            "lim_sup": round(lim_sup, 4) if metodo == "iqr" else None,
        }

    res = pd.DataFrame(resultados).T.sort_values("outliers", ascending=False)
    print("=" * 60)
    print(f"DETECCIÓN DE OUTLIERS (método: {metodo})")
    print("=" * 60)
    print(res.to_string())
    print()

    return res


def distribuciones(df, cols=None, bins=30, figsize=(15, 10)):
    num = df.select_dtypes(include=[np.number])
    if cols is not None:
        num = num[[c for c in cols if c in num.columns]]
    if num.empty:
        print("No hay columnas numéricas para graficar.")
        return

    n = len(num.columns)
    ncols = 3
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
    axes = axes.flatten()

    for i, col in enumerate(num.columns):
        ax = axes[i]
        ax.hist(num[col].dropna(), bins=bins, color="steelblue", edgecolor="white", alpha=0.8)
        ax.set_title(col, fontsize=11)
        ax.set_xlabel("")
        ax.set_ylabel("Frecuencia")
        ax.grid(axis="y", linestyle="--", alpha=0.4)

    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    fig.suptitle("Distribuciones de variables numéricas", fontsize=14, y=1.02)
    plt.tight_layout()
    plt.show()


def boxplots(df, cols=None, figsize=(15, 8)):
    num = df.select_dtypes(include=[np.number])
    if cols is not None:
        num = num[[c for c in cols if c in num.columns]]
    if num.empty:
        print("No hay columnas numéricas para graficar.")
        return

    n = len(num.columns)
    ncols = 4
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
    axes = axes.flatten()

    for i, col in enumerate(num.columns):
        ax = axes[i]
        ax.boxplot(num[col].dropna(), patch_artist=True,
                   boxprops=dict(facecolor="lightblue"),
                   medianprops=dict(color="red"))
        ax.set_title(col, fontsize=10)
        ax.set_xticks([])

    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    fig.suptitle("Boxplots de variables numéricas", fontsize=14, y=1.02)
    plt.tight_layout()
    plt.show()


def categoricas(df, cols=None, top=10, figsize=(12, 8)):
    cat = df.select_dtypes(include=["object", "category", "bool"])
    if cols is not None:
        cat = cat[[c for c in cols if c in cat.columns]]
    if cat.empty:
        print("No hay columnas categóricas.")
        return

    for col in cat.columns:
        print(f"\n{'=' * 50}")
        print(f"COLUMNA: {col}")
        print(f"Valores únicos: {cat[col].nunique()}")
        print(f"Cardinalidad: {cat[col].nunique() / len(cat) * 100:.1f}%")
        print(f"{'=' * 50}")
        counts = cat[col].value_counts().head(top)
        print(counts.to_string())

    for col in cat.columns[:4]:
        fig, ax = plt.subplots(figsize=figsize)
        counts = cat[col].value_counts().head(top)
        ax.barh(range(len(counts)), counts.values, color="steelblue", edgecolor="white")
        ax.set_yticks(range(len(counts)))
        ax.set_yticklabels(counts.index)
        ax.set_xlabel("Frecuencia")
        ax.set_title(f"Top {top} - {col}")
        ax.invert_yaxis()
        plt.tight_layout()
        plt.show()


def correlaciones(df, method="pearson", figsize=(10, 8)):
    num = df.select_dtypes(include=[np.number])
    if num.shape[1] < 2:
        print("Se necesitan al menos 2 columnas numéricas.")
        return

    corr = num.corr(method=method)
    mask = np.triu(np.ones_like(corr, dtype=bool), k=1)

    fig, ax = plt.subplots(figsize=figsize)
    cmap = sns.diverging_palette(230, 20, as_cmap=True)
    sns.heatmap(corr, mask=mask, annot=True, cmap=cmap, center=0,
                square=True, linewidths=0.5, cbar_kws={"shrink": 0.6})
    ax.set_title(f"Matriz de correlación ({method})", fontsize=13)
    plt.tight_layout()
    plt.show()

    pares = corr.unstack().dropna()
    pares = pares[pares < 1]
    print("Pares más correlacionados (positivo):")
    print(pares.sort_values(ascending=False).head(5).to_string())
    print("\nPares más correlacionados (negativo):")
    print(pares.sort_values().head(5).to_string())


def reporte_eda(df, target=None, bins=30):
    print("\n" + "#" * 70)
    print("# INFORME DE ANÁLISIS EXPLORATORIO (EDA)")
    print("#" * 70 + "\n")

    resumen(df)
    describir(df)
    analizar_nulos(df)

    if df.select_dtypes(include=["object", "category", "bool"]).shape[1] > 0:
        print("\nVARIABLES CATEGÓRICAS:")
        print("-" * 40)
        cat = df.select_dtypes(include=["object", "category", "bool"])
        for col in cat.columns:
            print(f"  {col}: {cat[col].nunique()} valores únicos | {cat[col].isnull().sum()} nulos")
            print(f"    Top: {cat[col].value_counts().index[0]} ({cat[col].value_counts().iloc[0]})")
    print()

    detectar_outliers(df)
    correlaciones(df)

    num = df.select_dtypes(include=[np.number])
    if len(num.columns) > 0:
        distribuciones(df, bins=bins)
        boxplots(df)

    if target is not None and target in num.columns:
        print(f"\nANÁLISIS vs TARGET '{target}':")
        print("-" * 40)
        corr_target = num.corr()[target].drop(target).sort_values(ascending=False)
        print(corr_target.round(4).to_string())

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        axes[0].hist(df[target].dropna(), bins=bins, color="coral", edgecolor="white")
        axes[0].set_title(f"Distribución de {target}")
        axes[0].set_xlabel(target)

        axes[1].boxplot(df[target].dropna(), patch_artist=True,
                        boxprops=dict(facecolor="lightgreen"),
                        medianprops=dict(color="red"))
        axes[1].set_title(f"Boxplot de {target}")
        axes[1].set_xticks([])
        plt.tight_layout()
        plt.show()

    print("\n" + "#" * 70)
    print("# FIN DEL INFORME EDA")
    print("#" * 70)
