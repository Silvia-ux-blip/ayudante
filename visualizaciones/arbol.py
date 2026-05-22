from sklearn.tree import plot_tree
import matplotlib.pyplot as plt

def grafica_arbol(model, X):
    plt.figure(figsize=(20,10))
    plot_tree(model, filled=True, feature_names=X.columns, rounded=True, fontsize=10)
    plt.title("Árbol de Decisión de Regresión")
    plt.show()

if __name__ == "__main__":
    import pandas as pd
    from sklearn.tree import DecisionTreeRegressor
    df = pd.read_csv("data/hormigon.csv")
    X = df.drop(columns=["strength"])
    y = df["strength"]
    model = DecisionTreeRegressor(max_depth=4, random_state=42)
    model.fit(X, y)
    grafica_arbol(model, X)
    plt.savefig("arbol.png")