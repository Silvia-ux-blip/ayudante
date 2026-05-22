import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def grafica_importancias(model, X, save_path=None):
    importances = model.feature_importances_
    feature_names = X.columns

    feature_importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': importances
    })
    feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)

    plt.figure(figsize=(12, 8))
    sns.barplot(x='Importance', y='Feature', data=feature_importance_df, palette='viridis', hue='Feature')
    plt.title('Importancia de las Características en el modelo entrenado')
    plt.xlabel('Importancia')
    plt.ylabel('Características')
    plt.grid(axis='x', linestyle='--', alpha=0.7)

    if save_path:
        plt.savefig(save_path)
    plt.show()


if __name__ == "__main__":
    import pandas as pd
    from sklearn.tree import DecisionTreeRegressor

    df = pd.read_csv("data/hormigon.csv")
    X = df.drop(columns=["strength"])
    y = df["strength"]

    model = DecisionTreeRegressor(max_depth=4, random_state=42)
    model.fit(X, y)

    grafica_importancias(model, X, save_path="importancias.png")