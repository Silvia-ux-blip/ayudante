
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

def par_real_predicho(y_test, y_pred, magnitud=""):
    # 1. Crear la cuadrícula conjunta para tener control total
    g = sns.JointGrid(x=y_test, y=y_pred, marginal_ticks=True)

    # 2. Gráfico de dispersión en el área central
    g.plot_joint(sns.scatterplot, color="steelblue", alpha=0.7, edgecolor="white", s=60)

    # 3. Línea bisectriz fija (y = x) desde min(y_test) a max(y_test)
    min_v, max_v = np.min(y_test), np.max(y_test)
    g.ax_joint.plot([min_v, max_v], [min_v, max_v],
                    color='red', linestyle='--', linewidth=2, label='Bisectriz (y=x)')

    # IMPORTANTE: Forzar relación de aspecto 1:1 para que la bisectriz sea visualmente diagonal real
    g.ax_joint.set_aspect('equal', adjustable='box')

    # 4. KDE en los márgenes (reemplaza los histogramas por defecto)
    g.plot_marginals(sns.kdeplot, fill=True, color="steelblue", alpha=0.4)

    # (OPCIONAL) Si prefieres que el eje Y marginal muestre explícitamente la distribución
    # de los RESIDUOS en lugar de la distribución de y_pred, descomenta estas líneas:
    # residuos = y_pred - y_test
    # g.ax_marg_y.clear()
    # sns.kdeplot(residuos, ax=g.ax_marg_y, fill=True, color="darkorange", alpha=0.5)

    # 5. Etiquetas y formato final
    g.ax_joint.set_xlabel(f'Valor real de {magnitud}')
    g.ax_joint.set_ylabel(f'Valor predicho de {magnitud}')
    g.ax_joint.legend()

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    import pandas as pd
    y_test = pd.Series([1,2,3])
    y_pred = pd.Series([0.95, 1.87, 2.64])
    par_real_predicho(y_test, y_pred,"Valores")
    plt.savefig("regresiones.png")

from scipy import stats

def par_real_predicho_res(y_test, y_pred, magnitud="", mostrar_normal=True):
    residuos = y_pred - y_test

    # 1. Cuadrícula conjunta
    g = sns.JointGrid(x=y_test, y=residuos, marginal_ticks=True, height=8, ratio=4)

    # 2. Dispersión de residuos vs valor real
    g.plot_joint(sns.scatterplot, color="steelblue", alpha=0.7, edgecolor="white", s=50)

    # 3. Línea horizontal en y=0 (bisectriz transformada)
    g.ax_joint.axhline(y=0, color='red', linestyle='--', linewidth=2, label='Residuo = 0 (bisectriz)')

    # ✅ CORREGIDO: linestyle va dentro de line_kws
    # (ci=None es compatible con seaborn <0.12, errorbar=None con >=0.12)
    try:
        sns.regplot(x=y_test, y=residuos, ax=g.ax_joint, scatter=False, errorbar=None,
                    color='gray', line_kws={'linestyle': ':', 'label': 'Tendencia residuos'})
    except TypeError:
        sns.regplot(x=y_test, y=residuos, ax=g.ax_joint, scatter=False, ci=None,
                    color='gray', line_kws={'linestyle': ':', 'label': 'Tendencia residuos'})

    g.ax_joint.set_xlabel(f'Valor real de {magnitud}')
    g.ax_joint.set_ylabel(f'Residuo (predicho - real) de {magnitud}')
    g.ax_joint.legend(fontsize=9)
    g.ax_joint.grid(axis='y', alpha=0.3)

    # 4. Margen superior: KDE de valores reales
    g.ax_marg_x.clear()
    sns.kdeplot(x=y_test, ax=g.ax_marg_x, fill=True, color="lightgray", alpha=0.5)
    g.ax_marg_x.set_xlabel('')
    g.ax_marg_x.set_xticklabels([])

    # 5. Margen derecho: KDE de residuos
    g.ax_marg_y.clear()
    sns.kdeplot(y=residuos, ax=g.ax_marg_y, fill=True, color="darkorange",
                alpha=0.6, label='KDE residuos', linewidth=2)

    # Normal teórica superpuesta (sin escalados hacky, densities coinciden directamente)
    if mostrar_normal and len(residuos) > 10:
        mu, sigma = np.mean(residuos), np.std(residuos)
        y_vals = np.linspace(np.min(residuos), np.max(residuos), 100)
        normal_pdf = stats.norm.pdf(y_vals, mu, sigma)
        g.ax_marg_y.plot(normal_pdf, y_vals, color='green', linestyle='-', linewidth=1.5,
                         label=f'Normal teórica\n(μ={mu:.3f}, σ={sigma:.3f})')

    g.ax_marg_y.set_xlabel('Densidad')
    g.ax_marg_y.set_ylabel('')
    g.ax_marg_y.set_yticklabels([])
    g.ax_marg_y.legend(fontsize=8)
    g.ax_marg_y.grid(axis='x', alpha=0.3)

    if magnitud:
        g.fig.suptitle(f'Diagnóstico de residuos: {magnitud}', fontsize=14, y=1.02)

    plt.tight_layout()
    #plt.show()

    return {
        'mean_residual': np.mean(residuos),
        'std_residual': np.std(residuos),
        'mae': np.mean(np.abs(residuos)),
        'rmse': np.sqrt(np.mean(residuos**2))
    }

if __name__ == "__main__":
    import pandas as pd
    y_test = pd.Series([1,2,3])
    y_pred = pd.Series([0.95, 1.87, 2.64])
    par_real_predicho_res(y_test, y_pred,"Valores")
    plt.savefig("regresiones.png")
