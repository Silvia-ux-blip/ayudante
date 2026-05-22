import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

def heatmap_corr(df):
    sns.set_theme(style="white")

    numeric_df = df.select_dtypes(include=["number", "bool"])

    # Compute the correlation matrix
    corr = numeric_df.corr()

    # Generate a mask for the upper triangle
    mask = np.triu(np.ones_like(corr, dtype=bool))

    # Set up the matplotlib figure
    f, ax = plt.subplots(figsize=(11, 9))

    # Generate a custom diverging colormap
    cmap = sns.diverging_palette(230, 20, as_cmap=True)

    # Draw the heatmap with the mask and correct aspect ratio
    sns.heatmap(corr, mask=mask, annot=True, cmap=cmap, vmax=.3, center=0,
                square=True, linewidths=.5, cbar_kws={"shrink": .5})
    
if __name__ == "__main__":
    import pandas as pd
    df = pd.read_csv("data/hormigon.csv")
    heatmap_corr(df)
    plt.savefig("heatmap.png")
    