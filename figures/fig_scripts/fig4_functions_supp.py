import os
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import matplotlib.colors as mcolors
import matplotlib.cm as cm
from scipy.stats import spearmanr
from scipy.stats import t
import scipy.stats as stats

BASE_DIR = Path(__file__).resolve().parent.parent.parent 
SAVE_DIR = BASE_DIR / "figures" / "panels" / "figure_4"

os.makedirs(SAVE_DIR, exist_ok=True)

from src.utils.utils import set_style

set_style()

##############################################################################
# Constants
##############################################################################

palletes = {
    "BMP10": "Blues",
    "BMP4": "Oranges",
    "BMP6": "Greens",
    "BMP9": "Reds",
    "GDF5": "Purples",
    "TGFb1": "YlOrBr",
}

##############################################################################
# Supplementary plot (A): BMP4 Single Cell PC1 vs. PC2
##############################################################################

def plot_pca_results(
    pca_sc_df,
    pca_mean_df,
    exp_var_dict,
    color_column="concentration",
    lims=None,
    save=False,
    reverse_pc1=False,
    plot_mean=False,
):
    concentration = pca_sc_df[color_column].unique().shape[0]
    cmap = plt.get_cmap(palletes[pca_mean_df["ligand"].mode()[0]])

    discrete_values = sorted(pca_mean_df[color_column].unique())

    norm = mcolors.BoundaryNorm(
        boundaries=[*discrete_values, discrete_values[-1] + 1],
        ncolors=cmap.N,
    )
    c = pca_sc_df[color_column]
    ligand = pca_mean_df["ligand"].mode()[0]

    pc1_sc = pca_sc_df["PC1"]
    pc1_mean = pca_mean_df["PC1"]

    if reverse_pc1:
        pc1_sc = pc1_sc * (-1)
        pc1_mean = pc1_mean * (-1)

    # Set up plot
    fig, ax = plt.subplots(figsize=(8, 4))

    # Scatter for single-cell data
    scatter = ax.scatter(
        pc1_sc,
        pca_sc_df["PC2"],
        c=c,
        cmap=cmap,
        norm=norm,
        edgecolors="gray",
        linewidth=0.4,
        alpha=0.5,
        marker="o",
        s=50,
    )

    if plot_mean:
        ax.scatter(
            pc1_mean.iloc[0],
            pca_mean_df["PC2"].iloc[0],
            c="gray",
            edgecolors="black",
            linewidth=1.0,
            marker="o",
            s=100,
        )
        # Overlay mean PCA points

        ax.scatter(
            pc1_mean.iloc[1:],
            pca_mean_df["PC2"].iloc[1:],
            c=discrete_values[1:],
            cmap=cmap,
            norm=norm,
            edgecolors="black",
            linewidth=1.0,
            marker="o",
            s=100,
        )

        ax.plot(
            pc1_mean,
            pca_mean_df["PC2"],
            color="gray",
            linewidth=3.0,
            linestyle="--",
        )

    if lims is not None:
        ax.set_xlim(lims[0])
        ax.set_ylim(lims[1])

    else:
        xmin, xmax = ax.get_xlim()
        ymin, ymax = ax.get_ylim()
        ax.set_xlim(xmin + xmin * 0.25, xmax + xmax * 0.25)
        ax.set_ylim(ymin + ymin * 0.25, ymax + ymax * 0.25)

    if plot_mean:
        ax.set_xlabel("PC1", fontsize=14)
        ax.set_ylabel("PC2", fontsize=14)

    else:
        # Axis labels
        ax.set_xlabel(
            f"PC1 ({exp_var_dict['PC1']:.2f}% variance explained)", fontsize=14
        )
        ax.set_ylabel(
            f"PC2 ({exp_var_dict['PC2']:.2f}% variance explained)", fontsize=14
        )

    # Enable tick markers and set their sizing/direction
    ax.tick_params(axis="x", labelsize=12, bottom=True, direction="out")
    ax.tick_params(axis="y", labelsize=12, left=True, direction="out")

    # Create a separate mappable for the colorbar (with full opacity)
    sm = cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])  # Required to create a valid ScalarMappable

    # Add the colorbar using the mappable without alpha
    cbar = plt.colorbar(sm, ax=ax, orientation="vertical")

    # Set label and ticks based on whether it's a concentration plot
    label = "Concentrations (ng/mL)"
    cbar.set_ticks(discrete_values)
    cbar.ax.set_yticklabels([str(v) for v in discrete_values])

    plt.tight_layout()

    if save:
        plt.savefig(
            SAVE_DIR / f"{ligand}_{concentration}_PC1_vs_PC2_sem.pdf",
            format="pdf",
            bbox_inches="tight",
        )

    plt.show()

##############################################################################
# Supplementary plot (B): BMP4 Single Cell PC1 vs. PC2
##############################################################################

def spearman_r_for_p_value(N, alpha=0.05, two_tailed=True):
    # Adjust alpha for two-tailed test
    alpha_val = alpha / 2 if two_tailed else alpha

    # Degrees of freedom
    df = N - 2

    # Critical t-value using inverse survival function (1 - alpha)
    t_critical = stats.t.isf(alpha_val, df)

    # Solve for r from the t-statistic formula: t = r * sqrt((N-2) / (1-r^2))
    # r = t / sqrt(t^2 + N - 2)
    r_critical = t_critical / np.sqrt(t_critical**2 + df)

    return r_critical

def pearson_r_for_p_value(N, alpha=0.05):
    df = N - 2
    t_crit = t.ppf(1 - alpha / 2, df)
    r_critical = t_crit / np.sqrt(t_crit**2 + df)

    return r_critical

def generate_correlation_dict(pca_df_bmp4):
    data = {"concentration": [], "correlation": [], "p_value": [], "threshold": []}

    unique_concentrations = np.sort(pca_df_bmp4["concentration"].unique())

    # Bonferroni correction for multiple comparisons
    alpha_corrected = 0.05  # / len(unique_concentrations)

    for concentration in unique_concentrations:
        df_filtered = pca_df_bmp4[pca_df_bmp4["concentration"] == concentration]

        # Calculate Spearman correlation
        correlation, p_value = spearmanr(df_filtered["log_h2b"], df_filtered["PC1"])

        # Calculate critical r threshold
        threshold = spearman_r_for_p_value(len(df_filtered), alpha=alpha_corrected)

        data["concentration"].append(concentration)
        data["correlation"].append(correlation)
        data["p_value"].append(p_value)
        data["threshold"].append(threshold)

    corr_dict_df = pd.DataFrame(data)
    corr_dict_df["concentration"] = corr_dict_df["concentration"].astype(str)
    corr_dict_df["opp_threshold"] = -corr_dict_df["threshold"]

    return corr_dict_df

def plot_pc1_h2b_corr_line(corr_dict_df, ligand, save=False):
    plt.figure(figsize=(7, 5))
    sns.set(style="whitegrid")
    plt.grid(True)

    # Plot correlation line
    sns.lineplot(
        data=corr_dict_df,
        x="concentration",
        y="correlation",
        marker="o",
        color="black",
        linewidth=2,
        markersize=8,
        markeredgewidth=0,
    )

    # Plot threshold lines
    plt.plot(
        corr_dict_df["concentration"],
        corr_dict_df["threshold"],
        color="gray",
        linestyle="--",
    )
    plt.plot(
        corr_dict_df["concentration"],
        corr_dict_df["opp_threshold"],
        color="gray",
        linestyle="--",
    )

    # Fill between threshold and opp_threshold
    plt.fill_between(
        corr_dict_df["concentration"],
        corr_dict_df["threshold"],
        corr_dict_df["opp_threshold"],
        color="gray",
        alpha=0.2,
        label="Non-significant\n region",
    )

    # Format ticks and force tick markers to show outside the grid
    plt.xticks(size=12)
    plt.yticks(size=12)
    plt.gca().tick_params(axis="both", which="both", bottom=True, left=True, direction="out")

    plt.xlabel("BMP4 Concentrations (ng/mL)")
    plt.ylabel("Spearman Coefficient")
    plt.legend()

    if save:
        plt.savefig(
            SAVE_DIR / f"{ligand}_PC1_vs_H2B_corr_line.pdf",
            format="pdf",
            bbox_inches="tight",
        )

    plt.show()

##############################################################################
# Supplementary plot (C): BMP4 PC1 Weights
##############################################################################

def extract_pca_weights(lig_model):
    lig_pca_model = lig_model.pca_model_mean

    pc1_components = lig_pca_model.components_[0]
    feature_names = lig_pca_model.feature_names_in_
    pca_weights_df = pd.DataFrame({"PC1": pc1_components}, index=feature_names)

    pca_weights_df = pca_weights_df.sort_values(by="PC1", ascending=False)

    return pca_weights_df


def plot_weights_bar_plot(pca_weights_df, save=False):
    plt.figure(figsize=(10, 6))

    plt.bar(
        pca_weights_df.index,
        pca_weights_df["PC1"],
        color=["limegreen" if v >= 0 else "tomato" for v in pca_weights_df["PC1"]],
    )

    plt.axhline(0, color="black", linewidth=0.8)
    plt.grid(axis="x")
    plt.xticks([])
    plt.ylabel("Weights")

    if save:
        plt.savefig(
            SAVE_DIR / "PC1_weights_bar_plot.pdf",
            format="pdf",
            bbox_inches="tight",
        )
    plt.show()

##############################################################################
# Supplementary plot (Removed): BMP4 vs. H2BCITRINE separated by concentrations
##############################################################################

def plot_pca_vs_h2b_across_concentration(pca_df, save=False):
    unique_concentrations = np.sort(pca_df["concentration"].unique())

    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    axes = axes.flatten()

    corr_dict = {"concentration": [], "correlation": [], "p_value": []}

    for i, concentration in enumerate(unique_concentrations):
        df_filtered = pca_df[pca_df["concentration"] == concentration]

        axes[i].scatter(
            df_filtered["PC1"],
            df_filtered["log_h2b"],
            alpha=0.4,
            edgecolors="none",
            color="royalblue",
            s=75,
        )

        axes[i].set_title(f"Concentration: {concentration} ng/mL")
        axes[i].set_xlabel("PC1")

        correlation, p_value = spearmanr(df_filtered["log_h2b"], df_filtered["PC1"])
        corr_dict["concentration"].append(concentration)
        corr_dict["correlation"].append(correlation)
        corr_dict["p_value"].append(p_value)

        axes[i].set_title(
            f"Concentration: {concentration} ng/mL\nCorrelation: {correlation:.2f}\nP-value: {p_value:.2e}",
            fontsize=10,
        )
        axes[i].grid(False)

    if save:
        os.makedirs(SAVE_DIR, exist_ok=True)
        plt.savefig(
            SAVE_DIR / "PCA_vs_H2B_across_concentration.pdf",
            format="pdf",
            bbox_inches="tight",
        )

    # plt.grid(False)
    plt.tight_layout()
    plt.show()