import os
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import math

from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from scipy.cluster.hierarchy import linkage, leaves_list
from scipy.stats import zscore

BASE_DIR = Path(__file__).resolve().parent.parent.parent 
SAVE_DIR = BASE_DIR / "figures" / "panels" / "figure_2"

os.makedirs(SAVE_DIR, exist_ok=True)
os.makedirs(SAVE_DIR / "supplementary", exist_ok=True)


from src.utils.utils import (
    add_df_ligand_names_and_concentrations_columns, set_style
)

set_style()

###########################################################
# Variables:
###########################################################

COLORS = {
    "CTRL_1": "gray",
    "BMP10": "blue",
    "BMP4": "orange",
    "BMP6": "green",
    "BMP9": "red",
    "GDF5": "purple",
    "TGFb1": "brown",
}

CLUSTERS_COLORS = {
    "BMP10": "tab:blue",
    "BMP4": "tab:orange",
    "BMP6": "tab:green",
    "BMP9": "tab:red",
    "GDF5": "tab:purple",
    "TGFb1": "tab:brown",
}

###########################################################
# Panel A - 3D PCA:
###########################################################


def perform_pca(df, n_components=8, log_transform=True, epsilon=1):
    """Aggregates to pseudobulk, handles CTRL, and runs PCA without scaling."""
    
    # 1. Prepare and Log Transform
    df_proc = df.copy() + epsilon
    if log_transform:
        df_proc = np.log2(df_proc)

    # 2. Pseudobulk Aggregation
    # Group by index to get sample means
    df_mean = df_proc.groupby(df_proc.index).mean()
    
    # 3. Aggregate CTRL logic
    # Collapse all CTRL rows into a single mean row
    is_ctrl = df_mean.index.str.contains("CTRL")
    ctrl_combined = df_mean[is_ctrl].mean(axis=0)
    
    # Filter out individual CTRLs and append the single aggregated CTRL
    df_final = df_mean[~is_ctrl].copy()
    df_final.loc["CTRL"] = ctrl_combined

    # 4. Run PCA (StandardScaler skipped per instructions)
    pca_model = PCA(n_components=n_components)
    pca_results = pca_model.fit_transform(df_final)

    # 5. Format Results
    cols = [f"PC{i+1}" for i in range(n_components)]
    pca_results_df = pd.DataFrame(pca_results, columns=cols, index=df_final.index)
    
    # Optional: Re-attach metadata if your helper function is available
    try:
        pca_results_df = add_df_ligand_names_and_concentrations_columns(pca_results_df)
        pca_results_df = pca_results_df.sort_values(by="concentration")
    except NameError:
        pass

    return pca_results_df, pca_model

def get_variance_explained(pca_model):
    """Returns the % variance explained for PC1 and PC2 directly from the model."""
    var_ratios = pca_model.explained_variance_ratio_ * 100
    return {"PC1_exp_var": var_ratios[0],
            "PC2_exp_var": var_ratios[1],
            "PC3_exp_var": var_ratios[2],
            "PC1_PC2_exp_var_sum": var_ratios[:3].sum()}


def plot_ligand_pca_3d(
    df,
    colors=COLORS,
    xlabel="PC2",
    ylabel="PC3",
    zlabel="PC1",
    fig_size=(12, 8),
    label_fontsize=14,
    tick_fontsize=12,
    label_pad=10,
    elev=30,
    azim=75,
    inv_pc_1=False,
    save=False,
):
    sorted_PCA_df = df.sort_values(by=["ligand", "concentration"])

    if inv_pc_1:
        sorted_PCA_df["PC1"] = -sorted_PCA_df["PC1"]

    ligands = sorted_PCA_df["ligand"].unique()
    fig = plt.figure(figsize=fig_size)
    threedee = fig.add_subplot(projection="3d")

    light_grey = (0.94, 0.94, 0.94, 1.0)
    threedee.set_facecolor(light_grey)
    fig.patch.set_facecolor(light_grey)
    threedee.xaxis.set_pane_color(light_grey)
    threedee.yaxis.set_pane_color(light_grey)
    threedee.zaxis.set_pane_color(light_grey)

    for ligand in ligands:
        ligand_subset = sorted_PCA_df[sorted_PCA_df["ligand"] == ligand]
        x = ligand_subset[xlabel]
        y = ligand_subset[ylabel]
        z = ligand_subset[zlabel]
        ctrl_subset = sorted_PCA_df[sorted_PCA_df["ligand"].str.contains("CTRL")]
        if not ctrl_subset.empty:
            x_ctrl = ctrl_subset[xlabel].iloc[0]
            y_ctrl = ctrl_subset[ylabel].iloc[0]
            z_ctrl = ctrl_subset[zlabel].iloc[0]

            threedee.plot(
                [x_ctrl, x.iloc[0]],
                [y_ctrl, y.iloc[0]],
                [z_ctrl, z.iloc[0]],
                c=colors.get(ligand, "black"),
                linestyle="-",
                linewidth=2,
            )

        indices = np.argsort(ligand_subset["concentration"])
        x_sorted = x.iloc[indices]
        y_sorted = y.iloc[indices]
        z_sorted = z.iloc[indices]

        threedee.scatter(
            x,
            y,
            z,
            c=colors.get(ligand, "black"),
            s=275,
            marker="o",
            alpha=1,
            edgecolors="white",
            linewidth=0.3,
        )
        threedee.plot(
            x_sorted,
            y_sorted,
            z_sorted,
            c=colors.get(ligand, "black"),
            linestyle="-",
            linewidth=2,
        )

    threedee.set_xlabel(xlabel, fontsize=label_fontsize, labelpad=label_pad)
    threedee.set_ylabel(ylabel, fontsize=label_fontsize, labelpad=label_pad)
    threedee.set_zlabel(zlabel, fontsize=label_fontsize, labelpad=label_pad)
    threedee.tick_params(axis="both", which="major", labelsize=tick_fontsize)
    threedee.tick_params(axis="z", which="major", labelsize=tick_fontsize)

    threedee.view_init(elev=elev, azim=azim)

    if save:
        plt.savefig(
            SAVE_DIR / "pca_3d_plot.pdf", format="pdf", bbox_inches="tight"
        )

    plt.show()

###########################################################
# Panel B: KM heatmap
###########################################################

def prepare_kmeans_cluster_data(
    fc_all_genes_df,
    data_cols,
    k,
    random_state=42,
    n_init=50,
):
    fc_df = fc_all_genes_df.copy()

    fc_scaled = pd.DataFrame(
        zscore(fc_df.div(fc_df.mean(axis=1), axis=0), axis=1),
        index=fc_df.index,
        columns=fc_df.columns,
    )

    km_final = KMeans(n_clusters=k, random_state=random_state, n_init=n_init)
    cluster_labels = km_final.fit_predict(fc_scaled[data_cols])

    fc_scaled["cluster"] = cluster_labels
    fc_df["cluster"] = cluster_labels

    plot_data = fc_df.loc[fc_scaled.sort_values("cluster").index, data_cols + ["cluster"]].copy()
    plot_data["mean_log2FC"] = plot_data.drop(columns=["cluster"]).mean(axis=1)
    plot_data = plot_data.sort_values(["cluster", "mean_log2FC"], ascending=[True, False])
    plot_data = plot_data.rename(columns={"cluster": "row_cluster"})

    return {
        "fc_df": fc_df,
        "fc_scaled": fc_scaled,
        "plot_data": plot_data,
        "cluster_labels": cluster_labels,
        "model": km_final,
    }


def plot_kmeans_heatmap(plot_data, fc_scaled, k, figsize=(5, 6), cmap="RdYlGn", save=False):
    plot_data = plot_data.copy()

    heatmap_data = plot_data.drop(columns=["row_cluster", "mean_log2FC"], errors="ignore")

    fig, ax = plt.subplots(figsize=figsize)

    sns.heatmap(
        heatmap_data,
        cmap=cmap,
        center=0,
        yticklabels=False,
        xticklabels=True,
        ax=ax,
        cbar_kws={"label": "Z-scored log2FC", "shrink": 0.4},
    )

    cumulative = 0
    cluster_sizes = fc_scaled["cluster"].value_counts().sort_index()

    for cluster_id in range(k):
        cumulative += cluster_sizes.get(cluster_id, 0)
        if cumulative < len(fc_scaled):
            ax.axhline(cumulative, color="white", linewidth=2.5)

    plt.tight_layout()
    if save:
        plt.savefig(
            SAVE_DIR / "kmeans_clustered_heatmap.pdf", format="pdf", bbox_inches="tight"
        )
    plt.show()

###########################################################
# Panel C - Mean Expression per ligand by cluster:
###########################################################


def build_signed_fc_df(df_fc, plot_data):
    df_fc = df_fc.copy()
    sign_vec = np.sign(plot_data["mean_log2FC"])
    df_fc = df_fc.loc[:, plot_data.index].multiply(
        sign_vec,
        axis=1,
    )
    sign_df = pd.concat([sign_vec, plot_data["row_cluster"]+1], axis=1)
    return df_fc, sign_df  

def generate_clusters_dict(fc_ordered, sign_df=None, separate_up_down=False):
    gene_dict = {}
    if separate_up_down and sign_df is not None:
        for cluster in np.unique(sign_df["row_cluster"]):
            cluster_genes = sign_df[sign_df["row_cluster"] == cluster].index
            cluster_data = sign_df.loc[cluster_genes]  # Filter once
            up_genes = cluster_data[cluster_data["mean_log2FC"] > 0].index.tolist()
            down_genes = cluster_data[cluster_data["mean_log2FC"] < 0].index.tolist()
            gene_dict[str(int(cluster)) + "_up"] = up_genes
            gene_dict[str(int(cluster)) + "_down"] = down_genes

    elif sign_df == None:
        gene_dict = {
            int(cluster_id + 1): fc_ordered[fc_ordered["row_cluster"] == cluster_id].index.tolist()
            for cluster_id in sorted(fc_ordered["row_cluster"].unique())
        }
    return gene_dict

def get_ligands_df_dict(fc_df):
    fc_df = add_df_ligand_names_and_concentrations_columns(fc_df)

    ligands_df_dict = {}
    for ligand in fc_df["ligand"].unique():
        df = fc_df[fc_df["ligand"] == ligand].copy()
        df = df.sort_values("concentration")  # Sort by concentration ascending
        df["ordinal"] = range(1, len(df) + 1)  # Assign ordinal numbers
        ligands_df_dict[ligand] = df

    return ligands_df_dict


def plot_clusters(ligands_df_dict, cluster_dict, colors=CLUSTERS_COLORS, save=False):
    n_clusters = len(cluster_dict)
    n_cols = 3
    n_rows = math.ceil(n_clusters / n_cols)

    fig, ax = plt.subplots(
        nrows=n_rows,
        ncols=n_cols,
        figsize=(5 * n_cols, 4 * n_rows),
    )

    ax = ax.flatten()

    for i, (cl, gene_list) in enumerate(cluster_dict.items()):
        gene_list = gene_list
        current_ax = ax[i]

        for ligand, lig_df in ligands_df_dict.items():
            try:
                filt_lig_df = lig_df[gene_list].mean(axis=1)
                filt_lig_df = pd.concat([pd.Series([0], index=["CTRL"]), filt_lig_df])
            except TypeError:
                print(f"Gene {gene_list} not found in ligand {ligand}. Skipping.")
                continue

            x_values = range(0, 8)

            current_ax.plot(
                x_values,
                filt_lig_df,
                marker="o",
                label=ligand,
                color=colors[ligand],
                linewidth=5,
                markersize=14,
                alpha=0.7,
                #remove stroke:
                markeredgewidth=0,
            )

        current_ax.set_title(f"Cluster_{cl} (n={len(gene_list)})")
        # get current axis limits:
        current_ylim = current_ax.get_ylim()
        if (np.abs(current_ylim) > 2).any():
            current_ax.set_ylim(-5, 5.5)
        else:
            current_ax.set_ylim(-2.5, 2.5)
        current_ax.axhline(y=0, color="black", linestyle="-", linewidth=1)

        current_ax.yaxis.set_major_locator(plt.MaxNLocator(nbins=5))

        if cl >= (n_rows - 1) * n_cols:
            current_ax.set_xlabel("Concentration Level")
        if cl % n_cols == 0:
            current_ax.set_ylabel("Log2FC")

    for j in range(n_clusters, len(ax)):
        fig.delaxes(ax[j])

    handles, labels = ax[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper right", bbox_to_anchor=(1.1, 1.02))

    #limit x-axis to 0-7:
    for current_ax in ax[:n_clusters]:
        current_ax.set_ylim(-0.5, 2)

    plt.tight_layout()

    if save:
        plt.savefig(
            SAVE_DIR / "clusters_plot.pdf", format="pdf", bbox_inches="tight"
        )

    plt.show()

################################################################################
# Supplementary Panel A:  Pair-wise Pearson correlations between the ligands:
################################################################################


def plot_corr_matrix(fc_all_genes_df, save=False):
    corr = fc_all_genes_df.corr()
    # cluster rows and columns
    row_linkage = linkage(
        corr, method="average", metric="euclidean", optimal_ordering=True
    )
    col_linkage = linkage(
        corr.T, method="average", metric="euclidean", optimal_ordering=True
    )
    row_order = leaves_list(row_linkage)
    col_order = leaves_list(col_linkage)
    corr = corr.iloc[row_order, col_order]

    plt.figure(figsize=(6.5, 5))
    sns.heatmap(
        corr,
        annot=False,
        cmap="YlOrRd",
        linewidths=0.5,
        cbar_kws={"shrink": 0.5},
    )

    plt.yticks(
        rotation=0,
    )
    if save:
        plt.savefig(
            SAVE_DIR / "supplementary" / "correlation_matrix.pdf", format="pdf", bbox_inches="tight"
        )

    plt.show()


################################################################################
# Supplementary Panel B:  Elbow method plot for k-means clustering 
################################################################################

def plot_kmeans_elbow(
    X,
    k_range=range(2, 16),
    random_state=42,
    n_init=20,
    figsize=(4, 2.5),
    save=False,
):
    inertias = []

    for k in k_range:
        km = KMeans(n_clusters=k, random_state=random_state, n_init=n_init)
        labels = km.fit_predict(X)
        inertias.append(km.inertia_)

    metrics_df = pd.DataFrame(
        {
            "k": list(k_range),
            "inertia": inertias,
        }
    )

    fig, axes = plt.subplots(1, 1, figsize=figsize)

    axes.plot(metrics_df["k"], metrics_df["inertia"], "o-", color="steelblue", linewidth=2)
    axes.set_xlabel("Number of clusters (k)")
    axes.set_ylabel("Inertia")
    axes.set_title("Elbow Method")
    axes.set_xticks(list(k_range))

    plt.tight_layout()

    if save:
        plt.savefig(
            SAVE_DIR / "supplementary" / "kmeans_elbow_plot.pdf", format="pdf", bbox_inches="tight"
        )

    plt.show()