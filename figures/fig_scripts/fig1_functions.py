import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import umap
import matplotlib.colors as mcolors

from matplotlib.lines import Line2D
from pathlib import Path
from sklearn.decomposition import PCA
from scipy.cluster.hierarchy import linkage, leaves_list

from src.utils.utils import add_df_ligand_names_and_concentrations_columns, set_style

set_style()

########### Define colors and ligands ##########

BASE_DIR = Path(__file__).resolve().parent.parent.parent # Resolve project root from figures/fig_scripts/
SAVE_DIR = BASE_DIR / "figures" / "panels" / "figure_1"

os.makedirs(SAVE_DIR, exist_ok=True)
os.makedirs(SAVE_DIR / "supplementary", exist_ok=True)

COLORS = {
    "CTRL_1": "gray",
    "BMP10": "blue",
    "BMP4": "orange",
    "BMP6": "green",
    "BMP9": "red",
    "GDF5": "purple",
    "TGFb1": "brown",
}

LIGANDS = ["BMP4", "BMP6", "BMP9", "BMP10", "TGFb1", "GDF5"]

###############################################################
# Panel B: BMP4 significant genes count per concentration level
###############################################################

def plot_fdr_gene_counts(ligand, sig_df, colors=COLORS, save=False):
    bar_width = 0.85

    fig, ax = plt.subplots(figsize=(6, 4))

    data = sig_df.loc[sig_df.index.str.contains(ligand)].sum(axis=1)
    data_df = data.to_frame(name="count")
    data_df = add_df_ligand_names_and_concentrations_columns(data_df)

    for i, (cond, count) in enumerate(data.items()):
        ax.bar(
            i,
            count,
            color=colors[ligand],
            width=bar_width,
            label=ligand if i == 0 else "",
        )

    ax.set_xticks(range(len(data_df)))
    ax.set_xticklabels(data_df["concentration"], fontsize=10, rotation=0)
    ax.set_xlabel("Concentration", fontsize=14)
    ax.set_ylabel("Number of\n Significant Genes", fontsize=14)

    # ax.yaxis.grid(True, linestyle="--", alpha=0.7)
    ax.xaxis.grid(False)

    if save:
        plt.savefig(
            SAVE_DIR / f"{ligand}_fdr_gene_count.pdf",
            format="pdf",
            bbox_inches="tight",
        )

    plt.tight_layout()
    plt.show()

###############################################################
# Panel C: BMP4 responsive genes log2FC across concentrations
###############################################################

def cluster_heatmap_data(fold_change_filt, method="average", metric="euclidean", reverse_order=False):
    filtered_data = fold_change_filt[fold_change_filt.sum(axis=1) != 0.0].copy()
    filtered_data.replace([np.inf, -np.inf, np.nan], 0, inplace=True)
    linkage_matrix = linkage(
        filtered_data.T, method=method, metric=metric, optimal_ordering=True
    )
    row_order = leaves_list(linkage_matrix)
    if reverse_order:
        row_order = row_order[::-1]
    reordered_data = filtered_data.T.iloc[row_order]
    gene_order = reordered_data.index.tolist()

    return reordered_data, gene_order


def plot_heatmap(
    ligand,
    reordered_data,
    save=False,
):
    plt.figure(figsize=(2.5, 8))
    ax = sns.heatmap(
        reordered_data,
        cmap="RdYlGn",
        center=0,
        cbar=True,
        xticklabels=True,
        yticklabels=False,
        # cbar_kws={"shrink": 0.5},
    )

    plt.xticks(fontsize=8)
    plt.xlabel("Concentration")
    plt.ylabel("Genes")

    if save:
        plt.savefig(
            SAVE_DIR / f"{ligand}_heatmap.pdf", format="pdf", bbox_inches="tight"
        )

    plt.show()

###############################################################
# Panel D: BMP4 significant genes correlation matrix
###############################################################

def plot_corr_heatmap(fold_change_filt, gene_order, method="spearman", save=False):
    corr_matrix = fold_change_filt.corr(method=method)
    corr_matrix = corr_matrix.reindex(gene_order, axis=0).reindex(gene_order, axis=1)

    plt.figure(figsize=(10, 8))  # Adjust the figure size as needed
    sns.heatmap(corr_matrix, annot=False, cmap="coolwarm", fmt=".2f")

    # add gray lines between squares
    for i in range(len(corr_matrix) + 1):
        plt.axhline(i, color="gray", lw=0.25)
        plt.axvline(i, color="gray", lw=0.25)

    plt.title(f"{method.capitalize()} Correlation Heatmap")
    if save:
        plt.savefig(
            SAVE_DIR / "BMP4_corr_heatmap.pdf", format="pdf", bbox_inches="tight"
        )
    plt.show()

###############################################################
# Panel E: BMP4 responsive genes normalized response curves
###############################################################

def plot_fractional_expression(
    ligand,
    norm_exp,
    concentrations,
    colors=COLORS,
    save=False,
    supp=False,
    plot_median=True,
):
    alphas = {
        "BMP10": 0.03,
        "BMP4": 0.1,
        "BMP6": 0.04,
        "BMP9": 0.04,
        "GDF5": 0.05,
        "TGFb1": 0.02,
    }

    concentrations.sort()
    concentrations_names = concentrations.astype(str)
    x_positions = np.arange(len(concentrations_names))

    fig, ax = plt.subplots(figsize=(7, 5))

    genes_to_plot = norm_exp.columns

    line_color = "gray" if ligand == "BMP10" else "black"
    if plot_median:
        norm_exp_mean = norm_exp.median(axis=1)

        ax.plot(
            concentrations_names,
            norm_exp_mean,
            color=line_color,
            alpha=1,
            linewidth=2.5,
            label="Mean Expression",
            marker="o",
            markersize=8,
            zorder=5,
            markerfacecolor=colors[ligand],
            markeredgewidth=1,
        )

    for gene_name in genes_to_plot:
        ax.plot(
            concentrations_names,
            norm_exp.T.loc[gene_name],
            color=colors[ligand],
            alpha=alphas[ligand],
            linewidth=3,
            label="Upregulated Genes",
        )

    ax.set_xlabel(f"{ligand} Concentration (ng/ml)", fontsize=18)
    ax.set_ylabel("Fractional Expression", fontsize=18)
    ax.set_xticks(x_positions)
    ax.set_xticklabels(concentrations_names, rotation=45, fontsize=16, ha="right")
    ax.tick_params(axis="y", labelsize=16)
    ax.set_ylim(-0.5, 1.75)

    plt.tight_layout()

    if supp and save:
        os.makedirs(SAVE_DIR / "supplementary", exist_ok=True)
        plt.savefig(
            SAVE_DIR / "supplementary" / f"{ligand}_norm_sig_genes.pdf",
            format="pdf",
            bbox_inches="tight",
        )

    elif save:
        plt.savefig(
            SAVE_DIR / f"{ligand}_norm_sig_genes.pdf",
            format="pdf",
            bbox_inches="tight",
        )

    plt.show()


###############################################################
# Panel F: BMP4 population level PCA
###############################################################

def plot_mean_pca(ligand, lig_model, colors=COLORS, ylim=None, save=False):
    cmaps = {
        "CTRL_1": "gray",
        "BMP10": "Blues",
        "BMP4": "Oranges",
        "BMP6": "Greens",
        "BMP9": "Reds",
        "GDF5": "Purples",
        "TGFb1": "YlOrBr",
    }

    pca_results = lig_model.pca_mean_fitted_df.copy()

    if pca_results["PC1"].iloc[0] > pca_results["PC1"].iloc[-1]:
        pca_results.iloc[:, :-2] = pca_results.iloc[:, :-2] * (-1)

    ev_1, ev_2 = lig_model.exp_var_dict["PC1"], lig_model.exp_var_dict["PC2"]

    plt.rcParams["figure.figsize"] = [5, 3]
    plt.rcParams["figure.autolayout"] = True

    f, ax = plt.subplots()

    # Create an array of values from 0 to 1 for the 8 points
    color_values = np.linspace(0, 1, len(pca_results))

    # Plot all points at once with YlOrBr colormap
    points = ax.scatter(
        pca_results["PC1"],
        pca_results["PC2"],
        c=color_values,  # Use these values for coloring
        cmap=cmaps[ligand],  # Use YlOrBr colormap
        s=75,
        edgecolor="black",
        linewidth=0.5,
    )

    ctrl_points = ax.scatter(
        pca_results["PC1"].iloc[0],
        pca_results["PC2"].iloc[0],
        color="gray",
        s=90,
        edgecolor="black",
        linewidth=0.5,
    )

    # Connecting lines for increasing concentrations
    ax.plot(
        pca_results["PC1"],
        pca_results["PC2"],
        color="gray",
        linestyle="--",
        alpha=0.5,
    )

    legend_elements = [
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            label=ligand,
            markersize=10,
            markerfacecolor=colors[ligand],
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            label="Control",
            markersize=10,
            markerfacecolor="gray",
        ),
    ]
    ax.legend(handles=legend_elements)
    ax.set_xlim(pca_results["PC1"].min() * 1.5, pca_results["PC1"].max() * 1.5)
    ax.set_ylim((-6, 6) if ylim is None else ylim)
    ax.set_title(f"PCA for {ligand} Concentrations")
    ax.set_xlabel(f"PC1 ({ev_1:.2f}% variance explained)")
    ax.set_ylabel(f"PC2 ({ev_2:.2f}% variance explained)")
    plt.grid(True, linestyle="--", alpha=0.5)

    if save:
        plt.savefig(SAVE_DIR / f"{ligand}_mean_pca.pdf", format="pdf")

    plt.show()


###############################################################
# ### Supplementary panel A : ligands 6HR vs 3HR UMAP
###############################################################


def filter_single_cells(df_exp1, sig_genes_list, max_index_len=10):
    singles = [s for s in df_exp1.index.unique() if len(s) < max_index_len]
    return df_exp1.loc[df_exp1.index.isin(singles), sig_genes_list]


def log2_transform(df, pseudocount=1.0):
    return np.log2(df + pseudocount)


def compute_umap_embedding(
    df_log,
    n_pca_max=50,
    n_neighbors=15,
    min_dist=0.1,
    metric="correlation",
    pca_random_state=0,
    umap_random_state=42,
):
    x = df_log.fillna(0)
    n_pca = min(n_pca_max, x.shape[1])

    pca = PCA(n_components=n_pca, random_state=pca_random_state)
    x_pca = pca.fit_transform(x.values)

    reducer = umap.UMAP(
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        metric=metric,
        random_state=umap_random_state,
    )
    emb = reducer.fit_transform(x_pca)

    emb_df = pd.DataFrame(emb, index=x.index, columns=["UMAP1", "UMAP2"])
    emb_df["time"] = (
        emb_df.index.to_series().str.extract(r"(^\d+)HR", expand=False).fillna("unknown")
    )

    return emb_df, pca, reducer


def add_ligand_column(emb_df, sep="-", position=1, fallback="unknown"):
    out = emb_df.copy()
    if "ligand" not in out.columns:
        parts = out.index.to_series().str.split(sep)
        out["ligand"] = parts.str[position].fillna(fallback)
    return out


def make_non_overlapping_ligand_palette(ligands, time_palette_list, base_palette="Set2", darken_factor=0.65):
    pal_lig_list = sns.color_palette(base_palette, n_colors=len(ligands))
    time_hex = {mcolors.to_hex(c) for c in time_palette_list}

    fixed_lig_cols = []
    for col in pal_lig_list:
        hexc = mcolors.to_hex(col)
        if hexc in time_hex:
            col = tuple(max(0, c * darken_factor) for c in col)
        fixed_lig_cols.append(col)

    return dict(zip(ligands, fixed_lig_cols))


def _legend_handles(categories, palette, marker_size=9):
    return [
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor=palette[c],
            markersize=marker_size,
        )
        for c in categories
    ]


def plot_umap_by_time_and_ligand(
    emb_df,
    fig_size=(12, 5),
    point_size=10,
    alpha=0.6,
    time_palette_name="tab10",
    ligand_palette_name="Set2",
    save=False
):
    fig, axes = plt.subplots(1, 2, figsize=fig_size, sharex=True, sharey=True)

    # Left panel: time
    ax = axes[0]
    times = emb_df["time"].unique().tolist()
    pal_time_list = sns.color_palette(time_palette_name, n_colors=len(times))
    pal_time = dict(zip(times, pal_time_list))

    sns.scatterplot(
        data=emb_df,
        x="UMAP1",
        y="UMAP2",
        hue="time",
        palette=pal_time,
        s=point_size,
        alpha=alpha,
        ax=ax,
        legend=False,
        edgecolor='none',
    )
    ax.set_title("UMAP colored by time")
    handles_time = _legend_handles(times, pal_time)
    labels_time = [f"{t}HR" if str(t).isdigit() else str(t) for t in times]
    ax.legend(handles=handles_time, labels=labels_time, title="Time", loc="best", frameon=True)

    # Right panel: ligand
    ax = axes[1]
    ligands = emb_df["ligand"].unique().tolist()
    
    # Use specific colors for each ligand
    ligand_colors = {
        "BMP4": "#ffa12f",
        "BMP6": "#a6d854",
        "BMP9": "#d85d54",
        "BMP10": "#6bafda",
        "GDF5": "#c38ae7",
        "CTRL_1": "gray",
        "CTRL": "gray",
    }
    pal_lig = {lig: ligand_colors.get(lig, "gray") for lig in ligands}

    sns.scatterplot(
        data=emb_df,
        x="UMAP1",
        y="UMAP2",
        hue="ligand",
        palette=pal_lig,
        s=point_size,
        alpha=alpha,
        ax=ax,
        legend=False,
        edgecolor='none',
    )
    ax.set_title("UMAP colored by ligand")
    handles_lig = _legend_handles(ligands, pal_lig)
    ax.legend(handles=handles_lig, labels=ligands, title="Ligand", loc="best", frameon=True)

    plt.tight_layout()

    if save:
        plt.savefig(
            SAVE_DIR / "supplementary" / "umap_time_ligand.pdf",
            format="pdf",
            bbox_inches="tight",
        )

    plt.show()

def run_umap_panel_from_df_exp1(df_exp1, sig_genes_list, save=False):
    df_exp1_filt = filter_single_cells(df_exp1, sig_genes_list)
    df_log = log2_transform(df_exp1_filt, pseudocount=1.0)
    emb_df, pca, _ = compute_umap_embedding(df_log)
    emb_df = add_ligand_column(emb_df)
    plot_umap_by_time_and_ligand(emb_df, save=save)
    return {
        "df_exp1_filt": df_exp1_filt,
        "df_log": df_log,
        "emb_df": emb_df,
        "pca": pca,
    }

def get_fc_values(df, log_addition=1e-3, control_pattern="CTRL"):
    df_mean = df.groupby(df.index).mean()
    df_ctrl = df_mean[df_mean.index.str.contains(control_pattern)]
    mean_df_ctrl_mean = df_ctrl.mean()
    mean_df_ligand_no_ctrl = df_mean[~df_mean.index.str.contains(control_pattern)]

    df_fc = np.log2(
        (mean_df_ligand_no_ctrl + log_addition) / (mean_df_ctrl_mean + log_addition)
    )
    df_fc = df_fc.loc[:, ~df_fc.columns.duplicated()]
    return df_fc


def build_timepoint_fc_table(
    df_exp1_singles,
    times=("3HR", "6HR"),
    log_addition=1e-3,
    row_order=None,
):
    fc_frames = []
    for t in times:
        df_t = df_exp1_singles[df_exp1_singles.index.str.contains(t)]
        fc_frames.append(get_fc_values(df_t, log_addition=log_addition))

    df_filt = pd.concat(fc_frames)

    if row_order is not None:
        missing = [idx for idx in row_order if idx not in df_filt.index]
        if missing:
            raise ValueError(f"Missing requested rows in fold-change table: {missing}")
        df_filt = df_filt.loc[row_order]

    return df_filt


def plot_reg_3hr_6hr(ax, sample_3hr, sample_6hr, title=None, point_color="skyblue", point_size=20):
    sns.scatterplot(x=sample_3hr, y=sample_6hr, s=point_size, ax=ax, color=point_color, edgecolor='none',)

    pearson_corr = np.corrcoef(sample_3hr, sample_6hr)[0, 1]

    lims = [
        min(ax.get_xlim()[0], ax.get_ylim()[0]),
        max(ax.get_xlim()[1], ax.get_ylim()[1]),
    ]
    ax.plot(lims, lims, "k--", alpha=0.75, zorder=0)
    ax.set_aspect("equal")

    ax.set_title(title, fontsize=12)
    ax.set_xlabel("3HR-sample")
    ax.set_ylabel("6HR-sample")

    x_min, x_max = (
        min(sample_3hr.min(), sample_6hr.min()),
        max(sample_3hr.max(), sample_6hr.max()),
    )
    y_min, y_max = x_min, x_max

    ax.set_xlim(x_min + x_min * 0.2, x_max + x_max * 0.2)
    ax.set_ylim(y_min + y_min * 0.2, y_max + y_max * 0.2)

    ax.text(
        0.05,
        0.95,
        f"Pearson: {pearson_corr:.2f}",
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=10,
        bbox=dict(
            boxstyle="round,pad=0.2",
            facecolor="white",
            alpha=0.8,
            edgecolor="none",
        ),
    )
    ax.legend(["Gene log2FC", "y=x line"], loc="lower right")
    return pearson_corr


def plot_3hr_6hr_regression_grid(
    df_filt,
    ligands=("BMP4", "BMP9", "BMP10", "GDF5"),
    time_3hr="3HR",
    time_6hr="6HR",
    fig_size=(6, 6),
    save=False,
    save_name="3hr_vs_6hr_regression_sig_genes.pdf",
):
    fig, axes = plt.subplots(2, 2, figsize=fig_size)
    axes = axes.flatten()

    ligand_palette = {
        "BMP4": "#ffa12f",
        "BMP6": "#a6d854",
        "BMP9": "#d85d54",
        "BMP10": "#6bafda",
        "GDF5": "#c38ae7",
        "CTRL_1": "gray",
        "CTRL": "gray",
    }

    corr_by_ligand = {}
    for i, ligand in enumerate(ligands):
        row_3hr = f"{time_3hr}-{ligand}"
        row_6hr = f"{time_6hr}-{ligand}"

        if row_3hr not in df_filt.index or row_6hr not in df_filt.index:
            raise ValueError(f"Missing rows for ligand {ligand}: {row_3hr}, {row_6hr}")

        sample_3hr = df_filt.loc[row_3hr]
        sample_6hr = df_filt.loc[row_6hr]
        point_color = ligand_palette[ligand] if ligand_palette is not None else "skyblue"
        corr = plot_reg_3hr_6hr(
            axes[i],
            sample_3hr,
            sample_6hr,
            title=ligand,
            point_color=point_color,
        )
        corr_by_ligand[ligand] = corr

    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()

    if save:
        os.makedirs(SAVE_DIR / "supplementary", exist_ok=True)
        plt.savefig(
            SAVE_DIR / "supplementary" / save_name,
            format="pdf",
            bbox_inches="tight",
        )

    plt.show()
    return 

