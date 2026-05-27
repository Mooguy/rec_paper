import os
from pathlib import Path
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tqdm
import matplotlib.colors as mcolors
import matplotlib.cm as cm
import pingouin as pg
import warnings

from scipy.stats import spearmanr
from scipy.stats import ks_2samp
from statsmodels.stats.multitest import multipletests

BASE_DIR = Path(__file__).resolve().parent.parent.parent 
SAVE_DIR = BASE_DIR / "figures" / "panels" / "figure_4"

os.makedirs(SAVE_DIR, exist_ok=True)

from src.utils.utils import add_df_ligand_names_and_concentrations_columns, set_style

set_style()

##############################################################################
# Variables
##############################################################################

colors = {
    "CTRL_1": "gray",
    "BMP10": "blue",
    "BMP4": "orange",
    "BMP6": "green",
    "BMP9": "red",
    "GDF5": "purple",
    "TGFb1": "brown",
}

palletes = {
    "BMP10": "Blues",
    "BMP4": "Oranges",
    "BMP6": "Greens",
    "BMP9": "Reds",
    "GDF5": "Purples",
    "TGFb1": "YlOrBr",
}

##############################################################################
# Panel (A): BMP4 Single Cell PC1 vs. Concentration Boxplot
##############################################################################

def plot_pc1_concentration_boxplot(
    df, colors=palletes, save=False, ylim=None, reverse=False
):
    # Copy and categorize
    df = df.copy()
    df["concentration"] = pd.Categorical(df["concentration"])
    ligand = df["ligand"].mode()[0]

    if reverse:
        df["PC1"] = df["PC1"] * (-1)

    color_pallete = colors[ligand]

    # Get categories and create palette
    categories = df["concentration"].cat.categories
    palette = sns.color_palette(color_pallete, n_colors=len(categories))
    color_dict = dict(zip(categories, palette))

    # Set plot style
    sns.set(style="whitegrid")
    fig, ax = plt.subplots(figsize=(7, 5))

    # Stripplot manually by category
    for i, cat in enumerate(categories):
        y_vals = df[df["concentration"] == cat]["PC1"]
        x_vals = [i] * len(y_vals)
        ax.scatter(
            x=pd.Series(x_vals) + np.random.normal(0, 0.05, size=len(y_vals)),
            y=y_vals,
            color=color_dict[cat],
            alpha=0.4,
            edgecolor="none",
        )

    # Boxplot
    sns.boxplot(
        data=df,
        x="concentration",
        y="PC1",
        showcaps=True,
        boxprops={"facecolor": "none", "edgecolor": "grey", "alpha": 0.8},
        whiskerprops={"color": "grey", "alpha": 0.8},
        capprops={"color": "grey", "alpha": 0.8},
        medianprops={"color": "grey", "alpha": 0.8},
        flierprops={"marker": "o", "alpha": 0.5},
        linewidth=1.5,
        width=0.7,
        ax=ax,
    )

    if ylim is not None:
        ax.set_ylim(ylim)
    ax.set_xticks(range(len(categories)))
    ax.set_xticklabels(categories, rotation=45, ha="right", fontsize=16, color="black")
    # Ensure ticks are drawn and styled
    ax.tick_params(
        axis='both', 
        which='major', 
        length=6, 
        width=2, 
        colors='black', 
        bottom=True,  # Explicitly show bottom ticks
        left=True     # Explicitly show left ticks
    )
    # ax.set_title("PC1 by Concentration")
    ax.set_ylabel("Perception Score", color="black", fontsize=20)
    ax.set_xlabel("BMP4 Concentration(ng/ml)", color="black", fontsize=20)
    plt.tight_layout()

    if save:
        plt.savefig(
            SAVE_DIR / f"{ligand}_PC1_jitter_boxplot.pdf",
            format="pdf",
            bbox_inches="tight",
        )

    plt.show()

##############################################################################
# Panel B: BMP4 PC1 vs H2BCITRINE
##############################################################################


def get_pca_df_with_h2b(model, h2b_vec):
    pca_df = model.pca_sc_fitted_df.copy()
    h2b_vec_bmp9 = h2b_vec.loc[model.df.index.str.contains(f"{model.ligand}|CTRL_1"),]
    pca_df["h2b"] = h2b_vec_bmp9
    pca_df["log_h2b"] = np.log2(pca_df["h2b"] + 1)
    #remove any rows with 0 h2b expression:
    pca_df = pca_df[pca_df["h2b"] > 0].copy()   
    return pca_df


def plot_pca_vs_h2b(
    pca_df, model_name, log_h2b=False, reverse=False, save=False, plot_mean=False
):
    # Set y-axis column based on log flag
    y_col = "log_h2b" if log_h2b else "h2b"

    # Handle PC1 direction reversal
    x_col = pca_df["PC1"] * (-1) if reverse else pca_df["PC1"]

    # Get colormap based on ligand mode
    cmap = plt.get_cmap(palletes[pca_df["ligand"].mode()[0]])

    # Define unique discrete concentration values
    discrete_values = sorted(pca_df["concentration"].unique())

    # Create boundary normalization for discrete colors
    norm = mcolors.BoundaryNorm(
        boundaries=[*discrete_values, discrete_values[-1] + 1],
        ncolors=cmap.N,
    )

    ligand = pca_df["ligand"].mode()[0]

    plt.figure(figsize=(8.5, 4.5))

    # Plot raw data points
    sns.scatterplot(
        x=x_col,
        y=pca_df[y_col],
        c=pca_df["concentration"],
        cmap=cmap,
        norm=norm,
        legend=None,
        edgecolors="gray",
        linewidth=0.4,
        alpha=0.4,
        marker="o",
        s=50,
    )

    # Calculate Spearman correlation
    spear_value, p_value_spearman = spearmanr(x_col, pca_df[y_col])

    if plot_mean:
        # Group by index to find mean coordinates
        pc1_mean_vec = (
            pca_df.loc[:, ["PC1", "h2b", "log_h2b"]].groupby(pca_df.index).mean()
        )
        # Reorder to ensure control group is first
        pc1_mean_vec = pc1_mean_vec.loc[
            ["CTRL_1"] + [ind for ind in pc1_mean_vec.index if ind != "CTRL_1"]
        ]
        
        # Extract and adjust PC1 means
        mean_pc1 = pc1_mean_vec["PC1"] * (-1) if reverse else pc1_mean_vec["PC1"]
        # Extract target H2B means
        mean_h2b = pc1_mean_vec[y_col]

        # Plot group means as solid black dots
        plt.scatter(
            mean_pc1,
            mean_h2b,
            color="black",
            edgecolors="white",
            linewidth=1.0,
            marker="o",
            s=50,
        )

    # Create a separate mappable for full opacity colorbar
    sm = cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])

    # Add the colorbar
    cbar = plt.colorbar(sm, ax=plt.gca(), orientation="vertical")
    cbar.set_ticks(discrete_values)
    cbar.ax.set_yticklabels([str(v) for v in discrete_values])

    plt.title(f"PC1 vs H2B for {model_name}")

    # Annotate plot with correlation coefficient
    plt.text(
        0.1,
        0.9,
        f"ρ: {spear_value:.3f}",
        transform=plt.gca().transAxes,
        fontsize=12,
        verticalalignment="top",
    )
    plt.xlabel("PC1", fontsize=14)
    plt.ylabel(
        "H2BCITRINE Log Expression" if log_h2b else "H2BCITRINE Expression",
        fontsize=14,
    )

    # Force tick markers to appear on both X and Y axes
    plt.gca().tick_params(axis="both", which="both", bottom=True, left=True, direction="out")

    if save:
        plt.savefig(
            SAVE_DIR / f"{ligand}_PC1_vs_H2B.pdf",
            format="pdf",
            bbox_inches="tight",
        )

    plt.show()


##############################################################################
# Panel C: BMP4 Gene partial correlation with PC1 vs. KS test p-value
##############################################################################

##############################################################################
# Data preparation
##############################################################################

def filter_off_genes(norm_df_lig, lig_pc1):
    df = norm_df_lig.copy()
    df["PC1"] = lig_pc1.values
    df["conc"] = df.index.to_numpy()

    gene_cols = [c for c in df.columns if c not in ["PC1", "conc"]]
    keep_genes = [g for g in gene_cols if df[g].sum() > 0]

    out = df.loc[:, keep_genes + ["PC1", "conc"]].copy()
    return out, lig_pc1, keep_genes

def prepare_ligand_data_from_model(lig_model, ctrl_conc="CTRL_1"):
    ligand = lig_model.ligand
    if not ctrl_conc:
        ctrl_conc = lig_model.get_ctrl_for_ligand()

    lig_pc1 = lig_model.pca_sc_fitted_df["PC1"].copy()
    
    print("Number of genes before filtering significant genes:", len(lig_model.df.columns))

    gene_cols = [
        g for g in lig_model.df.columns
        if g not in lig_model.all_genes_list # and "MT-" not in g
    ]

    print("Number of genes after filtering significant genes:", len(gene_cols))

    norm_df_lig = lig_model.df.loc[
        lig_model.df.index.str.contains(f"{ligand}|{ctrl_conc}"),
        gene_cols,
    ].copy()

    print("Number of genes before filtering off genes with zero expression:", norm_df_lig.shape[1])

    df_pc1_lig, lig_pc1, keep_genes = filter_off_genes(norm_df_lig, lig_pc1)

    print("Number of genes after filtering off genes with zero expression:", len(keep_genes))
    return df_pc1_lig, lig_pc1, keep_genes


##############################################################################
# KS test helpers
##############################################################################

def run_ks_test(zero_df, mid_df, gene):
    data_mid = mid_df[gene].to_numpy().ravel()
    data_zero = zero_df[gene].to_numpy().ravel()
    return ks_2samp(data_mid, data_zero)


def compute_ks_for_ligand(norm_df_lig, ctrl_conc="CTRL", genes=None, show_progress=True):
    if genes is None:
        genes = norm_df_lig.columns.tolist()

    lig_concs = norm_df_lig.index.unique().to_list()
    if ctrl_conc in lig_concs:
        lig_concs.remove(ctrl_conc)

    lig_zero = norm_df_lig[norm_df_lig.index.str.contains(ctrl_conc)].copy()
    out = {}

    for conc in lig_concs:
        lig_conc = norm_df_lig[norm_df_lig.index.str.contains(conc)].copy()

        if lig_zero.empty or lig_conc.empty:
            warnings.warn(f"Empty group for conc {conc}, skipping KS tests.")
            continue

        iterator = tqdm.tqdm(genes) if show_progress else genes
        rows = []

        for gene in iterator:
            ks_stat, p_value = run_ks_test(lig_zero, lig_conc, gene)
            rows.append((gene, ks_stat, p_value))

        ks_test_results_df = pd.DataFrame(rows, columns=["Gene", "KS_Stat", "P_Value"]).set_index("Gene")
        ks_test_results_df["Adj_P_Value"] = multipletests(
            ks_test_results_df["P_Value"].values, method="fdr_bh"
        )[1]

        out[conc] = ks_test_results_df

    return out


def process_all_ligands_ks(ligand_dict, ctrl_conc="CTRL", genes=None, show_progress=True):
    out = {}
    for ligand, norm_df_lig in ligand_dict.items():
        if show_progress:
            print(f"Computing KS tests for {ligand}...")
        out[ligand] = compute_ks_for_ligand(
            norm_df_lig,
            ctrl_conc=ctrl_conc,
            genes=genes,
            show_progress=show_progress,
        )
    return out 


##############################################################################
# Partial correlation helpers
##############################################################################

def get_partial_corr(data, col_B, method, covar=None, col_A="PC1"):
    if covar is None:
        covar = []
    return pg.partial_corr(
        data=data,
        x=col_A,
        y=col_B,
        covar=covar,
        method=method,
    )


def get_ligand_category_and_concentration(norm_df):
    id_conc_df = add_df_ligand_names_and_concentrations_columns(norm_df)
    id_conc_df_mid = (
        pd.DataFrame(id_conc_df["concentration"].unique(), columns=["concentration"])
        .assign(idx=id_conc_df.index.unique())
        .sort_values(by="concentration")
    )
    return id_conc_df_mid["idx"].to_list()


def prepare_pc1_gene_df(norm_df_lig, lig_pc1, category_func=get_ligand_category_and_concentration):
    df = norm_df_lig.copy()
    df["PC1"] = lig_pc1
    df["conc"] = df.index.to_numpy()

    lig_conc_list = category_func(norm_df_lig)
    df["cat"] = pd.Categorical(df["conc"], categories=lig_conc_list, ordered=False)

    dummies = pd.get_dummies(df["cat"])
    if len(df["cat"].cat.categories) > 0:
        dummies = dummies[df["cat"].cat.categories[1:]]

    df = pd.concat([df, dummies], axis=1).drop(columns=["conc", "cat"])

    covar_cols = dummies.columns.tolist()
    non_gene_cols = set(["PC1"] + covar_cols)
    gene_cols = [c for c in df.columns if c not in non_gene_cols]

    return df, gene_cols, covar_cols


def _safe_fdr_bh(pvals):
    pvals = pd.Series(pvals, dtype=float)
    out = pd.Series(np.nan, index=pvals.index, dtype=float)
    mask = pvals.notna()
    if mask.any():
        out.loc[mask] = multipletests(pvals.loc[mask].values, method="fdr_bh")[1]
    return out


def compute_partial_corrs(df_pc1_genes, gene_cols, covar_cols, show_progress=True):
    cols = [
        "partial_corr_spear",
        "partial_corr_spear_pval",
        "partial_corr_pear",
        "partial_corr_pear_pval",
    ]
    res_df = pd.DataFrame(index=gene_cols, columns=cols, dtype=float)
    
    # Reset index to avoid duplicate index issues with pingouin
    df_pc1_genes = df_pc1_genes.reset_index(drop=True)

    iterator = tqdm.tqdm(gene_cols) if show_progress else gene_cols

    for gene in iterator:
        if gene == "PC1":
            continue

        r_s, p_s = np.nan, np.nan
        try:
            out_s = get_partial_corr(
                data=df_pc1_genes, col_B=gene, method="spearman", covar=covar_cols
            ).iloc[0]
            r_s, p_s = out_s["r"], out_s["p_val"]
        except Exception as e:
            print(f"Spearman error for {gene}: {e}")
            pass

        r_p, p_p = np.nan, np.nan
        try:
            out_p = get_partial_corr(
                data=df_pc1_genes, col_B=gene, method="pearson", covar=covar_cols
            ).iloc[0]
            r_p, p_p = out_p["r"], out_p["p_val"]
        except Exception as e:
            print(f"Pearson error for {gene}: {e}")
            pass

        res_df.loc[gene] = [r_s, p_s, r_p, p_p]

    res_df["partial_corr_pear_pval"] = _safe_fdr_bh(res_df["partial_corr_pear_pval"])
    res_df["partial_corr_spear_pval"] = _safe_fdr_bh(res_df["partial_corr_spear_pval"])

    return res_df


def process_all_ligands_corr(ligand_dict, ligand_pc1_dict, category_func=get_ligand_category_and_concentration):
    out = {}
    for ligand, norm_df_lig in ligand_dict.items():
        print(f"Processing partial correlations for {ligand}...")
        lig_pc1 = ligand_pc1_dict[ligand]
        df_pc1_genes, gene_cols, covar_cols = prepare_pc1_gene_df(
            norm_df_lig, lig_pc1, category_func=category_func
        )
        out[ligand] = compute_partial_corrs(
            df_pc1_genes, gene_cols, covar_cols, show_progress=True
        )
    return out


##############################################################################
# KS vs correlation plot data
##############################################################################

def generate_ks_corr_df(partial_corr_dict, ks_results_dict, ligand):
    partial_corr_df_lig = partial_corr_dict[ligand]
    ks_results_df_lig = ks_results_dict[ligand]

    full_conc_pval_df = pd.DataFrame(
        index=partial_corr_df_lig.index,
        columns=list(ks_results_df_lig.keys()),
        dtype=float,
    )

    for conc, df in ks_results_df_lig.items():
        full_conc_pval_df[conc] = df["Adj_P_Value"].reindex(partial_corr_df_lig.index)

    joint_p_vals = full_conc_pval_df.min(axis=1, skipna=True)

    data = pd.DataFrame(
        {
            "-log10(ks_pval)": (-np.log10(joint_p_vals.where(joint_p_vals > 0))),
            "partial_corr": partial_corr_df_lig["partial_corr_pear"],
        }
    )

    return data


def mask_and_filter(data, threshold=0.05, top_n=100):
    p_val_sig_limit = -np.log10(threshold)

    nonsig = data["-log10(ks_pval)"] < p_val_sig_limit
    corr_candidates = data.loc[nonsig, "partial_corr"].abs().dropna()

    corr_lim = corr_candidates.sort_values(ascending=False).head(top_n).min()

    mask = nonsig & (data["partial_corr"].abs() >= corr_lim)

    data_red = data.loc[mask].copy()
    data_blue = data.loc[~mask].copy()

    return data_red, data_blue, p_val_sig_limit, corr_lim

##############################################################################
# functions to generate data for pcorr vs ks plots
##############################################################################

# Define colors:

COLORS = {
    "BMP4": "#fa953d",
    "BMP6": "#59da1e",
    "BMP9": "#ec4949",
    "BMP10": "#3f90db",
    "GDF5": "#9467bd",
    "TGFb1": "#8c564b",
}


def plot_ks_corr(data, ligand, gene_list=[], empty=False, save=False):
    data_red, data_blue, p_val_sig_limit, corr_lim = mask_and_filter(
        data, threshold=0.05
    )

    if empty:
        # plot empty figure
        plt.figure()
        plt.xlabel("Correlation with perception score", fontsize=14)
        plt.ylabel("Differntial expression\n-log10(P-value)", fontsize=14)

    else:
        sns.scatterplot(
            data=data_red,
            x="partial_corr",
            y="-log10(ks_pval)",
            alpha=0.3,
            color="black",
            s=75,
        )

        sns.scatterplot(
            data=data_blue,
            x="partial_corr",
            y="-log10(ks_pval)",
            alpha=0.1,
            color=COLORS[ligand],
            s=75,
        )

        # make the gene_list dots in black:
        sns.scatterplot(
            data=data.loc[[gene_ for gene_ in gene_list if gene_ in data.index]],
            x="partial_corr",
            y="-log10(ks_pval)",
            color="black",
            s=75,
            marker="o",
            label="Genes of Interest",
        )

        if gene_list:
            for gene in gene_list:
                if gene in data.index:
                    x = data.loc[gene, "partial_corr"]
                    y = data.loc[gene, "-log10(ks_pval)"]
                    if x < 0:
                        plt.text(
                            x - 0.2,
                            y,
                            gene,
                            fontsize=12,
                            fontweight="bold",
                            color="black",
                            ha="right",
                            va="center",
                        )
                    else:
                        plt.text(
                            x + 0.25,
                            y,
                            gene,
                            fontsize=12,
                            fontweight="bold",
                            color="black",
                            ha="center",
                            va="center",
                        )

    plt.xlabel("Correlation with perception score", fontsize=14)
    plt.ylabel("Differntial expression\n-log10(P-value)", fontsize=14)

    plt.axhline(y=p_val_sig_limit, color="black", linestyle="--", linewidth=1)
    plt.axvline(x=corr_lim, color="black", linestyle="--", linewidth=1)
    plt.axvline(x=corr_lim * (-1), color="black", linestyle="--", linewidth=1)

    plt.xticks(size=14)
    plt.yticks(size=14)

    # Force tick markers to appear on both X and Y axes
    plt.gca().tick_params(axis="both", which="both", bottom=True, left=True, direction="out")

    plt.ylim(-0.15, 5)
    # center the plot around 0
    x_lim = max(abs(data["partial_corr"].min()), abs(data["partial_corr"].max()))
    plt.xlim(-x_lim, x_lim)

    plt.tight_layout()

    if save:
        plt.savefig(
            SAVE_DIR / f"{ligand}_corr_ks_plot.pdf",
            format="pdf",
            bbox_inches="tight",
        )
    plt.show()


##############################################################################
# Plot (4): BMP4 Gene partial correlation with PC1 vs. KS test p-value
##############################################################################

def update_matrix_labels(simplified_go, non_red_dict):
    # Copy to prevent modifying the original object data
    mat = simplified_go.S_sorted.copy()
    
    # 1. Extract row_color (Values = Colors, Index = GO IDs)
    row_color = mat.pop("row_color")
    
    # 2. Map the matrix axes (GO ID -> New Name)
    mat.index = mat.index.map(non_red_dict)
    mat.columns = mat.columns.map(non_red_dict)
    
    # 3. Update the row_color index to match the new matrix index
    # We do NOT map the values of row_color, only its index
    row_color.index = mat.index
    
    # 4. Re-insert the colors
    mat["row_color"] = row_color
    
    return mat