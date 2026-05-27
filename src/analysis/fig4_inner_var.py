import os
import pandas as pd
import tqdm
import warnings
import numpy as np
import pingouin as pg
import matplotlib.pyplot as plt
import seaborn as sns

from statsmodels.stats.multitest import multipletests
from scipy.stats import ks_2samp

from pyscripts.fig4_functions import prepare_data_from_model
from pyscripts.utils import add_df_ligand_names_and_concentrations_columns
# from pyscripts.fig1_functions import COLORS


##############################################################################
# functions preparing data for KS test and partial correlations
##############################################################################


def filter_off_genes(norm_df_lig, lig_pc1):
    df_pc1_genes = norm_df_lig
    df_pc1_genes["PC1"] = lig_pc1
    df_pc1_genes["conc"] = df_pc1_genes.index.values
    gene_cols = df_pc1_genes.columns[:-3]
    keep_genes = gene_cols[df_pc1_genes[gene_cols].sum(axis=0) > 0].tolist()
    df_pc1_genes = pd.concat(
        [df_pc1_genes.loc[:, keep_genes], df_pc1_genes.iloc[:, -3:]], axis=1
    )

    return df_pc1_genes, lig_pc1, keep_genes


def perpare_ligand_data_from_model(lig_model, filter=True):
    lig_df_genes, lig_pc1, lig_genes = prepare_data_from_model(lig_model)
    all_genes = lig_model.df.columns.to_list()
    all_genes = [g for g in all_genes if (g not in lig_genes) & ("MT-" not in g)]

    norm_df_lig = lig_model.df.loc[
        lig_model.df.index.str.contains(f"{lig_model.ligand}|CTRL_1"), all_genes
    ].copy()

    if filter:
        df_pc1_lig, lig_pc1, keep_genes = filter_off_genes(norm_df_lig, lig_pc1)
        all_genes = keep_genes

        return df_pc1_lig, lig_pc1, all_genes

    return norm_df_lig


##############################################################################
# functions to compute KS test results
##############################################################################


def run_ks_test(zero_df, mid_df, gene):
    data_mid = mid_df.loc[:, gene].to_numpy().flatten()
    data_zero = zero_df.loc[:, gene].to_numpy().flatten()
    ks_stat, p_value = ks_2samp(data_mid, data_zero)
    return ks_stat, p_value


def compute_ks_for_ligand(
    norm_df_lig, ctrl_conc="CTRL_1", genes=None, show_progress=True
):
    if genes is None:
        genes = norm_df_lig.columns.tolist()

    lig_concs = norm_df_lig.index.unique().to_list()
    try:
        lig_concs.remove(ctrl_conc)
    except ValueError:
        pass

    lig_zero = norm_df_lig[norm_df_lig.index.str.contains(ctrl_conc)].copy()
    out = {}

    for conc in lig_concs:
        lig_conc = norm_df_lig[norm_df_lig.index.str.contains(conc)].copy()

        if lig_zero.shape[0] == 0 or lig_conc.shape[0] == 0:
            warnings.warn(f"Empty group for conc {conc}, skipping KS tests.")
            continue

        rows = []
        iterator = genes
        if show_progress:
            iterator = tqdm.tqdm(genes)

        for gene in iterator:
            ks_stat, p_value = run_ks_test(lig_zero, lig_conc, gene)
            rows.append((gene, ks_stat, p_value))

        ks_test_results_df = pd.DataFrame(rows, columns=["Gene", "KS_Stat", "P_Value"])
        ks_test_results_df["Adj_P_Value"] = multipletests(
            ks_test_results_df["P_Value"], method="fdr_bh"
        )[1]
        ks_test_results_df.set_index("Gene", inplace=True)
        ks_test_results_df.index.name = None

        out[conc] = ks_test_results_df

    return out


def process_all_ligands_ks(
    ligand_dict, ctrl_conc="CTRL_1", genes=None, show_progress=True
):
    ks_results_dict_genes = {}
    for ligand, norm_df_lig in ligand_dict.items():
        if show_progress:
            print(f"Computing KS tests for {ligand}...")
        ks_results_dict_genes[ligand] = compute_ks_for_ligand(
            norm_df_lig, ctrl_conc=ctrl_conc, genes=genes, show_progress=show_progress
        )
    return ks_results_dict_genes


##############################################################################
# functions to compute partial correlations for all ligands
##############################################################################


def get_partial_corr(data, col_B, method, covar=[], col_A="PC1"):
    global_partial_corr = pg.partial_corr(
        data=data,
        x=col_A,
        y=col_B,
        covar=covar,
        method=method,
    )

    return global_partial_corr


def get_ligand_category_and_concentration(norm_df):
    id_conc_df = add_df_ligand_names_and_concentrations_columns(norm_df)
    id_conc_df_mid = pd.DataFrame(
        id_conc_df["concentration"].unique(), columns=["concentration"]
    )
    id_conc_df_mid["idx"] = id_conc_df.index.unique()
    id_conc_df_mid.sort_values(by="concentration", inplace=True)
    cat_list = id_conc_df_mid["idx"].to_list()
    return cat_list


def mask_and_filter(data, threshold=0.05):
    p_val_sig_limit = -np.log10(0.05)
    corr_lim = (
        data[data["-log10(ks_pval)"] < p_val_sig_limit]["partial_corr"]
        .abs()
        .sort_values(ascending=False)
        .head(100)
        .min()
    )

    mask = (data["-log10(ks_pval)"] < p_val_sig_limit) & (
        data["partial_corr"].abs() >= corr_lim
    )

    data_red = data[mask]
    data_blue = data[~mask]

    return data_red, data_blue, p_val_sig_limit, corr_lim


def prepare_pc1_gene_df(
    norm_df_lig, lig_pc1, category_func=get_ligand_category_and_concentration
):
    df = norm_df_lig.copy()
    df["PC1"] = lig_pc1
    df["conc"] = df.index.values

    # build categorical and one-hot (drop first category)
    lig_conc_list = category_func(norm_df_lig)
    df["cat"] = pd.Categorical(df["conc"], categories=lig_conc_list, ordered=False)
    dummies = pd.get_dummies(df["cat"])
    if df["cat"].cat.categories.size > 0:
        dummies = dummies[df["cat"].cat.categories[1:]]  # keep all except first (CTRL)
    df = pd.concat([df, dummies], axis=1)
    df.drop(columns=["conc", "cat"], inplace=True)

    # derive gene columns (everything except PC1 and the dummy covariates)
    covar_cols = dummies.columns.tolist()
    non_gene_cols = set(["PC1"] + covar_cols)
    gene_cols = [c for c in df.columns if c not in non_gene_cols]

    return df, gene_cols, covar_cols


def compute_partial_corrs(df_pc1_genes, gene_cols, covar_cols, show_progress=True):
    cols = [
        "partial_corr_spear",
        "partial_corr_spear_pval",
        "partial_corr_pear",
        "partial_corr_pear_pval",
    ]
    res_df = pd.DataFrame(index=gene_cols, columns=cols, dtype=float)

    iterator = gene_cols
    if show_progress:
        iterator = tqdm.tqdm(gene_cols)

    for gene in iterator:
        if gene == "PC1":
            continue
        # spearman
        r_s, p_s = (np.nan, np.nan)
        try:
            out_s = get_partial_corr(
                data=df_pc1_genes, col_B=gene, method="spearman", covar=covar_cols
            ).iloc[0]
            r_s, p_s = out_s["r"], out_s["p-val"]
        except Exception:
            pass

        # pearson
        r_p, p_p = (np.nan, np.nan)
        try:
            out_p = get_partial_corr(
                data=df_pc1_genes, col_B=gene, method="pearson", covar=covar_cols
            ).iloc[0]
            r_p, p_p = out_p["r"], out_p["p-val"]
        except Exception:
            pass

        res_df.loc[gene, "partial_corr_spear"] = r_s
        res_df.loc[gene, "partial_corr_spear_pval"] = p_s
        res_df.loc[gene, "partial_corr_pear"] = r_p
        res_df.loc[gene, "partial_corr_pear_pval"] = p_p

    # multiple testing correction (skip NaNs)
    res_df["partial_corr_pear_pval"] = multipletests(
        res_df["partial_corr_pear_pval"].astype(float), method="fdr_bh"
    )[1]
    res_df["partial_corr_spear_pval"] = multipletests(
        res_df["partial_corr_spear_pval"].astype(float), method="fdr_bh"
    )[1]

    return res_df


def process_all_ligands_corr(
    ligand_dict, ligand_pc1_dict, category_func=get_ligand_category_and_concentration
):
    out = {}
    for ligand, norm_df_lig in ligand_dict.items():
        print(f"Processing partial correlations for {ligand}...")
        lig_pc1 = ligand_pc1_dict[ligand]
        df_pc1_genes, gene_cols, covar_cols = prepare_pc1_gene_df(
            norm_df_lig, lig_pc1, category_func=category_func
        )
        pcorr_df = compute_partial_corrs(
            df_pc1_genes, gene_cols, covar_cols, show_progress=True
        )
        out[ligand] = pcorr_df
    return out


##############################################################################
# functions to generate data for pcorr vs ks plots
##############################################################################


def generate_ks_corr_df(partial_corr_dict, ks_results_dict, ligand):
    partial_corr_df_lig = partial_corr_dict[ligand]
    ks_results_df_lig = ks_results_dict[ligand]

    # create DataFrame indexed by genes present in partial_corr_df_lig
    full_conc_pval_df = pd.DataFrame(
        index=partial_corr_df_lig.index, columns=list(ks_results_df_lig.keys())
    )

    for conc in ks_results_df_lig.keys():
        # extract adj p-values and align to partial_corr genes (missing -> NaN)
        adj = ks_results_df_lig[conc]["Adj_P_Value"].reindex(partial_corr_df_lig.index)
        full_conc_pval_df[conc] = adj

    # joint p-value per gene (min across concentrations), skip NaNs
    joint_p_vals = full_conc_pval_df.min(axis=1, skipna=True)

    data = pd.DataFrame(
        {
            "-log10(ks_pval)": joint_p_vals.apply(
                lambda x: -np.log10(x) if pd.notnull(x) and x > 0 else np.nan
            ),
            "partial_corr": partial_corr_df_lig.loc[
                joint_p_vals.index, "partial_corr_pear"
            ],
        }
    )

    return data


def mask_and_filter(data, threshold=0.05):
    p_val_sig_limit = -np.log10(0.05)
    corr_lim = (
        data[data["-log10(ks_pval)"] < p_val_sig_limit]["partial_corr"]
        .abs()
        .sort_values(ascending=False)
        .head(100)
        .min()
    )

    mask = (data["-log10(ks_pval)"] < p_val_sig_limit) & (
        data["partial_corr"].abs() >= corr_lim
    )

    data_red = data[mask]
    data_blue = data[~mask]

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

    plt.ylim(-0.15, 5)
    # center the plot around 0
    x_lim = max(abs(data["partial_corr"].min()), abs(data["partial_corr"].max()))
    plt.xlim(-x_lim, x_lim)

    plt.tight_layout()

    if save:
        os.makedirs(f"images/figure_4/", exist_ok=True)
        plt.savefig(
            f"images/figure_4/{ligand}_corr_ks_plot.pdf",
            format="pdf",
            bbox_inches="tight",
        )
    plt.show()
