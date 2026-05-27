from pathlib import Path

import numpy as np
import pandas as pd
import scanpy as sc
import scipy.stats as stats
from sklearn.decomposition import PCA

from figures.fig_scripts.fig5_plotting import plot_variances
from src.utils.utils import add_df_ligand_names_and_concentrations_columns, load_ligand_models, set_style


set_style()


def get_ctrl_for_ligand(ligand):
    ctrl_lig_dict = {
        "GDF5": "CTRL_1",
        "BMP10": "CTRL_2",
        "TGFb1": "CTRL_3",
        "BMP6": "CTRL_4",
        "BMP9": "CTRL_5",
        "BMP4": "CTRL_6",
    }

    return ctrl_lig_dict[ligand]


def get_ligand_train_test_df(adata, train_barcodes, test_barcodes, ligand_sig_genes, ligand):
    lig_ctrl = get_ctrl_for_ligand(ligand)

    train_mask = adata.obs["cell_barcode"].isin(train_barcodes) & adata.obs["sample_id"].str.contains(
        ligand + "|" + lig_ctrl
    )
    adata_ligand = adata[train_mask, ligand_sig_genes]
    ligand_train_df = pd.DataFrame(
        adata_ligand.X.toarray(),
        columns=adata_ligand.var_names,
        index=adata_ligand.obs["sample_id"],
    )

    test_mask = adata.obs["cell_barcode"].isin(test_barcodes) & adata.obs["sample_id"].str.contains(
        ligand + "|" + lig_ctrl
    )
    adata_ligand_test = adata[test_mask, ligand_sig_genes]
    ligand_test_df = pd.DataFrame(
        adata_ligand_test.X.toarray(),
        columns=adata_ligand_test.var_names,
        index=adata_ligand_test.obs["sample_id"],
    )

    ligand_train_df.index = ligand_train_df.index.astype(str)
    ligand_test_df.index = ligand_test_df.index.astype(str)

    ligand_train_df["barcode"] = adata_ligand.obs["cell_barcode"].values
    ligand_test_df["barcode"] = adata_ligand_test.obs["cell_barcode"].values

    return ligand_train_df, ligand_test_df


def run_fitted_pca(df, n_components=8):
    """
    Fit PCA on the aggregated train dataframe.
    The dataframe is expected to contain a barcode column that will be dropped.
    """
    if "barcode" in df.columns:
        df = df.drop(columns=["barcode"])

    log_df_mean = df.groupby(df.index).mean()
    ctrl_sample = log_df_mean.index[log_df_mean.index.str.contains("CTRL")][0]
    log_df_mean = log_df_mean.loc[[ctrl_sample] + [idx for idx in log_df_mean.index if idx != ctrl_sample], :]

    pca_model_mean = PCA(n_components=n_components)
    pca_model_mean = pca_model_mean.fit(log_df_mean)

    return pca_model_mean, log_df_mean


def run_fitted_sc_pca(df, pca_model_mean, n_components=8):
    barcodes = None
    if "barcode" in df.columns:
        barcodes = df["barcode"].copy()
        df = df.drop(columns=["barcode"])

    pca_sc_fitted = pca_model_mean.transform(df)
    pca_sc_fitted_df = pd.DataFrame(
        pca_sc_fitted,
        index=df.index,
        columns=[f"PC{i + 1}" for i in range(n_components)],
    )
    pca_sc_fitted_df = add_df_ligand_names_and_concentrations_columns(pca_sc_fitted_df)
    if barcodes is not None:
        pca_sc_fitted_df["barcode"] = barcodes.values

    return pca_sc_fitted_df


def generate_ps_pred_var(pca_sc_df, xgb_pred_df):
    comp_df = pd.DataFrame(
        {
            "PC1": pca_sc_df["PC1"].values,
            "exp_conc": xgb_pred_df["exp_conc"],
            "ordinal_conc": xgb_pred_df["conc"],
        }
    )

    zcore_df = pd.DataFrame(
        index=comp_df.index,
        columns=[col + "_score_z" for col in comp_df.columns if col != "ordinal_conc"],
    )
    zcore_df[
        [col + "_score_z" for col in comp_df.columns if col != "ordinal_conc"]
    ] = stats.zscore(comp_df[[col for col in comp_df.columns if col != "ordinal_conc"]])
    zcore_df["ordinal_conc"] = comp_df["ordinal_conc"]

    variances = (
        zcore_df.groupby(zcore_df["ordinal_conc"])
        .agg({col + "_score_z": "var" for col in comp_df.columns if col != "ordinal_conc"})
        .round(10)
    )

    return zcore_df, variances


def build_panel_d_variance_df(data_dir: Path, ligand="BMP4", n_components=8):
    adata = sc.read_h5ad(data_dir / "adata_before_hvg.h5ad")
    train_barcodes = np.load(data_dir / "barcode_train_2250.npy", allow_pickle=True)
    test_barcodes = np.load(data_dir / "barcode_test_2250.npy", allow_pickle=True)

    models = load_ligand_models(subset=[ligand])
    ligand_model = models[ligand]
    ligand_sig_genes = (
        ligand_model.sig_genes_dict[ligand]["up"] + ligand_model.sig_genes_dict[ligand]["down"]
    )

    ligand_train_df, ligand_test_df = get_ligand_train_test_df(
        adata,
        train_barcodes,
        test_barcodes,
        ligand_sig_genes,
        ligand,
    )
    pca_model_mean, _ = run_fitted_pca(ligand_train_df, n_components=n_components)
    pca_sc = run_fitted_sc_pca(ligand_test_df, pca_model_mean, n_components=n_components)

    pred_df = pd.read_pickle(data_dir / "xgb_pred_df_2250.pkl")
    pred_df["barcode"] = test_barcodes
    pred_df = pred_df[pred_df["barcode"].isin(pca_sc["barcode"])].copy()

    pca_sc = pca_sc[pca_sc["barcode"].isin(pred_df["barcode"])].copy()
    pca_sc = pca_sc.set_index("barcode").loc[pred_df["barcode"]].reset_index()
    pred_df = pred_df.set_index("barcode").loc[pca_sc["barcode"]].reset_index()

    zscore_df, variances = generate_ps_pred_var(pca_sc, pred_df)
    vars_for_plot = variances.add_suffix(f"_{ligand.lower()}")

    return {
        "adata": adata,
        "train_barcodes": train_barcodes,
        "test_barcodes": test_barcodes,
        "ligand_train_df": ligand_train_df,
        "ligand_test_df": ligand_test_df,
        "pca_sc": pca_sc,
        "pred_df": pred_df,
        "zscore_df": zscore_df,
        "variances": variances,
        "vars_for_plot": vars_for_plot,
    }


def run_panel_d_variance_analysis(data_dir: Path, ligand="BMP4", n_components=8, save=False):
    results = build_panel_d_variance_df(data_dir=data_dir, ligand=ligand, n_components=n_components)
    plot_variances(
        results["vars_for_plot"],
        models=results["vars_for_plot"].columns[:],
        colors=["tab:orange", (0.60407977, 0.21017746, 0.43913439, 1.0), "lightgreen"],
        size=(9, 3),
        box=(1.7, 1),
        log=False,
        save=save,
    )
    return results
