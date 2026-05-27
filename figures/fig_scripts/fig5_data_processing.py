import os
from pathlib import Path
import pandas as pd
import numpy as np
import scipy
import scanpy as sc
import anndata as ad
import xgboost as xgb
import copy
import scipy.stats as stats

from dataclasses import dataclass, fields, replace
from typing import Optional

from scipy.interpolate import interp1d
from collections import Counter
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

from src.utils.utils import add_df_ligand_names_and_concentrations_columns, set_style
from src.analysis.reconstruct_conc import reconstruct_conc

BASE_DIR = Path(__file__).resolve().parent.parent.parent
RAW_DATA_FOLDER = BASE_DIR / "data" / "raw" 
DATA_SAVE_DIR = BASE_DIR / "data" / "processed" / "fig5_data"

os.makedirs(DATA_SAVE_DIR, exist_ok=True)

set_style()

##############################################################################
# Global Variables
##############################################################################

CONCENTRATIONS = np.array([0.0064, 0.032, 0.16, 0.8, 4.0, 20.0, 100.0, 500.0])


def _print_filter_summary(label, removed, retained, result_shape, noun):
    print(label)
    print(f"  Removed : {removed} {noun}")
    print(f"  Retained: {retained} {noun}")
    print(f"  Matrix after {noun} filtering: {result_shape}")


def _print_shape_change(label, before_shape, after_shape):
    print(label)
    print(f"  Shape before: {before_shape}")
    print(f"  Shape after : {after_shape}")

# ---------------------------------------------------------------------------
# Data container
# ---------------------------------------------------------------------------


@dataclass
class PipelineData:
    # name: str
    # version: int

    df_final: Optional[pd.DataFrame] = None
    adata_final: Optional[sc.AnnData] = None
    log2: bool = False
    sig_genes: Optional[list] = None
    min_cells: Optional[int] = None

    X_train: Optional[pd.DataFrame] = None
    X_test: Optional[pd.DataFrame] = None
    y_train: Optional[pd.Series] = None
    y_test: Optional[pd.Series] = None
    scale: bool = False
    scaler: Optional[object] = None
    dtrain: Optional[xgb.DMatrix] = None
    dtest: Optional[xgb.DMatrix] = None
    barcode_train: Optional[pd.Series] = None   
    barcode_test: Optional[pd.Series] = None

    dspatial: Optional[xgb.DMatrix] = None
    adata_spatial: Optional[sc.AnnData] = None
    df_spatial: Optional[pd.DataFrame] = None
    cutoffs_df: Optional[pd.DataFrame] = None

    # model
    model: Optional[xgb.Booster] = None
    evals_result: Optional[dict] = None

    # predicted concentrations
    conc_cols: Optional[list] = None
    log_concentrations: Optional[np.ndarray] = None
    probabilities: Optional[np.ndarray] = None
    hard_pred: Optional[np.ndarray] = None
    xgb_pred_df: Optional[pd.DataFrame] = None
    pred_conc_medians: Optional[pd.DataFrame] = None

    # spatial predictions
    y_pred_spatial: Optional[pd.DataFrame] = None
    y_pred_spatial_crypt: Optional[pd.DataFrame] = None
    y_pred_spatial_no_crypt: Optional[pd.DataFrame] = None
    y_pred_spatial_no_crypt_binned: Optional[pd.DataFrame] = None
    grouped_bins: Optional[pd.DataFrame] = None
    grouped_crypt: Optional[pd.DataFrame] = None

    # reconstructed concentrations
    rc_pred_conc_medians: Optional[pd.DataFrame] = None

    # exponential decay fit parameters:
    decay_results: Optional[dict] = None

    def copy(self):
        # Start with a shallow copy to bypass pickling errors (handles XGBoost objects safely)
        new_obj = replace(self)

        # Iterate over fields to deep copy only DataFrames, AnnData, and Lists
        for f in fields(self):
            val = getattr(self, f.name)
            if val is None:
                continue

            # Copy Pandas/AnnData/Numpy objects (they all have a .copy() method)
            if hasattr(val, "copy") and not isinstance(
                val, (xgb.DMatrix, xgb.Booster, StandardScaler, MinMaxScaler)
            ):
                setattr(new_obj, f.name, val.copy())

            # Deep copy lists/dicts
            elif isinstance(val, (list, dict)):
                setattr(new_obj, f.name, copy.deepcopy(val))

        return new_obj


##############################################################################
# Functions to process raw data files and prepare data for modeling
##############################################################################


def load_raw_data_files(RAW_DATA_FOLDER=RAW_DATA_FOLDER):
    path = RAW_DATA_FOLDER / "Exp3_2021_Feb/"
    raw_data = pd.read_pickle(path / "data_file.p")
    features = pd.read_pickle(path / "feat_file.p")
    meta_data = pd.read_pickle(path / "meta_file.p")
    barcodes = pd.read_pickle(path / "barcodes.p")
    # Change genes to be the columns and and cells to be the rows:
    raw_data = raw_data.T.tocsc()

    # fix gene names
    features[1] = features[1].map(lambda x: x.removeprefix("MM10_"))

    return raw_data, meta_data, features, barcodes


def generate_raw_adata(raw_data, meta_data, features, barcodes):
    meta_data.set_index("cell_barcode", inplace=True, drop=False)
    features.columns = ["gene_id", "gene_name", "type"]

    adata = ad.AnnData(X=raw_data, obs=meta_data, var=features["gene_name"].values)
    adata.var = adata.var.rename(columns={0: "gene_name"})
    adata.var.set_index("gene_name", drop=False, inplace=True)
    adata.var.index.name = None
    # adata.var_names_make_unique()
    adata.obs["ligand"] = adata.obs["sample_id"].apply(
        lambda x: x if "CTRL" in x else x.split("_")[0]
    )

    return adata


def prepare_cell_filtering(adata, threshold=4000):
    before_shape = adata.shape
    X = adata.X

    # Count expressed genes per cell
    count_genes_that_are_not_zero = (X > 0).sum(axis=1)

    # Calculate mitochondrial read percentage per cell
    mito_genes_mask = adata.var["gene_name"].str.upper().str.startswith("MT-")
    mito_gene_indices = np.where(mito_genes_mask)[0]
    mito_counts = X[:, mito_gene_indices].sum(axis=1)
    total_counts = X.sum(axis=1)
    mitochondrial_read_percent_per_cell = mito_counts / total_counts

    # Filter cells based on mitochondrial read percentage and gene count
    cells_after_filtering = (mitochondrial_read_percent_per_cell < 0.1) & (
        count_genes_that_are_not_zero > threshold
    )

    cells_after_filtering = np.array(cells_after_filtering).flatten()
    filtered_adata = adata[cells_after_filtering].copy()
    _print_filter_summary(
        "Cell filtering summary:",
        removed=int((~cells_after_filtering).sum()),
        retained=int(cells_after_filtering.sum()),
        result_shape=filtered_adata.shape,
        noun="cells",
    )
    _print_shape_change("  Cell matrix shape change:", before_shape, filtered_adata.shape)

    return filtered_adata


def remove_unknown_samples(adata):
    # Remove samples with unknown labels
    before_shape = adata.shape
    keep_mask = ~adata.obs["sample_id"].isin([
        "unknown",
    ])
    filtered_adata = adata[keep_mask].copy()
    _print_filter_summary(
        "Unknown sample filtering summary:",
        removed=int((~keep_mask).sum()),
        retained=int(keep_mask.sum()),
        result_shape=filtered_adata.shape,
        noun="cells",
    )
    _print_shape_change("  Sample matrix shape change:", before_shape, filtered_adata.shape)
    return filtered_adata


def remove_lower_duplicates(adata):
    before_shape = adata.shape
    X = adata.X.toarray() if scipy.sparse.issparse(adata.X) else adata.X
    df = pd.DataFrame(X, index=adata.obs_names, columns=adata.var["gene_name"])

    duplicate_gene_count = int(df.columns.duplicated().sum())
    if duplicate_gene_count == 0:
        _print_filter_summary(
            "Duplicate gene merging summary:",
            removed=0,
            retained=adata.n_vars,
            result_shape=adata.shape,
            noun="genes",
        )
        _print_shape_change("  Gene matrix shape change:", before_shape, adata.shape)
        return adata

    cols_to_drop = []
    col_counter = Counter(df.columns)
    for gene, count in col_counter.items():
        if count <= 1:
            continue
        col_positions = [i for i, c in enumerate(df.columns) if c == gene]
        subest_df = df.iloc[:, col_positions]
        subest_df_sum = subest_df.sum(axis=0)
        max_idx = int(np.argmax(subest_df_sum))
        for j, pos in enumerate(col_positions):
            if j != max_idx:
                cols_to_drop.append(pos)

    cols_to_keep_idx = [i for i in range(adata.n_vars) if i not in cols_to_drop]
    adata_merged = adata[:, cols_to_keep_idx].copy()

    _print_filter_summary(
        "Duplicate gene merging summary:",
        removed=len(cols_to_drop),
        retained=adata_merged.n_vars,
        result_shape=adata_merged.shape,
        noun="genes",
    )
    _print_shape_change("  Gene matrix shape change:", before_shape, adata_merged.shape)

    return adata_merged


# def remove_lower_duplicates(adata):
#     before_shape = adata.shape
#     df = pd.DataFrame(
#         adata.X.toarray(), index=adata.obs_names, columns=adata.var["gene_name"]
#     )

#     duplicate_gene_count = int(df.columns.duplicated().sum())

#     # Merge duplicated gene columns by summing them
#     merged_df = df.T.groupby(level=0, sort=False).sum().T

#     # Rebuild AnnData with unique gene names
#     new_var = pd.DataFrame(index=merged_df.columns)
#     new_var["gene_name"] = merged_df.columns

#     _print_filter_summary(
#         "Duplicate gene merging summary:",
#         removed=duplicate_gene_count,
#         retained=merged_df.shape[1],
#         result_shape=merged_df.shape,
#         noun="genes",
#     )
#     _print_shape_change("  Gene matrix shape change:", before_shape, merged_df.shape)

#     return ad.AnnData(X=merged_df.values, obs=adata.obs.copy(), var=new_var)

def seperate_h2b(adata, DATA_FOLDER=DATA_SAVE_DIR, save=False):
    before_shape = adata.shape
    h2b_ind = np.where(adata.var_names == "H2BCITRINE")[0]
    if isinstance(adata.X, scipy.sparse.csc_matrix):
        h2bcit_irne_expression = adata.X[:, h2b_ind].toarray().flatten()
    elif isinstance(adata.X, np.ndarray):
        h2bcit_irne_expression = adata.X[:, h2b_ind].flatten()
    keep_mask = adata.var_names != "H2BCITRINE"
    adata = adata[:, keep_mask].copy()
    _print_filter_summary(
        "H2BCITRINE separation summary:",
        removed=int((~keep_mask).sum()),
        retained=int(keep_mask.sum()),
        result_shape=adata.shape,
        noun="genes",
    )
    _print_shape_change("  Gene matrix shape change:", before_shape, adata.shape)
    if save:
        adata.write_h5ad(DATA_FOLDER / "adata_cell_dup_filtered.h5ad")
        print("adata without H2BCITRINE saved.")
        np.save(DATA_FOLDER / "h2bcitirne_expression.npy", h2bcit_irne_expression)
        print("H2BCITRINE expression saved.")
    return adata, h2bcit_irne_expression


def _rename_ctrl(data_obj):
    ctrl_lig_dict = {
        "CTRL_1": "CTRL_GDF5",
        "CTRL_2": "CTRL_BMP10",
        "CTRL_3": "CTRL_TGFb1",
        "CTRL_4": "CTRL_BMP6",
        "CTRL_5": "CTRL_BMP9",
        "CTRL_6": "CTRL_BMP4",
    }

    if isinstance(data_obj, pd.DataFrame):
        df = data_obj.copy()
        df = df.rename(index=ctrl_lig_dict)
        return df

    if isinstance(data_obj, ad.AnnData):
        data_obj.obs["ligand"] = (
            data_obj.obs["ligand"].astype(str).replace(ctrl_lig_dict)
        )
        data_obj.obs["ligand"] = data_obj.obs["ligand"].astype("category")
        return data_obj


def _filter_lig_df(adata, ligand):
    return adata[adata.obs["ligand"].str.contains(ligand)]


def filter_data(adata, ligands="BMP4|BMP6|BMP9"):
    adata = _rename_ctrl(adata)
    print(f"Filter to ligands: {ligands}")
    before_shape = adata.shape
    adata_bmp = _filter_lig_df(adata, ligands)
    _print_filter_summary(
        "Ligand filtering summary:",
        removed=int(before_shape[0] - adata_bmp.shape[0]),
        retained=int(adata_bmp.shape[0]),
        result_shape=adata_bmp.shape,
        noun="cells",
    )
    _print_shape_change("  Ligand subset shape change:", before_shape, adata_bmp.shape)
    return adata_bmp


def rank_ligand_concentrations(df, ligands):
    """
    Rank ligand concentrations and assign integer ranks.
    """
    for lig in ligands:
        ligand_mask = df["ligand"] == lig
        if lig.startswith("CTRL"):
            ligand_mask = df["ligand"].astype(str).str.startswith("CTRL")
            df.loc[ligand_mask, "ranked_conc"] = 0
        elif lig in ["6HR-BMP4", "6HR-BMP9"]:
            df.loc[ligand_mask, "ranked_conc"] = 8
        else:
            df.loc[ligand_mask, "ranked_conc"] = df.loc[
                ligand_mask, "concentration"
            ].rank(method="dense", ascending=True)
    
    df["ranked_conc"] = df["ranked_conc"].astype(int)
    df.drop(columns=["concentration"], inplace=True)

    return df


def process_df_for_final(
    adata,
    ligands,
    sig_genes=[],
    min_cell=None,
    lm_genes=[],
    log2=False,
    highly_variable_genes=True,
    n_top=2000,
    post_norm_filt=[],
    rank_conc=True,
    save_before_hvg=False,
    attach_barcodes=False,
    barcode_obs_col="cell_barcode",
    barcode_output_col="barcode",
) -> PipelineData:
    before_shape = adata.shape
    if sig_genes and min_cell:
        raise ValueError(
            "Specify at most one of 'sig_genes' and 'min_cell'. To skip filtering, pass neither."
        )
    if sig_genes:
        adata = adata[:, sig_genes].copy()
        print(f"Filtered adata to {len(sig_genes)} significant genes only")
        _print_filter_summary(
            "Significant gene filtering summary:",
            removed=int(before_shape[1] - adata.shape[1]),
            retained=int(adata.shape[1]),
            result_shape=adata.shape,
            noun="genes",
        )
        _print_shape_change("  Adata shape change:", before_shape, adata.shape)
    elif min_cell:
        n_before = adata.n_vars
        sc.pp.filter_genes(adata, min_cells=min_cell)
        _print_filter_summary(
            "Minimum-cell gene filtering summary:",
            removed=int(n_before - adata.n_vars),
            retained=int(adata.n_vars),
            result_shape=adata.shape,
            noun="genes",
        )
        _print_shape_change("  Adata shape change:", before_shape, adata.shape)
        print(f"Filtered adata to genes with at least {min_cell} cells")
    else:
        print(
            "No gene filtering applied (neither 'sig_genes' nor 'min_cell' provided)."
        )
    if lm_genes:
        lm_genes = [gene for gene in lm_genes if gene in adata.var_names]
        before_lm_shape = adata.shape
        adata = adata[:, ~adata.var_names.isin(lm_genes)].copy()
        _print_filter_summary(
            "Ligand marker gene removal summary:",
            removed=len(lm_genes),
            retained=int(adata.shape[1]),
            result_shape=adata.shape,
            noun="genes",
        )
        _print_shape_change("  Adata shape change:", before_lm_shape, adata.shape)
        print(f"Removed {len(lm_genes)} ligand marker genes from adata")

    print(f"Processing adata with dimensions: {adata.shape}")
    adata.layers["counts"] = adata.X.copy()

    sc.pp.normalize_total(adata, target_sum=1e4)
    if log2:
        print("Applying log2 transformation")
        sc.pp.log1p(adata, base=2)
    else:
        print("Applying natural log transformation")
        sc.pp.log1p(adata)
    adata.raw = adata

    if save_before_hvg:
        adata.write_h5ad(DATA_SAVE_DIR / "adata_before_hvg.h5ad")
        print("Adata before HVG filtering saved.")

    if highly_variable_genes:
        before_hvg_shape = adata.shape
        sc.pp.highly_variable_genes(
            adata, n_top_genes=n_top, subset=True, layer="counts", flavor="seurat_v3"
        )
        _print_filter_summary(
            "Highly variable gene selection summary:",
            removed=int(before_hvg_shape[1] - adata.shape[1]),
            retained=int(adata.shape[1]),
            result_shape=adata.shape,
            noun="genes",
        )
        _print_shape_change("  Adata shape change:", before_hvg_shape, adata.shape)

    if post_norm_filt:
        before_post_norm_shape = adata.shape
        print("Filtering to specified gene list after normalization")
        adata = adata[:, post_norm_filt].copy()
        _print_filter_summary(
            "Post-normalization gene filtering summary:",
            removed=int(before_post_norm_shape[1] - adata.shape[1]),
            retained=int(adata.shape[1]),
            result_shape=adata.shape,
            noun="genes",
        )
        _print_shape_change("  Adata shape change:", before_post_norm_shape, adata.shape)

    df = pd.DataFrame(
        adata.X.toarray() if scipy.sparse.issparse(adata.X) else adata.X,
        columns=adata.var_names,
        index=adata.obs["sample_id"],
    )
    df.index.name = None

    df = add_df_ligand_names_and_concentrations_columns(df)
    if attach_barcodes:
        if barcode_obs_col not in adata.obs.columns:
            raise ValueError(
                f"{barcode_obs_col} not found in adata.obs. "
                "Set barcode_obs_col to an existing obs column."
            )
        # Positional assignment keeps alignment with current row order
        df[barcode_output_col] = adata.obs[barcode_obs_col].to_numpy()

        df["ligand"] = df["ligand"].apply(lambda x: "CTRL" if x.startswith("CTRL") else x)

    if rank_conc:
        df = rank_ligand_concentrations(df, ligands)

    return PipelineData(df_final=df, adata_final=adata, log2=log2, sig_genes=sig_genes)


def _encode_labels(labels, column):
    labels_flat = labels[column].values.flatten()
    le = LabelEncoder()
    encoded_labels = le.fit_transform(labels_flat)
    # Display encoded labels
    print("Encoded Labels:", encoded_labels)

    # Display mapping of original labels to numbers
    print("Label Mapping:")
    for i, label in enumerate(le.classes_):
        print(f"{label} -> {i}")

    return encoded_labels


def prepare_data_for_modeling(
    pipeline: PipelineData,
    normal_stratify=True,
    scale=False,
    scaler=None,
    lm_genes=[],
    barcode_col="barcode",
    keep_barcodes=True,
) -> PipelineData:
    df = pipeline.df_final.copy()
    before_shape = df.shape

    # Keep ligand labels for optional joint stratification
    ligands_vec = df.pop("ligand")

    # Keep barcodes as metadata (not model features) if present
    barcode_vec = None
    if keep_barcodes and barcode_col in df.columns:
        barcode_vec = df.pop(barcode_col)

    bmp_labels = _encode_labels(df, "ranked_conc")
    df_bmp = df.drop(columns=["ranked_conc"])

    removed_cols = 2 + (1 if barcode_vec is not None else 0)
    _print_filter_summary(
        "Model feature preparation summary:",
        removed=removed_cols,
        retained=int(df_bmp.shape[1]),
        result_shape=df_bmp.shape,
        noun="columns",
    )
    _print_shape_change("  Modeling dataframe shape change:", before_shape, df_bmp.shape)

    X = df_bmp
    y = bmp_labels

    if normal_stratify:
        stratify = y
    else:
        stratify = pd.DataFrame(columns=["ranked_conc", "ligand"])
        stratify["ranked_conc"] = y
        stratify["ligand"] = ligands_vec.values

    print("Splitting data into train and test sets")
    if barcode_vec is not None:
        X_train, X_test, y_train, y_test, barcode_train, barcode_test = train_test_split(
            X,
            y,
            barcode_vec,
            test_size=0.2,
            random_state=42,
            stratify=stratify,
        )
        print(f"  Barcode vectors: train={barcode_train.shape[0]}, test={barcode_test.shape[0]}")
    else:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=stratify
        )
        barcode_train, barcode_test = None, None

    print(f"  Train matrix shape: {X_train.shape}")
    print(f"  Test matrix shape : {X_test.shape}")

    if scale:
        if scaler is None:
            scaler = StandardScaler()
        before_scale_shape = X_train.shape
        X_train = pd.DataFrame(
            scaler.fit_transform(X_train), columns=X_train.columns, index=X_train.index
        )
        X_test = pd.DataFrame(
            scaler.transform(X_test), columns=X_test.columns, index=X_test.index
        )
        print(f"Scaling data with {scaler.__class__.__name__}")
        _print_shape_change("  Scaled train matrix shape change:", before_scale_shape, X_train.shape)

    if lm_genes:
        lm_genes = [gene for gene in lm_genes if gene in X_train.columns]
        before_lm_shape = X_train.shape
        X_train = X_train.drop(columns=lm_genes)
        X_test = X_test.drop(columns=lm_genes)
        _print_filter_summary(
            "Model ligand marker removal summary:",
            removed=len(lm_genes),
            retained=int(X_train.shape[1]),
            result_shape=X_train.shape,
            noun="columns",
        )
        _print_shape_change("  Train matrix shape change:", before_lm_shape, X_train.shape)
        print(f"Removed {len(lm_genes)} ligand marker genes from data")

    dtrain = xgb.DMatrix(X_train, label=y_train)
    dtest = xgb.DMatrix(X_test, label=y_test)

    pipeline.X_train = X_train
    pipeline.X_test = X_test
    pipeline.y_train = y_train
    pipeline.y_test = y_test
    pipeline.scale = scale
    pipeline.scaler = scaler
    pipeline.dtrain = dtrain
    pipeline.dtest = dtest

    # Metadata vectors aligned to split rows
    pipeline.barcode_train = barcode_train
    pipeline.barcode_test = barcode_test

    return pipeline


def generate_conc_predictions_df(pipeline, model_xgb, conc_list=CONCENTRATIONS, copy=False):
    if copy:
        pipeline = pipeline.copy()
    pipeline.conc_cols = ["Conc. {}".format(i) for i in range(8)]
    pipeline.log_concentrations = np.log(conc_list)

    probs_xgb = model_xgb.predict(pipeline.dtest)
    hard_pred = np.argmax(probs_xgb, axis=1)

    # Generate y_pred DataFrame:
    y_pred_xgb_df = pd.DataFrame(
        probs_xgb,
        columns=pipeline.conc_cols,
        index=pipeline.X_test.index,
    )
    y_pred_xgb_df["conc"] = pipeline.y_test
    y_pred_xgb_df["exp_conc"] = np.dot(probs_xgb, pipeline.log_concentrations)
    y_pred_xgb_df["pseudo_conc"] = np.exp(y_pred_xgb_df["exp_conc"])

    pipeline.probabilities = probs_xgb
    pipeline.hard_pred = hard_pred
    pipeline.xgb_pred_df = y_pred_xgb_df
    pipeline.pred_conc_medians = get_agg_se(pipeline.xgb_pred_df, column="conc", agg="median")

    return pipeline


##############################################################################
# Functions to process spatial single cell data
##############################################################################


def process_umi_data(umi_data):
    before_shape = umi_data.shape
    umi_data = umi_data.T
    umi_data.columns = umi_data.iloc[0]
    umi_data = umi_data.drop(umi_data.index[0])
    umi_data.index.name = None
    umi_data.columns.name = None

    _print_filter_summary(
        "UMI preprocessing summary:",
        removed=1,
        retained=int(umi_data.shape[0]),
        result_shape=umi_data.shape,
        noun="rows",
    )
    _print_shape_change("  UMI matrix shape change:", before_shape, umi_data.shape)

    return umi_data


def generate_spatial_adata(umi_data_filt, cell_zone_table):
    before_shape = umi_data_filt.shape
    # make obs dataframe with the index of the umi_data_filt as a column and the cell_zone_table:
    obs = pd.DataFrame(index=umi_data_filt.index)
    obs["barcode"] = umi_data_filt.index
    obs[["zone", "spatial_coordinate"]] = cell_zone_table.loc[
        umi_data_filt.index, ["zone", "spatial_coordinate"]
    ]

    # make var object with uppercase column names:
    var = pd.DataFrame(index=umi_data_filt.columns)
    var["gene"] = umi_data_filt.columns.str.upper()

    umi_mat = umi_data_filt.values

    # make adata object:
    adata = sc.AnnData(X=umi_mat, obs=obs, var=var)
    # adata.X = np.asarray(adata.X)
    adata.var.set_index("gene", inplace=True, drop=False)
    adata.var.index.name = None

    print("Spatial AnnData generation summary:")
    print(f"  Input matrix shape : {before_shape}")
    print(f"  Output AnnData shape: {adata.shape}")

    return adata


def normalize_spatial_adata(adata, log2=False):
    sc.pp.normalize_total(adata, target_sum=1e4)
    if log2:
        print("Using log2 transformation for spatial data")
        sc.pp.log1p(adata, base=2)
    else:
        print("Using natural log transformation for spatial data")
        sc.pp.log1p(adata)
    return adata


def generate_df(adata, df_cols):
    before_shape = adata.shape
    df = pd.DataFrame(adata.X, columns=adata.var["gene"], index=adata.obs["barcode"])
    df.index.name = None
    df = df[df_cols]
    after_column_select_shape = df.shape
    duplicate_cols = int(df.columns.duplicated().sum())
    df = df.loc[:, ~df.columns.duplicated()]
    _print_filter_summary(
        "Spatial feature alignment summary:",
        removed=int(before_shape[1] - after_column_select_shape[1] + duplicate_cols),
        retained=int(df.shape[1]),
        result_shape=df.shape,
        noun="genes",
    )
    _print_shape_change("  Spatial dataframe shape change:", before_shape, df.shape)
    return df


def generate_spatial_data(
    pipeline: PipelineData, umi_data, cell_zone_table, copy=False
) -> PipelineData:
    if copy:
        pipeline = pipeline.copy()
    before_shape = pipeline.X_train.shape
    cell_zone_table.set_index("barcode", inplace=True, drop=True)

    model_columns = pipeline.X_train.columns
    cell_zone_table["barcode"] = cell_zone_table.index
    sc_umi_data = process_umi_data(umi_data)
    adata_spatial = generate_spatial_adata(sc_umi_data, cell_zone_table)

    if pipeline.sig_genes:
        before_sig_shape = adata_spatial.shape
        adata_spatial.var_names_make_unique()
        adata_spatial = adata_spatial[:, pipeline.sig_genes].copy()
        print("Filtered spatial adata to significant genes only")
        _print_filter_summary(
            "Spatial significant gene filtering summary:",
            removed=int(before_sig_shape[1] - adata_spatial.shape[1]),
            retained=int(adata_spatial.shape[1]),
            result_shape=adata_spatial.shape,
            noun="genes",
        )
        _print_shape_change("  Spatial AnnData shape change:", before_sig_shape, adata_spatial.shape)
    adata_spatial = normalize_spatial_adata(adata_spatial, log2=pipeline.log2)

    df_spatial = generate_df(adata_spatial, model_columns)
    if pipeline.scale:
        before_scale_shape = df_spatial.shape
        df_spatial = pd.DataFrame(
            pipeline.scaler.transform(df_spatial),
            columns=df_spatial.columns,
            index=df_spatial.index,
        )
        print(f"Scaling spatial data with {pipeline.scaler.__class__.__name__}")
        _print_shape_change("  Spatial dataframe shape change:", before_scale_shape, df_spatial.shape)

    pipeline.dspatial = xgb.DMatrix(df_spatial[pipeline.X_train.columns])
    pipeline.adata_spatial = adata_spatial
    pipeline.df_spatial = df_spatial
    print(f"  Spatial modeling matrix shape: {df_spatial.shape}")
    

    return pipeline


def _interpolate_cont_zones(pipeline: PipelineData, normalized=bool) -> None:

    cutoff_values = np.append(
        pipeline.cutoffs_df["Lower_Bound"].values, 1
    )  # add upper bound for interpolation
    # zone indices
    zones = np.arange(len(cutoff_values))

    # create interpolation function (cutoff → zone)
    f = interp1d(
        cutoff_values,
        zones,
        kind="linear",
        bounds_error=False,
        fill_value=(zones[0], zones[-1]),  # clamp outside range
    )

    # apply to your continuous scores
    pipeline.y_pred_spatial["continuous_zone"] = f(
        pipeline.y_pred_spatial["spatial_coordinate"]
    )

    if normalized:
        pipeline.y_pred_spatial["norm_continuous_zone"] = pipeline.y_pred_spatial["continuous_zone"] / zones[-1]  # normalize to [0,1]

def _fit_inter_data(pipeline, position=1.0):
    # 1. Prepare data
    x = pipeline.grouped_bins["spatial_median"].values
    y_log = np.log(pipeline.grouped_bins["pseudo_conc_median"].values)
    n = len(x) # Number of bins (e.g., 20)

    # 2. Linear regression on log-scale
    slope, intercept = np.polyfit(x, y_log, 1)
    
    # 3. Calculate Residuals and RSE (Residual Standard Error)
    y_log_pred = slope * x + intercept
    residuals = y_log - y_log_pred
    # Degrees of freedom: n - 2 (slope and intercept)
    rse = np.sqrt(np.sum(residuals**2) / (n - 2))

    # 4. Calculate Confidence Interval for the Source Concentration (at x=1.0)
    # We find the Standard Error (SE) for the prediction at x=1.0
    x_mean = np.mean(x)
    ssx = np.sum((x - x_mean)**2)
    
    # SE of the fit at a specific point x_val
    se_at_source = rse * np.sqrt(1/n + (position - x_mean)**2 / ssx)
    
    # 95% T-value for n-2 degrees of freedom
    t_val = stats.t.ppf(0.975, n - 2)
    ci_log_margin = t_val * se_at_source

    # 5. Extract Parameters
    r2 = np.corrcoef(x, y_log)[0, 1] ** 2
    k = np.abs(slope)
    
    # Source concentration and its Confidence Interval
    c_source = np.exp(slope * position + intercept)
    c_source_low = np.exp((slope * position + intercept) - ci_log_margin)
    c_source_high = np.exp((slope * position + intercept) + ci_log_margin)

    half_distance_norm = np.log(2) / k

    return {
        "params": (slope, intercept),
        "c_source": c_source,
        "c_source_ci": (c_source_low, c_source_high),
        "half_distance": half_distance_norm,
        "r2": r2,
        "rse_log": rse
    }


def get_exponential_fit_params(pipeline, position=1.0):
    fit_results = _fit_inter_data(pipeline, position=position)
    pipeline.decay_results = fit_results
    print(f"Fitted exponential decay constant k: {pipeline.decay_results['params'][0]:.4f}")
    print(f"Estimated concentration at source : {pipeline.decay_results['c_source']:.4f}")
    print(f"Estimated half-distance: {pipeline.decay_results['half_distance']:.4f}")
    print(f"R-squared of fit: {pipeline.decay_results['r2']:.4f}")
    # return pipeline


def generate_spatial_df(model, pipeline):
    if hasattr(model, "predict_proba"):
        pipeline.df_spatial = pipeline.df_spatial.copy()
        if hasattr(model, "decision_function"):
            y_pred_spatial = model.predict_proba(pipeline.dspatial)
        else:
            y_pred_spatial = model.predict_proba(pipeline.dspatial)
    else:
        y_pred_spatial = model.predict(pipeline.dspatial)
    y_pred_spatial_df = pd.DataFrame(
        y_pred_spatial,
        columns=pipeline.conc_cols,
        index=pipeline.adata_spatial.obs.index,
    )
    y_pred_spatial_df["exp_conc"] = np.dot(y_pred_spatial, pipeline.log_concentrations)
    y_pred_spatial_df["pseudo_conc"] = np.exp(y_pred_spatial_df["exp_conc"])
    y_pred_spatial_df["zone"] = pipeline.adata_spatial.obs["zone"]
    y_pred_spatial_df["spatial_coordinate"] = pipeline.adata_spatial.obs[
        "spatial_coordinate"
    ]

    return y_pred_spatial_df


def bin_spatial_coordinates(
    df, column="spatial_coordinate", bins=10, bin_labels=None, method="cut"
):
    df = df.copy()
    if method == "cut":
        if isinstance(bins, int):
            bin_labels = bin_labels if bin_labels else range(1, bins + 1)
            df.loc[:, "bins"] = pd.cut(df.loc[:, column], bins=bins, labels=bin_labels)
        else:
            bin_labels = bin_labels if bin_labels else range(1, len(bins))
            df.loc[:, "bins"] = pd.cut(
                df.loc[:, column], bins=bins, labels=bin_labels, include_lowest=True
            )

    elif method == "qcut":
        if isinstance(bins, int):
            bin_labels = bin_labels if bin_labels else range(1, bins + 1)
            df.loc[:, "bins"] = pd.qcut(df.loc[:, column], q=bins, labels=bin_labels)
        else:
            bin_labels = bin_labels if bin_labels else range(1, len(bins))
            df.loc[:, "bins"] = pd.qcut(
                df.loc[:, column], q=bins, labels=bin_labels, duplicates="drop"
            )

    return df


def process_spatial_data(
    pipeline: PipelineData, model, binned_col="spatial_coordinate", copy=False, normalized=True
) -> PipelineData:
    if copy:
        pipeline = pipeline.copy()

    pipeline.y_pred_spatial = y_pred_spatial = generate_spatial_df(model, pipeline)
    if pipeline.cutoffs_df is not None:
        _interpolate_cont_zones(pipeline, normalized=normalized)
    pipeline.y_pred_spatial_crypt = y_pred_spatial[y_pred_spatial["zone"] == 0]
    pipeline.y_pred_spatial_no_crypt = y_pred_spatial[y_pred_spatial["zone"] != 0]
    print("Spatial prediction split summary:")
    print(f"  Crypt cells   : {pipeline.y_pred_spatial_crypt.shape[0]}")
    print(f"  Non-crypt cells: {pipeline.y_pred_spatial_no_crypt.shape[0]}")
    print(f"  Matrix after zone split: {pipeline.y_pred_spatial.shape}")
    pipeline.y_pred_spatial_no_crypt_binned = bin_spatial_coordinates(
        pipeline.y_pred_spatial_no_crypt,
        column=binned_col,
        bins=20,
        method="qcut",
    )
    print(f"  Binned non-crypt matrix shape: {pipeline.y_pred_spatial_no_crypt_binned.shape}")
    pipeline.grouped_bins = get_agg_se(
        pipeline.y_pred_spatial_no_crypt_binned,
        column="bins",
        agg="median",
        binned_col=binned_col,
    )
    pipeline.grouped_crypt = get_agg_se(
        pipeline.y_pred_spatial_crypt,
        column="zone",
        agg="median",
        binned_col=binned_col,
    )

    return pipeline


def generate_conc_reconstruction_df(
    pipeline: PipelineData,
    binned_col="spatial_coordinate",
    source=None,
    reg_strength=0.05,
    copy=False,
) -> PipelineData:
    if copy:
        pipeline = pipeline.copy()
    if source is None:
        raise ValueError("source must be specified as 'test' or 'spatial'")

    if source == "test":
        before_shape = pipeline.xgb_pred_df.shape
        rc_df = reconstruct_conc(pipeline.xgb_pred_df, reg_strength=reg_strength)
        pipeline.xgb_pred_df = pd.concat([pipeline.xgb_pred_df, rc_df], axis=1)
        _print_shape_change("Reconstruction dataframe shape change (test):", before_shape, pipeline.xgb_pred_df.shape)
        pipeline.rc_pred_conc_medians = get_agg_se(pipeline.xgb_pred_df, column="conc", agg="median")
    elif source == "spatial":
        before_shape = pipeline.y_pred_spatial.shape
        rc_df = reconstruct_conc(pipeline.y_pred_spatial, reg_strength=reg_strength)
        pipeline.y_pred_spatial = pd.concat([pipeline.y_pred_spatial, rc_df], axis=1)
        _print_shape_change("Reconstruction dataframe shape change (spatial):", before_shape, pipeline.y_pred_spatial.shape)
        pipeline.y_pred_spatial_crypt = pipeline.y_pred_spatial[pipeline.y_pred_spatial["zone"] == 0]
        pipeline.y_pred_spatial_no_crypt = pipeline.y_pred_spatial[pipeline.y_pred_spatial["zone"] != 0]
        pipeline.y_pred_spatial_no_crypt_binned = bin_spatial_coordinates(
            pipeline.y_pred_spatial_no_crypt, column=binned_col, bins=20, method="qcut"
        )
        pipeline.grouped_bins = get_agg_se(
            pipeline.y_pred_spatial_no_crypt_binned, column="bins", agg="median", binned_col=binned_col,
        )
        pipeline.grouped_crypt = get_agg_se(pipeline.y_pred_spatial_crypt, column="zone", agg="median", binned_col=binned_col,)
    else:
        raise ValueError("source must be either 'test' or 'spatial'")

    return pipeline


##############################################################################
# Functions to aggregate predictions
##############################################################################


def get_agg_se(
    source, column="zone", agg="mean", binned_col="spatial_coordinate", n_boot=1000
):  
    if isinstance(source, pd.DataFrame):
        df = source.copy()
    else:
        raise ValueError("source must be  DataFrame")
    # Bootstrap SE for the median
    def _bootstrap_se_median(x, n_boot=n_boot):
        meds = [
            np.median(np.random.choice(x, size=len(x), replace=True))
            for _ in range(n_boot)
        ]
        return np.std(meds, ddof=1)

    def _se_mean(x):
        return np.std(x, ddof=1) / np.sqrt(len(x))

    # Validate aggregation choice
    if agg not in ["mean", "median"]:
        raise ValueError("agg must be either 'mean' or 'median'")

    # Aggregation and SE functions
    if agg == "mean":
        agg_func = np.mean
        se_func = _se_mean
    else:
        agg_func = np.median
        se_func = _bootstrap_se_median

    agg_col = "rc_log_conc" if "rc_log_conc" in df.columns else "exp_conc"

    # Main pseudo_conc outputs
    agg_dict = {
        f"pseudo_conc_{agg}": (
            agg_col,
            lambda x: np.exp(agg_func(x)),
        ),
        f"pseudo_conc_se_upper_{agg}": (
            agg_col,
            lambda x: np.exp(agg_func(x) + se_func(x)) - np.exp(agg_func(x)),
        ),
        f"pseudo_conc_se_lower_{agg}": (
            agg_col,
            lambda x: np.exp(agg_func(x)) - np.exp(agg_func(x) - se_func(x)),
        ),
    }

    # Spatial metrics if applicable
    if column in ["zone", "bins"]:
        # Use string aggregation names for pandas GroupBy to avoid
        # FutureWarning when passing numpy callables directly.
        if agg == "mean":
            spatial_agg = "mean"
            spatial_se = _se_mean
        else:
            spatial_agg = "median"
            spatial_se = _bootstrap_se_median

        agg_dict.update(
            {
                f"spatial_{agg}": (binned_col, spatial_agg),
                f"spatial_se_{agg}": (binned_col, spatial_se),
            }
        )

    grouped = df.groupby(column, observed=False).agg(**agg_dict).reset_index()
    return grouped


##############################################################################
# Functions to export processed data
##############################################################################


def _check_pipeline(pipeline: PipelineData, field_name: str):
    valid_fields = {f.name for f in fields(pipeline)}
    if field_name not in valid_fields:
        raise AttributeError(f"'{field_name}' is not a field on PipelineData.")
    if getattr(pipeline, field_name) is None:
        raise ValueError(f"'{field_name}' is not available in the pipeline.")


def _check_path(path: str) -> bool:
    return os.path.exists(path)


def _build_path(output_folder: str, field_name: str, prefix: str, ext: str) -> str:
    return os.path.join(output_folder, f"{field_name}{prefix}{ext}")


def export_train(pipeline: PipelineData, output_folder: str, prefix: str):
    exports = {
        "X_train": (".pkl", lambda val, path: val.to_pickle(path)),
        "X_test": (".pkl", lambda val, path: val.to_pickle(path)),
        "y_train": (".npy", lambda val, path: np.save(path, val)),
        "y_test": (".npy", lambda val, path: np.save(path, val)),
        "df_final": (".pkl", lambda val, path: val.to_pickle(path)),
        "adata_final": (".h5ad", lambda val, path: val.write_h5ad(path)),
        "barcode_train": (".npy", lambda val, path: np.save(path, val)),
        "barcode_test": (".npy", lambda val, path: np.save(path, val)),
    }

    os.makedirs(output_folder, exist_ok=True)

    for field_name, (ext, save_fn) in exports.items():
        path = _build_path(output_folder, field_name, prefix, ext)
        _check_pipeline(pipeline, field_name)
        if _check_path(path):
            print(f"Skipping {field_name} — {path} already exists.")
            continue
        save_fn(getattr(pipeline, field_name), path)
        print(f"Exported {field_name} -> {path}")


def export_spatial(pipeline: PipelineData, output_folder: str, prefix: str):
    exports = {
        "df_spatial": (".pkl", lambda val, path: val.to_pickle(path)),
        "adata_spatial": (".h5ad", lambda val, path: val.write_h5ad(path)),
    }

    os.makedirs(output_folder, exist_ok=True)

    for field_name, (ext, save_fn) in exports.items():
        path = _build_path(output_folder, field_name, prefix, ext)
        _check_pipeline(pipeline, field_name)
        if _check_path(path):
            print(f"Skipping {field_name} — {path} already exists.")
            continue
        save_fn(getattr(pipeline, field_name), path)
        print(f"Exported {field_name} -> {path}")
