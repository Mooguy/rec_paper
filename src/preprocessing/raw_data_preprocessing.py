import numpy as np
import pandas as pd
import scipy.sparse as sp
from pathlib import Path
import anndata as ad

# ─────────────────────────────────────────────
# CONFIGURATION & CONSTANTS
# ─────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent.parent.parent
RAW_DATA_FOLDER = BASE_DIR / "data" / "raw" 

from src.utils.utils import change_bmp4_conc_idx

BMP4_DICT = {
    "BMP4__00000_04": "BMP4__00000_032",
    "BMP4__00000_20": "BMP4__00000_16",
    "BMP4__00001_00": "BMP4__00000_80",
    "BMP4__00005_00": "BMP4__00004_00",
}


# ─────────────────────────────────────────────
# UTILITY FUNCTIONS
# ─────────────────────────────────────────────
def remap_bmp4_prefix_keep_barcode(idx: str, bmp4_idx_change_dict: dict = BMP4_DICT) -> str:
    """Remap BMP4 prefix while keeping barcode unchanged."""
    if not isinstance(idx, str) or "_" not in idx:
        return idx
    prefix, barcode = idx.rsplit("_", 1)  # split only before barcode
    return f"{bmp4_idx_change_dict.get(prefix, prefix)}_{barcode}"


# ─────────────────────────────────────────────
# DATA LOADING FUNCTIONS
# ─────────────────────────────────────────────
def load_files(DATA_FOLDER=RAW_DATA_FOLDER, exp1=False):
    """Load raw data, features, metadata, and barcodes from pickle files.
    
    Args:
        DATA_FOLDER: Path to data folder
        exp1: If True, load Exp1_2019_Dec; else load Exp3_2021_Feb
        
    Returns:
        Tuple of (raw_data, meta_data, features, barcodes)
    """
    if exp1:
        DATA_FOLDER = DATA_FOLDER / "Exp1_2019_Dec/"
        raw_data = pd.read_pickle(DATA_FOLDER / "data_file.p")
        features = pd.read_pickle(DATA_FOLDER / "feat_file.p")
        meta_data = pd.read_pickle(DATA_FOLDER / "meta_file.p")

        raw_data = raw_data.T.tocsc()

        meta_data["sample_id"] = (
            meta_data["sample_id"]
            .replace("3HR-BMP9-BMP10", "3HR-BMP4-BMP10")
            .replace("3HR-BMP10-GDF5", "3HR-BMP4-GDF5")
            .replace("6HR-BMP9-BMP10", "6HR-BMP4-BMP10")
            .replace("6HR-BMP10-GDF5", "6HR-BMP4-GDF5")
        )

        return raw_data, meta_data, features, None

    else:
        DATA_FOLDER = DATA_FOLDER / "Exp3_2021_Feb/"
        raw_data = pd.read_pickle(DATA_FOLDER / "data_file.p")
        features = pd.read_pickle(DATA_FOLDER / "feat_file.p")
        meta_data = pd.read_pickle(DATA_FOLDER / "meta_file.p")
        barcodes = pd.read_pickle(DATA_FOLDER / "barcodes.p")

        # Change genes to be the columns and cells to be the rows:
        raw_data = raw_data.T.tocsc()

        # Fix gene names
        features[1] = features[1].map(lambda x: x.removeprefix("MM10_"))

        meta_data["sample_id"] = meta_data["sample_id"].map(lambda x: change_bmp4_conc_idx(x))

        return raw_data, meta_data, features, barcodes


# ─────────────────────────────────────────────
# DATA VALIDATION & ORIENTATION FUNCTIONS
# ─────────────────────────────────────────────
def validate_and_prepare_data(raw_data, meta_data, features):
    """Validate data orientation and prepare for processing.
    
    Args:
        raw_data: CSC sparse matrix (genes × cells)
        meta_data: DataFrame with cell metadata
        features: DataFrame with gene features
        
    Returns:
        Tuple of (counts, gene_names, cell_barcodes) with counts as CSR matrix
    """
    # Rename features columns for clarity
    if features.shape[1] == 3:
        features.columns = ["ensembl_id", "gene_name", "feature_type"]
    else:
        features["feature_type"] = "gene"
        features.columns = ["ensembl_id", "gene_name", "feature_type"]
    
    # Convert to CSR for row-wise operations (per-cell)
    counts = raw_data.tocsr()

    # Validate dimensions
    assert counts.shape[0] == meta_data.shape[0], \
        f"Cell count mismatch: matrix has {counts.shape[0]}, metadata has {meta_data.shape[0]}"
    assert counts.shape[1] == features.shape[0], \
        f"Gene count mismatch: matrix has {counts.shape[1]}, features has {features.shape[0]}"

    gene_names = features["gene_name"].values
    cell_barcodes = meta_data["cell_barcode"].values

    print(f"Matrix shape (cells × genes): {counts.shape}")
    
    return counts, gene_names, cell_barcodes, features, meta_data


# ─────────────────────────────────────────────
# FILTERING FUNCTIONS
# ─────────────────────────────────────────────
def apply_qc_filtering(counts, features, meta_data, threshold=4000):
    """Apply quality control filtering based on gene count and mitochondrial percentage.
    
    Args:
        counts: CSR matrix (cells × genes)
        features: DataFrame with gene features
        meta_data: DataFrame with cell metadata
        threshold: Minimum number of genes per cell
        
    Returns:
        Tuple of (filtered_counts, filtered_meta_data, qc_mask, n_genes_per_cell, mito_pct_per_cell)
    """
    # Number of genes with non-zero counts per cell
    count_genes_that_are_not_zero = (counts > 0).sum(axis=1)

    # Mitochondrial gene indices (mouse: mt-, human: MT-)
    mitochondrial_genes_per_cell = np.where(
        features['gene_name'].str.contains("^mt-", case=False)
    )[0]

    # Sum of mito reads per cell
    mitochondrial_sum_reads_per_cell = counts[:, mitochondrial_genes_per_cell].sum(axis=1)

    # Total reads per cell
    total_reads = counts.sum(axis=1)

    # Mito read fraction per cell
    mitochondrial_read_percent_per_cell = mitochondrial_sum_reads_per_cell / total_reads

    # Boolean mask: True = cell passes QC
    qc_mask = np.array(
        (mitochondrial_read_percent_per_cell < 0.1)
        & (count_genes_that_are_not_zero > threshold)
    ).T[0]

    n_failed_qc = (~qc_mask).sum()
    print(f"\nStep 1 — QC cell filtering (genes > {threshold} & mito% < 10%)")
    print(f"  Removed : {n_failed_qc} cells")
    print(f"  Retained: {qc_mask.sum()} cells")

    counts = counts[qc_mask, :]
    meta_data = meta_data[qc_mask].reset_index(drop=True)

    print(f"  Matrix after cell filtering: {counts.shape}")
    
    return counts, meta_data, qc_mask, count_genes_that_are_not_zero, mitochondrial_read_percent_per_cell


def remove_unknown_cells(counts, meta_data, features):
    """Remove cells with unknown sample_id.
    
    Args:
        counts: CSR matrix (cells × genes)
        meta_data: DataFrame with cell metadata
        features: DataFrame with gene features
        
    Returns:
        Tuple of (filtered_counts, filtered_meta_data, filtered_features)
    """
    known_mask = meta_data["sample_id"].values != "unknown"

    n_unknown = (~known_mask).sum()
    print(f"\nStep 2 — Remove unknown cells")
    print(f"  Removed : {n_unknown} cells")
    print(f"  Retained: {known_mask.sum()} cells")

    counts = counts[known_mask, :]
    meta_data = meta_data[known_mask].reset_index(drop=True)

    print(f"  Matrix after removing unknowns: {counts.shape}")
    
    return counts, meta_data, features


def filter_genes(counts, features, meta_data):
    """Remove genes with zero counts across all remaining cells.
    
    Args:
        counts: CSR matrix (cells × genes)
        features: DataFrame with gene features
        meta_data: DataFrame with cell metadata
        
    Returns:
        Tuple of (filtered_counts, filtered_features, filtered_meta_data, gene_names)
    """
    gene_counts = np.asarray(counts.sum(axis=0)).flatten()
    gene_mask = gene_counts > 0

    n_zero_genes = (~gene_mask).sum()
    print(f"\nStep 3 — Gene filtering (remove zero-count genes)")
    print(f"  Removed : {n_zero_genes} genes")
    print(f"  Retained: {gene_mask.sum()} genes")

    counts = counts[:, gene_mask]
    features = features[gene_mask].reset_index(drop=True)
    gene_names = features["gene_name"].values

    print(f"  Matrix after gene filtering: {counts.shape}")
    
    return counts, features, meta_data, gene_names


def merge_duplicate_genes(counts, features, gene_names):
    """Merge duplicate gene symbols by summing their counts.
    
    Args:
        counts: CSR matrix (cells × genes)
        features: DataFrame with gene features
        gene_names: Array of gene names
        
    Returns:
        Tuple of (merged_counts, merged_features, unique_gene_names)
    """
    unique_genes, inverse_idx = np.unique(gene_names, return_inverse=True)

    n_duplicates = len(gene_names) - len(unique_genes)
    print(f"\nStep 4 — Merge duplicate gene symbols")
    print(f"  Genes before: {len(gene_names)}")
    print(f"  Duplicate gene names collapsed: {n_duplicates}")
    print(f"  Genes after : {len(unique_genes)}")

    if n_duplicates > 0:
        # Build a (n_original_genes x n_unique_genes) summation matrix
        n_orig = len(gene_names)
        n_unique = len(unique_genes)
        summation_matrix = sp.csr_matrix(
            (np.ones(n_orig), (np.arange(n_orig), inverse_idx)),
            shape=(n_orig, n_unique)
        )
        counts = counts.dot(summation_matrix)
        
        # Keep only the first occurrence of each unique gene in features
        _, first_indices = np.unique(inverse_idx, return_index=True)
        first_indices = np.sort(first_indices)
        features = features.iloc[first_indices].reset_index(drop=True)
    else:
        features = features.copy()

    features["gene_name"] = unique_genes

    print(f"  Matrix after merging: {counts.shape}")
    
    return counts, features, unique_genes


def apply_cpm_normalization(counts, gene_names, cell_barcodes):
    """Apply CPM (counts per million) normalization.
    
    Args:
        counts: CSR matrix (cells × genes)
        gene_names: Array of gene names
        cell_barcodes: Array of cell barcodes
        
    Returns:
        DataFrame with CPM-normalized counts
    """
    library_sizes = np.asarray(counts.sum(axis=1)).flatten()

    print(f"\nStep 5 — CPM normalisation")
    print(f"  Library size — min: {library_sizes.min()}, "
          f"median: {np.median(library_sizes):.0f}, "
          f"max: {library_sizes.max()}")

    reciprocal = sp.diags(1.0 / library_sizes)
    cpm_sparse = reciprocal.dot(counts) * 1_000_000

    print(f"  ✓ normalized_df shape: {cpm_sparse.shape}")
    
    return cpm_sparse


def make_anndata_object(counts, cpm_counts, gene_names, cell_barcodes, meta_data):
    """Create an AnnData object from counts, gene names, cell barcodes, and metadata.
    
    Args:
        counts: CSR matrix (cells × genes) - raw counts
        cpm_counts: DataFrame with CPM-normalized counts
        gene_names: Array of gene names
        cell_barcodes: Array of cell barcodes
        meta_data: DataFrame with cell metadata
    Returns:
        AnnData object with .X as CPM counts and .raw as raw counts
    """    
    adata = ad.AnnData(X=cpm_counts, obs=meta_data, var=pd.DataFrame(index=gene_names))
    adata.obs_names = cell_barcodes
    adata.var_names = gene_names
    
    adata.layers["raw_counts"] = counts

    print(f"\nStep 6 — Create AnnData object")
    print(f"  ✓ adata shape: {adata.shape}")
    
    return adata