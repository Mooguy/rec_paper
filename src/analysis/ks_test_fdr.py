import numpy as np
import pandas as pd
from statsmodels.stats.multitest import multipletests
from scipy import stats
from tqdm import tqdm

def generate_ks_test_pvals(df):
    mask = ~df.index.to_series().str.contains("CTRL")
    conditions_no_ctrl = df.loc[mask].index.unique()
    df_columns = df.columns.values

    ctrl_rows = df.index.str.contains("CTRL")

    KS_pvals = []
    for gene in tqdm(range(df.shape[1])):
        curr_vec = []
        for cond in conditions_no_ctrl:
            currenct_ctrl = df.iloc[:, gene][ctrl_rows]
            current_ctrl_val = currenct_ctrl.values

            currenct_cond = df.iloc[:, gene][df.index == cond]
            current_cond_val = currenct_cond.values

            current_pval = stats.ks_2samp(current_ctrl_val, current_cond_val).pvalue
            curr_vec.append(current_pval)

        KS_pvals.append(curr_vec)

    KS_pvals = np.array(KS_pvals)

    return KS_pvals, conditions_no_ctrl, df_columns


def generate_fdr_pvals(KS_pvals, conditions_no_ctrl, df_columns):
    ks_pvals_df = pd.DataFrame(KS_pvals.T, index=conditions_no_ctrl, columns=df_columns)

    # Extract the p-values from the dataframe
    p_values_raw = ks_pvals_df.values.flatten()

    # Apply Benjamini-Hochberg correction
    _, FDR_corrected_p_values, _, _ = multipletests(p_values_raw, method="fdr_bh")

    # Reshape the corrected p-values back to the original dataframe shape
    ks_fdr_corrected_pvals_df = pd.DataFrame(
        FDR_corrected_p_values.reshape(ks_pvals_df.shape),
        columns=ks_pvals_df.columns,
        index=ks_pvals_df.index,
    )

    return ks_fdr_corrected_pvals_df


def get_significant_genes(ks_fdr_corrected_pvals_df, alpha=0.05):
    # Iterate over the genes (columns)
    significant_genes_FDR_correction = []

    for gene in ks_fdr_corrected_pvals_df.columns:
        # Check if any condition has a p-value less than the significance threshold (e.g., 0.05)
        if (ks_fdr_corrected_pvals_df[gene] < alpha).any().any():
            significant_genes_FDR_correction.append(gene)

    return significant_genes_FDR_correction
