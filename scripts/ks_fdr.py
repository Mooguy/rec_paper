from pathlib import Path 

from src.utils.utils import find_project_root

BASE_DIR = find_project_root()
DATA_DIR = BASE_DIR / "data" / "processed"

from src.analysis.ks_test_fdr import * 

df = pd.read_parquet(DATA_DIR / "cpm_df.parquet")

def main():
    print("Generating KS test p-values...")
    KS_pvals, conditions_no_ctrl, df_columns = generate_ks_test_pvals(
        df
    )
    print("Applying FDR correction to KS test p-values...")
    ks_fdr_corrected_pvals_df = generate_fdr_pvals(KS_pvals, conditions_no_ctrl, df_columns)
    print("Saving KS test FDR-corrected p-values to parquet...")
    ks_fdr_corrected_pvals_df.to_parquet(Path(DATA_DIR) / "ks_fdr_corrected_pvals_cpm.parquet", compression='snappy')

if __name__ == "__main__":
    main()