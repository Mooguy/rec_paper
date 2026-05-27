import numpy as np
import pandas as pd
import warnings

from scipy.optimize import curve_fit
from pathlib import Path
from sklearn.decomposition import PCA

from src.utils.utils import add_df_ligand_names_and_concentrations_columns, timeit

BASE_DIR = Path(__file__).resolve().parent.parent.parent
DATA_DIR = BASE_DIR / "data" / "processed"

class FourPLGeneModel:
    def __init__(self, ligand, df, raw_df=None):
        self.ligand = ligand
        print(f"Initializing FourPLGeneModel for {ligand}...")
        self.df = df
        self.raw_df = raw_df
        self.ligands = ["BMP4", "BMP6", "BMP9", "BMP10", "GDF5", "TGFb1"]
        self.load_ks_fdr()
        self.get_pb_and_fc_values()
        self.get_significant_genes()
        self.list_sig_genes()
        self.run_fitted_pca()
        self.run_fitted_sc_pca()
        self.fit_pc1_to_4pl()
        self.generate_all_genes_params()
        self.get_pb_fc_values()

    """
    loading dataframes with corrected p-values for each gene in each ligand condition, 
    and creating a list of genes that are significant in at least one ligand condition
    """
    def load_ks_fdr(self):
        self.ks_fdr = pd.read_parquet(
            DATA_DIR / "ks_fdr_corrected_pvals.parquet"
        )
        self.ks_fdr_gene_list = self.ks_fdr.loc[
            :, (self.ks_fdr < 0.05).any()
        ].columns.tolist()


    def get_pb_and_fc_values(self):
        self.df_log = np.log2(self.df + 1)
        self.df_mean = self.df_log.groupby(self.df_log.index).mean()
        self.df_ctrl = self.df_mean[self.df_mean.index.str.contains("CTRL")]
        self.mean_df_ctrl_mean = self.df_ctrl.mean()
        self.mean_df_ctrl_mean = pd.DataFrame([self.mean_df_ctrl_mean], index=["CTRL"])
        self.df_mean = pd.concat(
            [
                self.mean_df_ctrl_mean,
                self.df_mean[~self.df_mean.index.str.contains("CTRL")],
            ]
        )
        self.df_mean_fdr = self.df_mean.loc[self.df_mean.index.str.contains(f"{self.ligand}|CTRL"),
                                             self.ks_fdr_gene_list]

        ctrl_row = self.df_mean.loc["CTRL"]                   
        self.df_fc = self.df_mean.drop("CTRL") - ctrl_row

        
    """
    Identifying significant genes based on log2 fold change and corrected p-values, 
    and creating a dictionary of significant genes for each ligand condition, 
    separated into upregulated and downregulated genes
    """
    @timeit
    def get_significant_genes(self):
        self.ks_fdr = self.ks_fdr.reindex(
            index=self.df_fc.index,
            columns=self.df_fc.columns,
        )

        sig_mask = self.ks_fdr < 0.05
        self.up_genes = self.df_fc.ge(1) & sig_mask
        self.down_genes = self.df_fc.le(-1) & sig_mask

        self.sig_df = self.up_genes + self.down_genes

        self.sig_genes_dict = {}

        for lig in self.ligands:
            up_genes_lig = self.up_genes.loc[
                :, self.up_genes[self.up_genes.index.str.contains(lig)].any()
            ].columns.tolist()
            down_genes_lig = self.down_genes.loc[
                :, self.down_genes[self.down_genes.index.str.contains(lig)].any()
            ].columns.tolist()
            self.sig_genes_dict[lig] = {
                "up": up_genes_lig,
                "down": down_genes_lig,
            }

    """
    Creating a list of all significant genes across all ligand conditions,
    and a list of significant genes for the specific ligand being analyzed
    """
    def list_sig_genes(self):
        all_genes = set()
        for ligand in self.sig_genes_dict:
            all_genes.update(self.sig_genes_dict[ligand]["up"])
            all_genes.update(self.sig_genes_dict[ligand]["down"])

        self.all_genes_list = list(all_genes)
        self.ligand_sig_genes_from_dict = (
            self.sig_genes_dict[self.ligand]["up"]
            + self.sig_genes_dict[self.ligand]["down"]
        )
        self.df_fc_ligand_sig = self.df_fc.loc[
            self.df_fc.index.str.contains(f"{self.ligand}"),
            self.ligand_sig_genes_from_dict,
        ]    

    """
    Defining the 4-parameter logistic function to be used for curve fitting,
    with error handling to catch any runtime warnings that may occur during the fitting process
    """
    @staticmethod
    def four_parameter_logistic(input, initial, final, ec50, n):
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always", RuntimeWarning)
            result = final + (initial - final) / (1 + (input / ec50) ** n)
            for warning in w:
                if issubclass(warning.category, RuntimeWarning):
                    print(f"RuntimeWarning in four_parameter_logistic.")
            return result

    """
    Aggregating the control conditions to calculate their mean expression
    """
    @staticmethod
    def aggregate_ctrl(df):
        df_ctrl = df.loc[df.index.str.contains("CTRL")]
        df_ctrl_mean = df_ctrl.groupby(df_ctrl.index).mean().mean(axis=0)
        df_ctrl_mean = pd.DataFrame([df_ctrl_mean], index=["CTRL"])
        return df_ctrl_mean
    
    def _fit_curve(self, x, y, model_fn, guess, bounds, nan_fallback):
        """Shared curve fitting wrapper with error handling."""
        try:
            params, _ = curve_fit(
                model_fn,
                x,
                y,
                p0=guess,
                bounds=bounds,
                method="trf",
                maxfev=10000
            )
            return params
        except RuntimeError as e:
            print(f"Curve fitting failed: {e}")
            return np.full(nan_fallback, np.nan)


    def get_gene_params(self, gene):
        y_data = self.df_mean_fdr[gene].values
        y_min, y_max = y_data.min(), y_data.max()
        y_scaled = (y_data - y_min) / (y_max - y_min)

        model_fn = lambda x, a, b: self.four_parameter_logistic(
            x, a, b, self.ec50_fixed, self.n_fixed
        )
        guess = [y_scaled[0], y_scaled[-1]]
        bounds = ([0, 0], [np.inf, np.inf])

        geneparams = self._fit_curve(self.concentrations, y_scaled, model_fn, guess, bounds, nan_fallback=2)
        geneparams_rescaled = y_min + geneparams * (y_max - y_min)

        gene_norm_exp = None
        if not np.isnan(geneparams_rescaled).any():
            gene_norm_exp = self.get_norm_exp(gene, geneparams_rescaled)

        return geneparams_rescaled, gene_norm_exp

    @timeit
    def fit_pc1_to_4pl(self):
        self.pc1_vec = self.pca_mean_fitted_df["PC1"].values
        self.concentrations = self.pca_mean_fitted_df["concentration"].values

        guess = [self.pc1_vec[0], self.pc1_vec[-1], np.median(self.concentrations), 1]
        bounds = ([-np.inf] * 4, [np.inf] * 4)

        self.pc1_params = self._fit_curve(
            self.concentrations, self.pc1_vec,
            self.four_parameter_logistic, guess, bounds, nan_fallback=4
        )

    """
    Generating PCA components for the mean expression profiles of the ligand and control conditions,
    and calculating the explained variance ratio for each principal component
    """
    @timeit
    def run_fitted_pca(self, n_components=8):
        self.df_mean_filt = self.df_mean.loc[
            self.df_mean.index.str.contains(f"{self.ligand}|CTRL"),
            self.ligand_sig_genes_from_dict,
        ]
        self.pca_model_mean = PCA(n_components=n_components)
        self.pca_model_mean = self.pca_model_mean.fit(self.df_mean_filt)
        self.pca_mean_fitted = self.pca_model_mean.transform(self.df_mean_filt)
        self.pca_mean_fitted_df = pd.DataFrame(
            self.pca_mean_fitted,
            index=self.df_mean_filt.index,
            columns=[f"PC{i + 1}" for i in range(n_components)],
        )
        self.pca_mean_fitted_df = add_df_ligand_names_and_concentrations_columns(
            self.pca_mean_fitted_df
        )
        self.exp_var_dict = self.calc_explained_variance_ratio(self.df_mean_filt)

    """
    Normalizing the pseudo-bulk expression values for each gene based on the fitted 4-parameter logistic curve parameters,
    and calculating the normalized expression values for each gene across the different conditions
    """
    def get_norm_exp(self, gene, gene_params):
        gene_exp = self.df_mean_fdr[gene].values
        norm_exp = (gene_exp - gene_params[0]) / (gene_params[1] - gene_params[0])
        return norm_exp

    """
    Generating the parameters for the 4-parameter logistic curve fitting for each gene
    """    
    @timeit
    def generate_all_genes_params(self):
        self.ec50_fixed, self.n_fixed = self.pc1_params[2], self.pc1_params[3]
        self.params_all_genes = pd.DataFrame(
            columns=[
                "initial_val",
                "final_val",
                "initial_cf",
                "final_cf",
                "direction",
            ]
        )
        self.norm_exp_all_genes = {}

        for gene in self.df_mean_fdr.columns:
            initial_val = self.df_mean_fdr[gene].iloc[0]
            final_val = self.df_mean_fdr[gene].iloc[-1]
            gene_params, gene_norm_exp = self.get_gene_params(gene)

            init, final = gene_params

            self.params_all_genes.loc[gene] = [
                initial_val,
                final_val,
                init,
                final,
                np.nan,
            ]
            self.norm_exp_all_genes[gene] = gene_norm_exp
        self.params_all_genes[
            [
                "initial_cf",
                "final_cf",
            ]
        ] = self.params_all_genes[
            [
                "initial_cf",
                "final_cf",
            ]

        ].astype(float)
        self.norm_exp_all_genes = pd.DataFrame(
            self.norm_exp_all_genes, index=self.df_mean_fdr.index
        )

    @timeit
    def get_pb_fc_values(self):
        self.params_all_genes["log2fc_cf"] = self.params_all_genes["final_cf"] - self.params_all_genes["initial_cf"]

        self.params_all_genes["direction"] = self.params_all_genes.index.map(
            lambda idx: "up"
            if self.params_all_genes.loc[idx, "log2fc_cf"] > 1
            else "down"
            if self.params_all_genes.loc[idx, "log2fc_cf"] < -1
            else "neutral"
        )

    def calc_explained_variance_ratio(self, log_df, all_sig=False):
        if all_sig:
            log_df_centered = log_df - self.pca_model_mean_all_sig.mean_
            pca_components = self.pca_model_mean_all_sig.components_
        else:
            log_df_centered = log_df - self.pca_model_mean.mean_
            pca_components = self.pca_model_mean.components_
        explained_variance = np.var(log_df_centered @ pca_components.T, axis=0, ddof=1)
        explained_variance_ratio = (
            explained_variance / log_df_centered.var(axis=0, ddof=1).sum()
        ) * 100
        explained_variance_ratio_df = {
            f"PC{i + 1}": explained_variance_ratio[i]
            for i in range(len(explained_variance_ratio))
        }
        return explained_variance_ratio_df
    
    # def get_ctrl_for_ligand(self):
    #     ctrl_lig_dict = {
    #         "GDF5":"CTRL_1",
    #         "BMP10":"CTRL_2",
    #         "TGFb1":"CTRL_3",
    #         "BMP6":"CTRL_4",
    #         "BMP9":"CTRL_5",
    #         "BMP4":"CTRL_6",a
    #     }

    #     return ctrl_lig_dict[self.ligand]

    def run_fitted_sc_pca(self, n_components=8):
        self.df_log_filt = self.df_log.loc[
            self.df_log.index.str.contains(f"{self.ligand}|CTRL_1"),
            self.ligand_sig_genes_from_dict,
        ]
        self.pca_sc_fitted = self.pca_model_mean.transform(self.df_log_filt)
        self.pca_sc_fitted_df = pd.DataFrame(
            self.pca_sc_fitted,
            index=self.df_log_filt.index,
            columns=[f"PC{i + 1}" for i in range(n_components)],
        )
        self.pca_sc_fitted_df = add_df_ligand_names_and_concentrations_columns(
            self.pca_sc_fitted_df
        )
        self.exp_var_sc_dict = self.calc_explained_variance_ratio(self.df_log_filt)
