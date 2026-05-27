import os 
from pathlib import Path 
import pandas as pd 
import argparse 

from src.utils.utils import load_ligand_models 

BASE_DIR = Path(__file__).resolve().parent.parent 
DATA_DIR = BASE_DIR / "data" / "processed" 

print("Loading FourPLGeneModel class...")

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--subset_genes",
        type=str,
        default="full",
        choices=["full", "sig"],
        help="Subset of genes to use: 'full' or 'sig' (default: 'full')",
    )
    parser.add_argument(
        "--method", default="mean", choices=["mean", "pb", "cpm"], help="Method for fitting 4PL model (default: mean)"
    )
    return parser.parse_args()


def generate_fold_all_genes_fc_df(args):
    models = load_ligand_models()
    bmp4_4pl_model = models["BMP4"]
    tgfb_4pl_model = models["TGFb1"]
    bmp6_4pl_model = models["BMP6"]
    bmp10_4pl_model = models["BMP10"]
    bmp9_4pl_model = models["BMP9"]
    gdf5_4pl_model = models["GDF5"]

    del models

    if args.subset_genes == "full":
        print("Using full set of genes.")
        params_list = [
            bmp4_4pl_model.params_all_genes,
            tgfb_4pl_model.params_all_genes,
            bmp6_4pl_model.params_all_genes,
            bmp10_4pl_model.params_all_genes,
            bmp9_4pl_model.params_all_genes,
            gdf5_4pl_model.params_all_genes,
        ]

    elif args.subset_genes == "sig":
        print("Using significant genes only.")
        params_list = [
            bmp4_4pl_model.params_all_genes.loc[bmp9_4pl_model.all_genes_list],
            tgfb_4pl_model.params_all_genes.loc[bmp9_4pl_model.all_genes_list],
            bmp6_4pl_model.params_all_genes.loc[bmp9_4pl_model.all_genes_list],
            bmp10_4pl_model.params_all_genes.loc[bmp9_4pl_model.all_genes_list],
            bmp9_4pl_model.params_all_genes.loc[bmp9_4pl_model.all_genes_list],
            gdf5_4pl_model.params_all_genes.loc[bmp9_4pl_model.all_genes_list],
        ]

    else:
        raise ValueError("subset_genes must be 'full' or 'sig'.")
        return

    ligands = [
        "BMP4",
        "TGFb1",
        "BMP6",
        "BMP10",
        "BMP9",
        "GDF5",
    ]

    fc_df_new = pd.DataFrame(columns=ligands, index=params_list[0].index)
    for i, ligand in enumerate(ligands):
        fc_df_new[ligand] = params_list[i]["log2fc_cf"]
    fc_df_new = fc_df_new.astype(float)

    fc_df_new.to_parquet(DATA_DIR / ("fc_all_genes_df_" + args.subset_genes + ".parquet"), compression="snappy")


if __name__ == "__main__":
    args = parse_args()

    generate_fold_all_genes_fc_df(args)
    print("Fold change all genes DataFrame generated and saved to Parquet.")