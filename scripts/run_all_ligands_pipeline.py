import os
from pathlib import Path
import sys
import pickle
import pandas as pd
import argparse

from src.analysis.Ligand4PL import FourPLGeneModel

BASE_DIR = Path(__file__).resolve().parent.parent 
DATA_DIR = BASE_DIR / "data" / "processed" 

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--overwrite", action="store_true", help="Overwrite existing model files"
    )
    parser.add_argument(
        "--method", default="mean", choices=["mean", "pb", "cpm"], help="Method for fitting 4PL model (default: mean)"
    )
    return parser.parse_args()


def main():
    args = parse_args()

    df = pd.read_parquet(DATA_DIR / "cpm_df.parquet")

    ligands = ["BMP9", "BMP6", "BMP4", "TGFb1", "BMP10", "GDF5"]

    if args.method == "cpm":
        dir = DATA_DIR / f"ligand_model"
    else:
        dir = DATA_DIR / f"ligand_model_{args.method}"
    os.makedirs(dir, exist_ok=True)

    for ligand in ligands:
        print(f"Processing ligand: {ligand}")
        model = FourPLGeneModel(ligand, df)
        file_path = dir / f"class_object_{ligand}.pkl"

        if os.path.exists(file_path) and not args.overwrite:
            print(f"Model for {ligand} already exists. Skipping save.")
            continue

        try:
            with open(file_path, "wb") as f:
                pickle.dump(model, f)
            print(f"Saved model for {ligand}")

        except Exception as e:
            print(f"Error saving model for {ligand}: {e}")


if __name__ == "__main__":
    main()
