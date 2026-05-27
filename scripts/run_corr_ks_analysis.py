from __future__ import annotations 

import logging 
import pickle
import argparse
import pandas as pd 
from pathlib import Path

# from src.analysis.Ligand4PL import FourPLGeneModel
from src.utils.utils import load_ligand_models
from figures.fig_scripts.fig4_functions import ( 
    prepare_ligand_data_from_model, 
    process_all_ligands_corr, 
    process_all_ligands_ks, 
)

BASE_DIR = Path(__file__).resolve().parent.parent 
DATA_DIR = BASE_DIR / "data" / "processed" 

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--method", default=None, choices=["mean", "pb", "cpm"], help="Method for fitting 4PL model (default: mean)"
    )
    return parser.parse_args()

logger = logging.getLogger(__name__)

def setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )

def save_pickle(obj: object, path: Path) -> None:
    with open(path, "wb") as f:
        pickle.dump(obj, f)

def build_ligand_input_dicts(ligands: list[str], method: str) -> tuple[dict, dict]:
    logger.info("Loading ligand models...")
    df = pd.read_parquet(DATA_DIR / "cpm_df.parquet")
    # models = {ligand: FourPLGeneModel(ligand, df) for ligand in LIGANDS}
    models = load_ligand_models(subset=[], test=method)

    all_ligand_df_dict = {}
    ligand_pc1_dict = {}

    for ligand in ligands:
        logger.info("Preparing data for %s...", ligand)
        norm_df, lig_pc1, _ = prepare_ligand_data_from_model(models[ligand])
        all_ligand_df_dict[ligand] = norm_df
        ligand_pc1_dict[ligand] = lig_pc1

    del models, df
    return all_ligand_df_dict, ligand_pc1_dict


def main() -> None:
    setup_logging()

    args = parse_args()

    LIGANDS_DICT_PATH = DATA_DIR / f"all_ligand_df_dict.pkl"
    LIGAND_PC1_DICT_PATH = DATA_DIR / f"ligand_pc1_dict.pkl"
    KS_RESULTS_PATH = DATA_DIR / f"ks_test_results_all_genes.pkl"
    PCORR_RESULTS_PATH = DATA_DIR / f"global_partial_corr_genes_dict.pkl"

    LIGANDS = ["BMP4", "BMP10", "BMP6", "BMP9", "GDF5", "TGFb1"]

    if LIGANDS_DICT_PATH.exists() and LIGAND_PC1_DICT_PATH.exists():
        logger.info("Ligand dictionaries already exist. Loading from disk...")
        with open(LIGANDS_DICT_PATH, "rb") as f:
            all_ligand_df_dict = pickle.load(f)
        with open(LIGAND_PC1_DICT_PATH, "rb") as f:
            ligand_pc1_dict = pickle.load(f)
    else:
        logger.info("Generating ligand dictionaries in this script...")
        all_ligand_df_dict, ligand_pc1_dict = build_ligand_input_dicts(LIGANDS, args.method)
        save_pickle(all_ligand_df_dict, LIGANDS_DICT_PATH)
        save_pickle(ligand_pc1_dict, LIGAND_PC1_DICT_PATH)

    logger.info("Running KS tests...")
    ks_results_dict_genes = process_all_ligands_ks(
        all_ligand_df_dict,
        show_progress=True,
    )
    save_pickle(ks_results_dict_genes, KS_RESULTS_PATH)
    logger.info("Saved KS results to %s", KS_RESULTS_PATH)

    logger.info("Running partial correlations...")
    global_partial_corr_genes_dict = process_all_ligands_corr(
        all_ligand_df_dict,
        ligand_pc1_dict,
    )
    save_pickle(global_partial_corr_genes_dict, PCORR_RESULTS_PATH)
    logger.info("Saved partial correlations to %s", PCORR_RESULTS_PATH)


if __name__ == "__main__":
    main()