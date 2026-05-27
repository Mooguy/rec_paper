# Rec Paper Analysis

Reproduction and analysis code for the spatial reconstruction and ligand-response project used in the lab paper.  
This repository contains the code, notebooks, and preprocessing utilities used to generate the training and spatial datasets for the figure 5 workflow.

## Overview

This project processes raw expression data, filters cells, prepares training objects, and exports the derived datasets used for modeling and downstream analysis.  
The reusable code lives in the Python package under `src/`, while notebooks in `notebooks/` and `figures/` document the analysis steps and figure-generation workflow.

## Repository Structure

- `src/` - reusable analysis, preprocessing, modeling, and utility code
- `scripts/` - executable scripts for running end-to-end workflows
- `notebooks/` - exploratory and reproducible analysis notebooks
- `figures/` - figure-generation notebooks and plotting helpers
- `data/` - local data inputs and processed outputs
- `models/` - saved trained models and outputs

## Requirements

The project uses a Python environment with the dependencies listed in `setup.py`, including:

- pandas
- numpy
- scipy
- matplotlib
- seaborn
- scanpy
- anndata
- xgboost
- optuna
- scikit-learn
- statsmodels
- gseapy
- pyarrow
- pingouin
- tqdm
- umap-learn
- goatools

## Setup

Create and activate a Python environment, then install the package in editable mode:

```bash
pip install -e .
```

If you are setting up a fresh environment, install any missing dependencies first or use the environment management workflow used in the lab.

## Data

The analysis expects local data files in `data/external/` and generated outputs in `data/processed/`.

Typical inputs include:

- `table_B_scRNAseq_UMI_counts.tsv`
- `cell_zone_table_with_crypt.txt`
- `landmark_genes.txt`
- raw sequencing and metadata files under `data/raw/`

Large raw or processed files are not necessarily tracked in Git and may need to be obtained separately.

## Main Workflow

The figure 5 preprocessing workflow is documented in `notebooks/fig5_data_processing.ipynb`.

That workflow:

1. loads raw data files
2. generates an AnnData object
3. applies baseline filtering
4. removes unknown samples and duplicates
5. separates H2B-related expression
6. prepares the final training object
7. exports training and spatial datasets

## Scripts

Useful project scripts include:

- `scripts/run_all_ligands_pipeline.py`
- `scripts/run_model_training.py`
- `scripts/run_corr_ks_analysis.py`
- `scripts/generate_fc_all_genes_df.py`
- `scripts/ks_fdr.py`

These scripts support the main analysis, model training, and downstream statistical testing steps.

## Notebooks

Key notebooks include:

- `notebooks/fig5_data_processing.ipynb`
- `notebooks/raw_data_preprocessing.ipynb`
- `notebooks/ks_test_fdr.ipynb`

The `figures/` folder contains figure-specific notebooks and plotting helpers for the paper outputs.

## Outputs

Generated training and spatial datasets are written to:

- `data/processed/fig5_data/`

Model artifacts may be saved in:

- `models/`

## Reproducibility Notes

- The repository uses local file paths resolved from the project root.
- Some inputs are lab-specific and may not be publicly distributable.
- Saved model artifacts and processed data may need to be regenerated from the raw inputs.

## Contact

For questions about the analysis or data layout, contact the project maintainers in the lab.
