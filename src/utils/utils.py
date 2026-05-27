from pathlib import Path
import re
import numpy as np
import pandas as pd
import os
import pickle
import sys
import glob
import time
import matplotlib.pyplot as plt 
import seaborn as sns
from functools import wraps

BASE_DIR = Path(__file__).resolve().parent.parent.parent 
DATA_DIR = BASE_DIR / "data" / "processed"

ligands = ["BMP10", "BMP4", "BMP9", "BMP6", "TGFb1", "GDF5"]


def find_project_root():
    current = Path.cwd()
    for parent in [current, *current.parents]:
        if (parent / "setup.py").exists():
            return parent
    raise FileNotFoundError("Could not find project root (no setup.py found)")

def set_style():
    """Sets global matplotlib and seaborn styles for the project."""
    
    plt.rc("pdf", fonttype=42) 
    plt.rc("ps", fonttype=42) 

    plt.style.use("ggplot") 
    sns.set_style("whitegrid") 
    

def change_bmp4_conc(df):
    bmp4_idx_change_dict = {
        "BMP4__00000_04": "BMP4__00000_032",
        "BMP4__00000_20": "BMP4__00000_16",
        "BMP4__00001_00": "BMP4__00000_80",
        "BMP4__00005_00": "BMP4__00004_00",
    }

    df = df.rename(index=bmp4_idx_change_dict)

    return df

def change_bmp4_conc_idx(idx):
    bmp4_idx_change_dict = {
        "BMP4__00000_04": "BMP4__00000_032",
        "BMP4__00000_20": "BMP4__00000_16",
        "BMP4__00001_00": "BMP4__00000_80",
        "BMP4__00005_00": "BMP4__00004_00",
    }

    return bmp4_idx_change_dict.get(idx, idx)

# Extract and process concentrations
def extract_concentration(idx, ligand):
    # Replace underscores with dots, remove the ligand name, and strip leading zeros
    modified_idx = re.sub(r"\.(?=[^.]*\.)", "", idx.replace("_", "."))
    concentration = modified_idx.replace(ligand, "").lstrip("0")
    return concentration


def add_df_ligand_names_and_concentrations_columns(df):
    # Ensure df is a copy to avoid SettingWithCopyWarning
    df = df.copy()

    # Extract the ligand name
    df["ligand"] = df.index.map(lambda x: re.sub(r"_[0-9]+", "", x).replace("_", ""))
    df["ligand"] = np.where(df["ligand"] == "CTRL", df.index, df["ligand"])

    df["concentration"] = df.apply(
        lambda row: extract_concentration(row.name, row["ligand"]), axis=1
    )

    # Convert concentrations to float with error handling
    def convert_to_float(x):
        try:
            return float(x) if re.match(r"^-?\d*\.?\d*$", x) else 0.0
        except ValueError:
            return 0.0

    df["concentration"] = df["concentration"].map(convert_to_float)

    return df


def get_ligand_concentrations_columns(df, order_cat=False):
    df["idx"] = df.index
    df["ligand"] = df["idx"].str.split("_", expand=True)[0]

    df["concentration"] = df.apply(
        lambda row: extract_concentration(row["idx"], row["ligand"]), axis=1
    )

    df.drop(["idx"], axis=1, inplace=True)

    df["concentration"] = df["concentration"].map(
        lambda x: "0" + x if x[0] == "." else x
    )

    df.loc[df["ligand"] == "CTRL", "concentration"] = "0"

    df["lig_conc"] = df.apply(
        lambda row: row["ligand"] + "_" + row["concentration"]
        if row["ligand"] != "CTRL"
        else row["ligand"],
        axis=1,
    )

    df["concentration"] = df["concentration"].astype(float)

    if order_cat:
        df["concentration"] = df["concentration"].astype("category")

        sorted_categories = sorted(
            df["concentration"].cat.categories, key=lambda x: float(x)
        )

        df["concentration"] = df["concentration"].cat.reorder_categories(
            sorted_categories, ordered=True
        )

    return df


def extract_time_and_ligand(df):
    """
    Extracts 'time_hr' and 'ligand' from the index of the DataFrame.
    Modifies 'ligand' to include 'time_hr' if ligand is 'CTRL'.

    Parameters:
        df (pd.DataFrame): DataFrame with string indices containing time and ligand information.

    Returns:
        pd.DataFrame: Modified DataFrame with 'time_hr' and 'ligand' columns.
    """
    # Split the index and create 'time_hr' and 'ligand' columns
    df["time_hr"] = df.index.str.split("-").str[0]

    # Determine 'ligand' based on the number of split values
    df["ligand"] = df.index.str.split("-").map(
        lambda x: x[1] if len(x) == 2 else f"{x[1]}_{x[2]}"
    )

    # If ligand is 'CTRL', add 'time_hr' to it
    df["ligand"] = df.apply(
        lambda row: f"{row['time_hr']}_CTRL"
        if row["ligand"] == "CTRL"
        else row["ligand"],
        axis=1,
    )

    return df


def save_var(
    var,
    var_name,
    path=DATA_DIR,
    file_format="pkl",
):
    """
    Saves a variable to disk in a specified format based on its type.

    Parameters:
    - var: The variable to be saved (supports pandas DataFrame, numpy array, list, and dict).
    - var_name: The name of the file (without extension).
    - path: The directory path where the file will be saved. Default is set to a predefined path.
    - file_format: The format for saving DataFrames ('pkl' or 'csv'). Default is 'pkl'.
    """

    # Ensure the directory exists
    os.makedirs(path, exist_ok=True)

    # Full path to the file without extension
    file_path = os.path.join(path, var_name)

    # Check if file already exists
    if any(os.path.exists(f"{file_path}{ext}") for ext in [".csv", ".npy", ".pkl"]):
        print(f"File '{var_name}' already exists. Skipping save.")
        return

    try:
        # Save based on type
        if isinstance(var, pd.DataFrame):
            if file_format == "csv":
                file_path += ".csv"
                var.to_csv(file_path, index=True)
                print(f"DataFrame saved as CSV at: {file_path}")
            else:
                file_path += ".pkl"
                with open(file_path, "wb") as f:
                    var.to_pickle(file_path)
                print(f"DataFrame saved as pickle at: {file_path}")

        elif isinstance(var, np.ndarray):
            file_path += ".npy"
            np.save(file_path, var)
            print(f"NumPy array saved as NPY at: {file_path}")

        elif isinstance(var, list):
            file_path += ".pkl"
            with open(file_path, "wb") as f:
                var.to_pickle(file_path)
            print(f"List saved as pickle at: {file_path}")

        elif isinstance(var, dict):
            file_path += ".pkl"
            with open(file_path, "wb") as f:
                pickle.dump(var, f)
            print(f"Dictionary saved as pickle at: {file_path}")

        else:
            print(f"Unsupported variable type: {type(var)}. Cannot save.")

    except Exception as e:
        print(f"An error occurred while saving '{var_name}': {e}")


def load_df_with_fdr_genes(
    DATA_FOLDER=DATA_DIR,
    fdr=True,
):
    # Load not normalized data:
    df = pd.read_pickle(DATA_FOLDER + "df_not_normalized_data_index.pkl")

    if fdr:
        fdr_list = pd.read_pickle(DATA_FOLDER + "significant_genes_FDR_correction.pkl")
        df = df[fdr_list]
        df = change_bmp4_conc(df)

        return df
    else:
        df = change_bmp4_conc(df)
        return df


def get_fc(df, log_addition=1):
    df_mean = df.groupby(df.index).mean()
    df_ctrl = df_mean[df_mean.index.str.contains("CTRL")]
    mean_df_ctrl_mean = df_ctrl.mean()
    mean_df_ligand_no_ctrl = df_mean[~df_mean.index.str.contains("CTRL")]
    fold_change = np.log2(
        (mean_df_ligand_no_ctrl + log_addition) / (mean_df_ctrl_mean + log_addition)
    )
    fold_change = fold_change.loc[:, ~fold_change.columns.duplicated()]

    return fold_change


def get_significant_genes(ligand, df, conc_per_ligand=2, log_addition=1e-3):
    df_filt = df = df[df.index.str.contains(f"{ligand}|CTRL")]
    fold_change = get_fc(df_filt, log_addition=log_addition)

    significant_up_genes = [
        gene
        for gene in fold_change.columns
        if (fold_change[gene] > 1).sum() >= conc_per_ligand
    ]
    significant_down_genes = [
        gene
        for gene in fold_change.columns
        if (fold_change[gene] < -1).sum() >= conc_per_ligand
    ]

    return significant_up_genes, significant_down_genes, fold_change


def significant_genes_dict(df, log_addition=1e-3):
    ligands = ["BMP10", "BMP4", "BMP9", "BMP6", "TGFb1", "GDF5"]
    sig_genes_dict = {}

    for ligand in ligands:
        sig_up, sig_down, df_fold_change = get_significant_genes(
            ligand, df, log_addition=1e-3
        )
        sig_genes_dict[ligand] = {"up": sig_up, "down": sig_down}

    return sig_genes_dict


def list_all_sig_genes(sig_genes_dict):
    all_sig_genes = []
    for ligand in sig_genes_dict.keys():
        all_sig_genes.extend(sig_genes_dict[ligand]["up"])
        all_sig_genes.extend(sig_genes_dict[ligand]["down"])

    all_sig_genes = list(dict.fromkeys(all_sig_genes))
    return all_sig_genes


def seperate_index_barcodes(df):
    # check if / is in the index:
    if df.index.str.contains("/").any():
        df = df.copy()  # Avoid modifying the original DataFrame
        df["barcodes"] = df.index.str.split("/").str[1]

        # Reorder columns to place 'barcodes' first
        cols = ["barcodes"] + df.columns[:-1].tolist()
        df = df[cols]

        # Set index with the first part of the index:
        df.index = df.index.str.split("/").str[0]

    return df


def show_largest_variables(local_vars, top_n=10, exclude_prefixes=("_", "__")):
    """
    Display the largest user-defined variables in memory by their size.

    Args:
        local_vars (dict): A dictionary of local variables (usually `locals()`).
        top_n (int): The number of top variables to display by size.
        exclude_prefixes (tuple): Variable name prefixes to exclude (e.g., system variables).
    """

    def sizeof_fmt(num, suffix="B"):
        """
        Format a size in bytes into a human-readable format.
        """
        for unit in ["", "Ki", "Mi", "Gi", "Ti", "Pi", "Ei", "Zi"]:
            if abs(num) < 1024.0:
                return f"{num:3.1f} {unit}{suffix}"
            num /= 1024.0
        return f"{num:.1f} Yi{suffix}"

    # Filter out system variables and variables starting with excluded prefixes
    filtered_vars = {
        name: value
        for name, value in local_vars.items()
        if not any(name.startswith(prefix) for prefix in exclude_prefixes)
    }

    # Sort and display the top variables
    sorted_vars = sorted(
        ((name, sys.getsizeof(value)) for name, value in filtered_vars.items()),
        key=lambda x: -x[1],
    )[:top_n]

    for name, size in sorted_vars:
        print("{:>30}: {:>8}".format(name, sizeof_fmt(size)))


def remove_duplicate_columns(df):
    duplicate_columns = df.columns[df.columns.duplicated(keep=False)]

    dup_cols_df = pd.DataFrame(index=df.index, columns=list(set(duplicate_columns)))

    for col_name in set(duplicate_columns):
        cols = df.loc[:, df.columns == col_name]

        chosen_col = cols.iloc[
            :, (cols != 0).sum().argmax() : (cols != 0).sum().argmax() + 1
        ]

        dup_cols_df[col_name] = chosen_col

    df = df[df.columns.drop_duplicates(keep=False)]

    df = pd.concat([df, dup_cols_df], axis=1)

    return df


def get_significant_genes_new(fold_change, ks_fdr, ligands=ligands):
    up_genes = (fold_change >= 1) * (ks_fdr[fold_change.columns] < 0.05)
    down_genes = (fold_change <= -1) * (ks_fdr[fold_change.columns] < 0.05)

    sig_df = up_genes + down_genes

    sig_genes_dict_new = {}

    for lig in ligands:
        up_genes_lig = up_genes.loc[
            :, up_genes[up_genes.index.str.contains(lig)].any()
        ].columns.tolist()
        down_genes_lig = down_genes.loc[
            :, down_genes[down_genes.index.str.contains(lig)].any()
        ].columns.tolist()
        sig_genes_dict_new[lig] = {
            "up": up_genes_lig,
            "down": down_genes_lig,
        }

    return sig_genes_dict_new, sig_df


def generate_sig_genes_objects(ligand, df):
    fold_change = get_fc(df, log_addition=1e-3)
    ks_fdr = pd.read_parquet(
        DATA_DIR / "ks_fdr_corrected_pvals_new.parquet",
        index_col=0,
    )
    ks_fdr = change_bmp4_conc(ks_fdr)
    sig_genes_dict_new, sig_df = get_significant_genes_new(fold_change, ks_fdr)
    sig_genes = list(
        dict.fromkeys(
            sig_genes_dict_new[ligand]["up"] + sig_genes_dict_new[ligand]["down"]
        )
    )
    df_lig_sig_genes_new = df.loc[df.index.str.contains(f"{ligand}|CTRL_1"), sig_genes]
    fold_change = fold_change.loc[
        fold_change.index.str.contains(f"{ligand}|CTRL_1"), sig_genes
    ]

    return (
        sig_genes_dict_new,
        sig_df,
        fold_change,
        df_lig_sig_genes_new,
    )


def load_ligand_models(subset=[], test=None):
    path = DATA_DIR / f"ligand_model_{test}" if test else DATA_DIR / "ligand_model"
    ligand_models = {}

    files = glob.glob(os.path.join(path, "*"))

    # check if subset is not empty
    if subset:
        files = [file for file in files if any(s in file for s in subset)]

    for file in files:
        try:
            ligand_name = (
                os.path.basename(file).split("/")[-1].split("_")[2].split(".pkl")[0]
            )
            ligand_model = pd.read_pickle(file)
            ligand_models[ligand_name] = ligand_model
        except Exception as e:
            print(f"Error loading model from {file}: {e}")

    return ligand_models


def timeit(func):
    @wraps(func)
    def timeit_wrapper(*args, **kwargs):
        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        end_time = time.perf_counter()
        total_time = end_time - start_time
        print(f"Function {func.__name__} Took {total_time:.4f} seconds")
        return result

    return timeit_wrapper
