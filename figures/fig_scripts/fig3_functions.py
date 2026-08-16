import os
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np
import gseapy as gp
import warnings
import json
import glob

from matplotlib.colors import LinearSegmentedColormap
from matplotlib import colors


BASE_DIR = Path(__file__).resolve().parent.parent.parent 
SAVE_DIR = BASE_DIR / "figures" / "panels" / "figure_3"

os.makedirs(SAVE_DIR, exist_ok=True)

from src.utils.utils import set_style
set_style()

###########################################################
# Running go analysis:
###########################################################

def run_go_analysis(genes_dict, background_genes, seperate_up_down=False):
    go_results_groups_dict = {}
    gene_set = "GO_Biological_Process_2021"
    separate = seperate_up_down

    def _run_enrichr(key, gene_list):
        # validate gene list
        if not gene_list:
            warnings.warn(f"No genes provided for {key}, skipping.")
            return pd.DataFrame()
        try:
            res = gp.enrichr(
                gene_list=list(set(gene_list)),
                gene_sets=[gene_set],
                organism="mouse",
                background=background_genes,
                outdir=None,
            )
        except Exception as e:
            warnings.warn(f"GO enrichment failed for {key}: {e}")
            return pd.DataFrame()

        # filter to the requested gene set (gseapy uses 'Gene_set' column)
        df = res.results
        if "Gene_set" in df.columns:
            df = df[df["Gene_set"] == gene_set].copy()
        else:
            # fallback: return full results if structure differs
            df = df.copy()

        if df.empty:
            warnings.warn(f"No GO results for {key}")
        return df

    for group, genes in genes_dict.items():
        # handle dict with 'up'/'down'
        if isinstance(genes, dict) and set(genes.keys()) >= {"up", "down"}:
            if separate:
                for direction in ("up", "down"):
                    key = f"{group}_{direction}"
                    print(f"Processing {key}...")
                    df = _run_enrichr(key, genes.get(direction, []))
                    go_results_groups_dict[key] = df
                # skip combined analysis when separate is True
                continue
            else:
                # combine up + down (unique)
                all_genes = list(set(genes.get("up", []) + genes.get("down", [])))
        else:
            # assume genes is an iterable of gene ids
            all_genes = list(genes) if genes is not None else []

        key = f"{group}"
        print(f"Processing {key}...")
        df = _run_enrichr(key, all_genes)
        go_results_groups_dict[key] = df

    return go_results_groups_dict

##############################################################################
# Functions for saving and loading GO and similarity data
##############################################################################


def save_go_results_dataframe(go_results_dict, path):
    os.makedirs(path, exist_ok=True)
    for group, df in go_results_dict.items():
        if df.empty:
            warnings.warn(f"No GO results for {group}, skipping.")
            continue
        elif os.path.exists(f"{path}/{group}_go_results.csv"):
            print(f"GO results for {group} already exist, skipping.")
            continue
        df = df.copy()
        df.to_csv(f"{path}/{group}_go_results.csv", index=True)


def load_go_dataframes_dict(path):
    go_results_dict = {}
    # read all files in path:
    files = glob.glob(os.path.join(path, "*"))
    for file in files:
        name = file.split("/")[-1].split("_go")[0]
        csv_file = pd.read_csv(file, index_col=0)
        go_results_dict[name] = csv_file

    return go_results_dict


def generate_json_for_R(go_results_dict):
    terms_dict_for_R = {}

    for lig, results_df in go_results_dict.items():
        results_df_sig = go_results_dict[lig][
            go_results_dict[lig]["Adjusted P-value"] < 0.05
        ]
        terms = results_df_sig["Term"].tolist()
        ids = [term.split("(")[1].strip(")") for term in terms]
        terms_dict_for_R[lig] = ids

    return terms_dict_for_R


def save_json_for_R(go_results_json, filename):
    # Create the full directory path
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    with open(filename, "w") as f:
        json.dump(go_results_json, f)


def save_individual_sim_matrices(go_sim_mat_clustered, path):
    os.makedirs(path, exist_ok=True)
    for group, sim_mat in go_sim_mat_clustered.items():
        sim_mat.to_csv(f"{path}/{group}_go_sim_matrix.csv", index=True)
        # print(f"Saved {group}")


##############################################################################
# Functions to generate the similarity matrix from R analysis:
##############################################################################


def extract_significant_go_terms(go_results_dict, intersect=False):
    sig_terms_dict = {}

    for ligand, go_df in go_results_dict.items():
        go_df_sig = go_df[go_df["Adjusted P-value"] < 0.05]["Term"]
        sig_terms_dict[ligand] = set(go_df_sig)
    if intersect:
        intersection_dict = {"terms": set.intersection(*sig_terms_dict.values())}
        return sig_terms_dict, intersection_dict

    return sig_terms_dict, None


def extract_dictionaries(sig_terms_dict):
    go_id_name_dict = {}
    for group in sig_terms_dict.keys():
        subset_terms = sig_terms_dict[group]
        # check if subset_terms is not empty
        if not subset_terms:
            continue
        subset_ids, subset_names = zip(
            *[
                (term.split("(")[1].strip(")"), term.split("(")[0].strip())
                for term in subset_terms
            ]
        )
        for subset_id, subset_name in zip(subset_ids, subset_names):
            if subset_id not in go_id_name_dict:
                go_id_name_dict[subset_id] = subset_name

    go_id_name_dict_reversed = {v: k for k, v in go_id_name_dict.items()}
    return go_id_name_dict, go_id_name_dict_reversed


def create_go_sim_matrix(data, sig_terms_dict, go_id_name_dict=None):
    go_sim_mat_dict = {}

    for ligand in data.keys():
        go_data = data[ligand]["sim_matrix"]["data"]
        go_rows = data[ligand]["sim_matrix"]["rownames"]
        go_cols = data[ligand]["sim_matrix"]["colnames"]
        go_sim_df = pd.DataFrame(go_data, index=go_rows, columns=go_cols)
        if go_id_name_dict:
            go_sim_df.index = go_sim_df.index.map(go_id_name_dict)
            go_sim_df.columns = go_sim_df.columns.map(go_id_name_dict)
        go_sim_mat_dict[ligand] = go_sim_df
        print(
            f"Length of {ligand} before sim analysis: {len(sig_terms_dict[ligand])} and after: {len(go_sim_df)}"
        )

    return go_sim_mat_dict

##############################################################################
# Plotting accessory functions:
##############################################################################

def insert_newline(term):
    words = term.split()
    if len(words) > 8:
        mid = len(words) // 2
        return " ".join(words[:mid]) + "\n" + " ".join(words[mid:])
    return term

def add_group_to_term(row):
    term = row["representative_term"]
    group_num = row["group"]

    # Split by newline if present
    if "\n" in term:
        parts = term.split("\n")
        # Add group number to the last word of the last part
        last_part_words = parts[-1].split()
        if last_part_words:
            last_part_words[-1] = f"{last_part_words[-1]}_{group_num}"
            parts[-1] = " ".join(last_part_words)
        return "\n".join(parts)
    else:
        # No newline, add to the last word
        words = term.split()
        if words:
            words[-1] = f"{words[-1]}_{group_num}"
        return " ".join(words)

def prepare_go_terms_for_plotting(goa_dict):
    goa_dict = goa_dict.copy()
    for cluster, obj in goa_dict.items():
        df = obj.go_sim_group_summary
        df["representative_term"] = df["representative_term"].apply(insert_newline)
        df["representative_term"] = df.apply(add_group_to_term, axis=1)

    return goa_dict

##############################################################################
# Plotting functions:
##############################################################################


def plot_go_term_groups_updated(
    goa_dict,
    clusters,
    plotsize=(6, 4),          # size of ONE dot-plot square, in inches (width, height)
    legend_location=(1.4, 1),
    ymargin=0.075,
    save=False,
    top_n=None,
):
    if not isinstance(clusters, (list, tuple)):
        clusters = [clusters]
    assert 1 <= len(clusters) <= 2, "Only 1 or 2 clusters supported"

    n = len(clusters)
    ax_w_in, ax_h_in = plotsize   # fixed physical size of the dots box

    # Fixed margins in INCHES -- constant regardless of n
    left_in, right_margin_in = 1.6, 1.8   # space for y-labels / cbar+legend
    top_in, bottom_in = 0.6, 0.6
    hspace_in = 0.5

    fig_w = left_in + ax_w_in + right_margin_in
    fig_h = top_in + n * ax_h_in + (n - 1) * hspace_in + bottom_in

    colors = ["blue", "purple", "red"]
    nodes = [0.0, 0.5, 1.0]
    cmap_blue_to_red = LinearSegmentedColormap.from_list(
        "blue_to_red", list(zip(nodes, colors))
    ).reversed()

    fig = plt.figure(figsize=(fig_w, fig_h))

    # Convert fixed inch geometry -> fractions of THIS figure
    left = left_in / fig_w
    ax_width_frac = ax_w_in / fig_w
    ax_height_frac = ax_h_in / fig_h
    hspace_frac = hspace_in / fig_h
    top = 1 - top_in / fig_h
    cbar_left = left + ax_width_frac + 0.05 * ax_width_frac
    cbar_width = 0.04

    all_data = []
    for cluster in clusters:
        cluster_data = goa_dict.get(cluster).go_sim_group_summary.copy()
        if top_n is not None:
            cluster_data = cluster_data.sort_values(
                by=["gene_ratio", "combined_score", "adjusted_p_value",
                    "term_count", "term_depth"],
                ascending=[False, False, True, False, False],
            ).head(top_n)
        cluster_data = cluster_data.iloc[::-1]
        all_data.append(cluster_data)

    all_ratios = np.concatenate([d["gene_ratio"].values for d in all_data])
    x_pad = (all_ratios.max() - all_ratios.min()) * 0.15 or 0.05
    xlim = (all_ratios.min() - x_pad, all_ratios.max() + x_pad)

    all_pvals = np.concatenate([d["adjusted_p_value"].values for d in all_data])
    vmin, vmax = all_pvals.min(), all_pvals.max()

    main_axes = []
    for i, (cluster, cluster_data) in enumerate(zip(clusters, all_data)):
        ax_top = top - i * (ax_height_frac + hspace_frac)
        ax_bottom = ax_top - ax_height_frac

        # Fixed physical size box -- identical in inches for n=1 or n=2
        ax = fig.add_axes([left, ax_bottom, ax_width_frac, ax_height_frac])
        main_axes.append(ax)

        scatter = ax.scatter(
            cluster_data["gene_ratio"],
            cluster_data["representative_term"],
            s=cluster_data["term_count"] * 50,
            c=cluster_data["adjusted_p_value"],
            cmap=cmap_blue_to_red,
            vmin=vmin,
            vmax=vmax,
            alpha=0.8,
            edgecolors="black",
            linewidth=0.6,
        )

        ax.set_axisbelow(True)
        ax.grid(True, linestyle="--", alpha=0.4)
        ax.margins(y=ymargin)
        ax.set_xlim(xlim)
        ax.set_ylim(ax.get_ylim()[0] + 0.1, ax.get_ylim()[1] - 0.1)
        ax.set_ylabel("Representative Term", fontsize=12)

        if i == 0:
            ax.set_title(f"GO Term Groups - Cluster {cluster}", fontsize=14, fontweight="bold")

        cax = fig.add_axes([cbar_left, ax_bottom, cbar_width, ax_height_frac])
        cbar = fig.colorbar(scatter, cax=cax)
        cbar.set_label("Adjusted P-Value", fontsize=10)

        actual_counts = sorted(cluster_data["term_count"].unique())
        if len(actual_counts) > 3:
            legend_sizes = [actual_counts[0], actual_counts[len(actual_counts) // 2], actual_counts[-1]]
        else:
            legend_sizes = actual_counts
        size_labels = [f"{s} terms" for s in legend_sizes]

        legend_elements = [
            plt.scatter([], [], s=80, c="gray", alpha=0.7, edgecolors="black", linewidth=0.5)
            for _ in legend_sizes
        ]
        ax.legend(
            legend_elements,
            size_labels,
            title="Term Count",
            loc="center left",
            bbox_to_anchor=legend_location,
            frameon=True,
            handleheight=1.5,
            labelspacing=0.8,
            borderpad=1.0,
        )

    main_axes[-1].set_xlabel("Gene Ratio", fontsize=12)
    if n == 2:
        main_axes[0].tick_params(labelbottom=False)

    if save:
        name = "_".join(str(c) for c in clusters)
        fig.savefig(SAVE_DIR / f"{name}_grouped_go_terms_top_10.pdf",
                    format="pdf", bbox_inches="tight")

    plt.show()

def plot_sim_mat(go_sim, size=(16, 14), font_size=12, save=False):
    columns_to_drop = []
    has_p = "p_value" in go_sim.columns
    has_row_color = "row_color" in go_sim.columns

    # p_value handling
    if has_p:
        pvals = go_sim["p_value"].to_numpy().ravel()
        norm = colors.Normalize(vmin=pvals.min(), vmax=pvals.max())
        pvals_norm = norm(pvals)
        columns_to_drop.append("p_value")

    # row_color -> map to integer codes (0..k-1)
    row_codes = None
    if has_row_color:
        raw = go_sim["row_color"]
        # ensure integer codes for palette indexing
        if np.issubdtype(raw.dtype, np.integer):
            row_codes = raw.to_numpy().astype(int)
        else:
            row_codes = pd.Categorical(raw).codes  # 0..k-1
        columns_to_drop.append("row_color")

    main_matrix = go_sim.drop(columns=columns_to_drop)
    n_rows = main_matrix.shape[0]

    cmap_pvals = plt.cm.Blues
    cmap_row_colors = sns.color_palette("tab20", n_colors=100)

    fig, ax = plt.subplots(figsize=size)
    square_width = 1
    linewidth = 0.5

    sns.heatmap(
        main_matrix,
        ax=ax,
        cmap="viridis",
        cbar=True,
        cbar_kws={"shrink": 0.5, "label": "similarity"},
        vmin=main_matrix.values.min(),
        vmax=main_matrix.values.max(),
        yticklabels=True,
        xticklabels=False,
        linewidths=linewidth,
        linecolor="gray",
    )

    yticks = np.arange(n_rows) + 0.5

    # compute annotation offsets (left of heatmap)
    offset = 0.0
    if has_p and has_row_color:
        offset_pval = (-2 * square_width) - 0.4
        offset_rowcolor = -square_width - 0.2
        offset = min(offset_pval, offset_rowcolor)
    elif has_p:
        offset_pval = -square_width - 0.2
        offset = offset_pval
    elif has_row_color:
        offset_rowcolor = -square_width - 0.2
        offset = offset_rowcolor

    # draw p_value column if present
    if has_p:
        for i, val in enumerate(pvals_norm):
            ax.add_patch(
                plt.Rectangle(
                    (offset_pval, i),
                    square_width,
                    1,
                    color=cmap_pvals(val),
                    ec="gray",
                    lw=linewidth,
                )
            )

    # draw row_color column if present (use safe palette indexing)
    if has_row_color:
        for i, code in enumerate(row_codes):
            color = cmap_row_colors[int(code) % len(cmap_row_colors)]
            ax.add_patch(
                plt.Rectangle(
                    (offset_rowcolor, i),
                    square_width,
                    1,
                    color=color,
                    ec="gray",
                    lw=linewidth,
                )
            )

    # expand x-limits so annotations are visible
    ax.set_xlim(offset - 0.2, main_matrix.shape[1])

    ax.set_yticks(yticks)
    ax.set_yticklabels(main_matrix.index, fontsize=font_size, va="center")
    ax.set_xticks([])

    # plt.tight_layout()

    if save:
        plt.savefig(
            SAVE_DIR / "shared_go_terms_similarity_heatmap.pdf",
            format="pdf",
            bbox_inches="tight",
        )

    plt.show()
