import os
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats

from scipy.stats import spearmanr, pearsonr
from sklearn.metrics import ConfusionMatrixDisplay, roc_curve, auc
from matplotlib.lines import Line2D

from figures.fig_scripts.fig5_data_processing import CONCENTRATIONS

BASE_DIR = Path(__file__).resolve().parent.parent.parent
SAVE_DIR = BASE_DIR / 'figures' / 'panels' / 'figure_5' 

os.makedirs(SAVE_DIR, exist_ok=True)

from src.utils.utils import set_style

set_style()

##############################################################################
# Panel A: Cumulative ROC plot
##############################################################################

def plot_cumulative_roc(pred_obj, save=False):
    probs_xgb = pred_obj.probabilities
    n_classes = probs_xgb.shape[1]
    y_test = pred_obj.y_test

    aucs = []
    roc_data = []

    for k in range(1, n_classes):  # thresholds 1..7 (for classes 0..7)
        # Binary labels: Y >= k
        y_binary = (y_test >= k).astype(int)

        # Cumulative probability P(Y >= k)
        p_cum = probs_xgb[:, k:].sum(axis=1)

        fpr, tpr, _ = roc_curve(y_binary, p_cum)
        auc_k = auc(fpr, tpr)

        aucs.append(auc_k)
        roc_data.append((fpr, tpr))

    plt.figure(figsize=(8, 6))

    for i, (fpr, tpr) in enumerate(roc_data):
        plt.plot(fpr, tpr, label=f"k={i + 1}, AUC={aucs[i]:.3f}", lw=2)

    plt.plot([0, 1], [0, 1], "k--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Cumulative ROC Curves by Concentration")
    plt.legend(loc="lower right")
    if save:
        plt.savefig(
            SAVE_DIR / "cumulative_roc.pdf",
            format="pdf",
            bbox_inches="tight",
        )
    plt.show()


##############################################################################
# Panel B: Aggregated Predicted Probabilities Heatmap
##############################################################################

def plot_agg_probabilities(pred_obj, save=False):
    y_pred_xgb_df = pred_obj.xgb_pred_df.copy()
    # unique_concs = y_pred_xgb_df["conc"].unique()
        # Group by the true class (conc) and average the predicted columns
    df_sorted = y_pred_xgb_df.sort_values("conc", ascending=False)
    grid_8x8 = df_sorted.iloc[:, :9].groupby("conc").mean()

    plt.figure(figsize=(6.5, 5))
    ax = sns.heatmap(grid_8x8[::-1], annot=False, fmt=".2f", cmap="viridis")
    # Put the origin at the lower-left so low concentrations are bottom-left and high are top-right

    plt.yticks(ticks=[i+0.5 for i in range(len(CONCENTRATIONS))], labels=[f"{c} ng/mL" for c in  CONCENTRATIONS[::-1]], fontsize=10, rotation=0, va="center", ha="right")

    # plt.title("Average Predicted Probability per True Concentration")
    plt.xlabel("Predicted Class Probability", fontsize=14, labelpad=10)
    plt.ylabel("Samples Grouped by Concentration", fontsize=14, labelpad=10)
    plt.tight_layout()

    if save:
        plt.savefig(
            SAVE_DIR / "agg_probabilities_heatmap.pdf",
            format="pdf",
            bbox_inches="tight",
        )

    plt.show()

##############################################################################
# Panel C: Concentration Prediction
##############################################################################

def plot_conc_prediction(
    pipeline, conc_list=CONCENTRATIONS, agg_func="median", reconstruction=False, save=False
):  
    if hasattr(pipeline, "xgb_pred_df") and pipeline.xgb_pred_df is not None:
        y_pred_df = pipeline.xgb_pred_df.copy()
        grouped_df = pipeline.rc_pred_conc_medians.copy() if reconstruction else pipeline.pred_conc_medians.copy()
    else:
        raise ValueError("No valid prediction dataframe found in the pipeline.")
    
    data_col = "rc_pseudo_conc" if reconstruction else "pseudo_conc"
        
    # Compute Spearman correlation (fixed syntax)
    spearman_result = spearmanr(y_pred_df["conc"], y_pred_df[data_col])[0]
    pearson_result = pearsonr(y_pred_df["conc"], y_pred_df[data_col])[0]

    # Create dot plot with mean and standard error
    plt.figure(figsize=(8, 6))

    # Sort the dataframe by concentration to ensure all values are plotted in order
    sorted_df = y_pred_df.sort_values(by="conc")

    # Create the stripplot with explicit hue_order
    sns.stripplot(
        data=sorted_df,
        x="conc",
        y=data_col,
        hue="conc",
        palette="flare",
        s=5,
        alpha=0.5,
        # zorder=0,
        jitter=0.15,
        hue_order=sorted(conc_list),  # Explicitly set the hue order
        dodge=False,  # Disable dodging which can sometimes affect the legend
    )

    # Plot mean points with error bars
    plt.errorbar(
        grouped_df["conc"],
        grouped_df[f"pseudo_conc_{agg_func}"],
        yerr=[
            grouped_df[f"pseudo_conc_se_lower_{agg_func}"],
            grouped_df[f"pseudo_conc_se_upper_{agg_func}"],
        ],
        fmt="o",
        color="black",
        capsize=7,
        capthick=2.5,
        markersize=9,
        zorder=10,
    )

    x_positions = np.arange(len(conc_list))
    actual_y = np.array(conc_list)  # actual concentration values

    # add green line that passes through the actual concentration points
    plt.plot(
        x_positions,
        actual_y,
        color="green",
        linestyle="--",
        linewidth=3,
        zorder=0,
        label="Actual Concentration",
    )

    # Labels and title
    plt.xlabel("Concentration (ng/mL)", fontsize=16)
    plt.ylabel("Predicted BMP Concentration", fontsize=16)
    plt.title(
        f"Predicted pseudo BMP concentration\n Spearman: {spearman_result:.2f}\nPearson: {pearson_result:.2f}",
        fontsize=16,
    )
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)

    # Get the current legend
    legend = plt.gca().get_legend()

    # Remove the current legend
    if legend is not None:
        legend.remove()

    # Create a custom legend with all concentration values
    handles = []
    labels = []

    # Add the mean point
    handles.append(
        Line2D([0], [0], marker="o", color="black", markersize=9, linestyle="None")
    )
    labels.append(f"{agg_func.capitalize()} \n± SE")

    # Add the actual concentration green dashed line to the legend
    handles.append(
        Line2D(
            [0],
            [0],
            color="green",
            linestyle="--",
            linewidth=3,
        )
    )
    labels.append("Real\nConc.")

    # Add the custom legend
    plt.legend(
        handles,
        labels,
        loc="upper left",
        fontsize=14,
        title=None,
        title_fontsize=12,
        bbox_to_anchor=(1, 1),
    )

    # make conc_list x labels:
    plt.xticks(
        ticks=np.arange(len(conc_list)),
        labels=conc_list,
        rotation=45,
        fontsize=16,
        ha="right",
    )
    plt.tick_params(axis="x", bottom=True, top=False, length=5, width=2)
    plt.yticks(fontsize=16)

    plt.yscale("log")
    # plt.ylim(1e-3, 1e3)
    plt.tight_layout()

    if save:
        plt.savefig(
            SAVE_DIR / "test_set_predicted_conc.pdf",
            format="pdf",
            bbox_inches="tight",
        )

    plt.show()

##############################################################################
# Panel D: Villi Spatial Concentration Prediction
##############################################################################

def plot_spatial_predicted_conc(
    pipeline,
    binned_col="spatial_coordinate",
    agg_func="median",
    plot_fit=False,
    hue=None,
    save=False,
    only_bins=False,
    ylog=True,
    ylim=(1e-2, 1e1),
):  
    if hasattr(pipeline, "y_pred_spatial") and pipeline.y_pred_spatial is not None:
        y_pred_df = pipeline.y_pred_spatial.copy()
    else:
        raise ValueError("No valid prediction dataframe found in the pipeline.")
    # y_pred_spatial_df = pipeline.y_pred_spatial.copy()
    y_pred_spatial_df_no_crypt_binned = pipeline.y_pred_spatial_no_crypt_binned.copy()
    y_pred_spatial_df_crypt = pipeline.y_pred_spatial_crypt.copy()
    grouped_bins = pipeline.grouped_bins.copy()
    grouped_crypt = pipeline.grouped_crypt.copy()

    conc_data_col = "rc_pseudo_conc" if "rc_pseudo_conc" in pipeline.y_pred_spatial.columns else "pseudo_conc"

    crypt_x_base = 0.0

    spear_corr = spearmanr(
        y_pred_spatial_df_no_crypt_binned[conc_data_col],
        y_pred_spatial_df_no_crypt_binned[binned_col],
    )[0]
    pearson_corr = pearsonr(
        y_pred_spatial_df_no_crypt_binned[conc_data_col],
        y_pred_spatial_df_no_crypt_binned[binned_col],
    )[0]

    if not only_bins:
        sns.scatterplot(
            data=y_pred_spatial_df_no_crypt_binned,
            y=conc_data_col,
            x=binned_col,
            hue=hue,
            palette="flare",
            s=40,
            alpha=0.5,
            edgecolor="none",
            linewidth=0,
        )

        sns.scatterplot(
            data=y_pred_spatial_df_crypt,
            y=conc_data_col,
            x="crypt_x",
            color="lightblue",
            s=40,
            alpha=0.5,
            edgecolor="none",
            linewidth=0,
        )

    first_x = crypt_x_base
    first_y = grouped_crypt[f"pseudo_conc_{agg_func}"].iloc[0]
    first_yerr = [
        grouped_crypt[f"pseudo_conc_se_lower_{agg_func}"].iloc[0],
        grouped_crypt[f"pseudo_conc_se_upper_{agg_func}"].iloc[0],
    ]


    if plot_fit and hasattr(pipeline, "decay_results") and pipeline.decay_results is not None:
        results = pipeline.decay_results
        slope, intercept = results["params"]
        rse = results["rse_log"]
        
        # 1. Prepare the fit line
        x_fit = np.linspace(0, 1, 100)
        y_log_fit = intercept + slope * x_fit
        y_fit = np.exp(y_log_fit)
        
        # 2. Calculate the Prediction Interval/Confidence Interval
        # We need the x-values used in the original fit to find the 'spread'
        x_orig = pipeline.grouped_bins["spatial_median"].values
        n = len(x_orig)
        x_mean = np.mean(x_orig)
        ssx = np.sum((x_orig - x_mean)**2)
        
        # 95% t-value
        t_val = stats.t.ppf(0.975, n - 2)
        
        # Standard Error of the fit at each point in x_fit
        se_fit = rse * np.sqrt(1/n + (x_fit - x_mean)**2 / ssx)
        
        # Calculate Upper and Lower bounds in log space
        lower_log = y_log_fit - (t_val * se_fit)
        upper_log = y_log_fit + (t_val * se_fit)
        
        # 3. Plotting
        # Plot the main dashed fit line
        plt.plot(x_fit, y_fit, color="red", linestyle="--", label="Exponential Fit")
        
        # Add the shaded confidence envelope
        plt.fill_between(x_fit, np.exp(lower_log), np.exp(upper_log), 
                        color="red", alpha=0.15, label="95% Confidence Interval")

    crypt_line = None

    plt.errorbar(
        first_x,
        first_y,
        xerr=0.0,
        yerr=[[first_yerr[0]], [first_yerr[1]]],
        fmt="o",
        color="green",
        capsize=3,
        markersize=6,
    )

    crypt_line = plt.axhline(
        y=first_y,
        color="green",
        linestyle="--",
        label=f"Crypt\n{agg_func.capitalize()}\n± SE",
    )

    bins_err = plt.errorbar(
        grouped_bins[f"spatial_{agg_func}"].iloc[:],
        grouped_bins[f"pseudo_conc_{agg_func}"].iloc[:],
        xerr=grouped_bins[f"spatial_se_{agg_func}"].iloc[:],
        yerr=[
            grouped_bins[f"pseudo_conc_se_lower_{agg_func}"].iloc[:],
            grouped_bins[f"pseudo_conc_se_upper_{agg_func}"].iloc[:],
        ],
        fmt="o",
        color="black",
        capsize=3,
        label=f"Bins\n{agg_func.capitalize()}\n± SE",
    )

    plt.xlabel("Normalized Position", fontsize=14)
    plt.ylabel("Predicted BMP Concentration", fontsize=14)
    plt.title(
        "Normalized Position vs. Predicted BMP Concentration by Bin\n"
        f"Spearman: {spear_corr:.2f}\nPearson: {pearson_corr:.2f}"
    )
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)

    if ylog:
        plt.yscale("log")
    # plt.xlim(-0.08, 1.0)
    plt.ylim(ylim[0], ylim[1])

    n_bins = len(y_pred_spatial_df_no_crypt_binned["bins"].unique())
    if n_bins <= 12:
        plt.legend(
            ncol=1,
            title="Bins",
            loc="upper right",
            bbox_to_anchor=(1.23, 1),
            fontsize=10,
        )
    else:
        handles = [h for h in (crypt_line, bins_err) if h is not None]
        labels = [h.get_label() for h in handles]
        plt.legend(
            handles,
            labels,
            ncol=1,
            loc="upper right",
            bbox_to_anchor=(1.23, 1),
            fontsize=10,
        )

    if save:
        plt.savefig(
            SAVE_DIR / "spatial_predicted_conc.pdf",
            format="pdf",
            bbox_inches="tight",
        )

    plt.show()

###############################################################
### Supplementary Panel A: Train vs Test Distribution
###############################################################


def plot_train_test_distribution(pred_obj, save=False):
    y_train = pred_obj.y_train
    y_test = pred_obj.y_test

    concentrations = sorted(np.unique(y_train))
    train_counts = [np.sum(y_train == c) for c in concentrations]
    test_counts = [np.sum(y_test == c) for c in concentrations]
    bar_width = 0.5
    plt.figure(figsize=(7, 4))
    plt.bar(concentrations, train_counts, width=bar_width, label="Train Set", color="skyblue")
    plt.bar(concentrations, test_counts, width=bar_width, label="Test Set", color="orangered", bottom=train_counts)
    plt.xlabel("Ordinal Concentration", fontsize=14)
    plt.ylabel("Number of Samples", fontsize=14)
    # plt.title("Distribution of Train and Test Samples Across Concentrations", fontsize=16)
    plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.tight_layout()

    if save:
        plt.savefig(
                SAVE_DIR / "train_test_distribution.pdf",
                format="pdf",
                bbox_inches="tight",
            )

    plt.show()


######################################################################
### Supplementary Panel B: Single Cell Predicted Probabilities Heatmap
######################################################################


def plot_sc_probabilities(pred_obj, save=False):
    y_pred_xgb_df = pred_obj.xgb_pred_df.copy()
    unique_concs = y_pred_xgb_df["conc"].unique()

    # Sort to keep concentration groups together
    df_sorted = y_pred_xgb_df.sort_values("conc", ascending=False)
    sns.heatmap(df_sorted.iloc[:, :8], cmap="GnBu", yticklabels=False)

    current_y = 0
    for c1, c2 in zip(unique_concs, CONCENTRATIONS[::-1]):
        count = len(df_sorted[df_sorted["conc"] == c1])
        # Draw separator line
        if c2 != 0.0064:  # Skip line for the lowest concentration
            plt.axhline(current_y + count, color="white", lw=4) 
        # Label the concentration on the left
        plt.text(-0.1, current_y + count / 2, f"{c2} ng/mL", ha="right", va="center")
        current_y += count

    plt.xlabel("Predicted Class Probability", fontsize=14, labelpad=10)
    plt.ylabel("Samples Grouped by Concentration", fontsize=14, labelpad=80)

    if save:
        plt.savefig(
            SAVE_DIR / "sc_probabilities_heatmap.pdf",
            format="pdf",
            bbox_inches="tight",
        )

    plt.show()

######################################################################
### Supplementary Panel C: Test set confusion matrix
######################################################################

def plot_confusion_matrix(pred_obj, save=False):
    y_true = pred_obj.y_test
    y_pred = pred_obj.hard_pred
    conc_cols = [f"Conc. {c}" for c in range(8)]
    
    ConfusionMatrixDisplay.from_predictions(
        y_true,
        y_pred,
        display_labels=conc_cols,
        # cmap=plt.cm.Blues,
        cmap="GnBu",
        normalize="true",
        values_format=".2f",
        xticks_rotation=45,
        # remove grid lines
        ax=plt.gca(),
        include_values=True
    )
    plt.gca().grid(False)
    plt.tight_layout()
    #invert y axis:
    plt.gca().invert_yaxis()
    if save:
        plt.savefig(
            SAVE_DIR / "confusion_matrix.pdf",
            format="pdf",
            bbox_inches="tight",
        )
    plt.show()

######################################################################
### Supplementary Panel D: Variance of Scores by Concentration
######################################################################


def plot_variances(
    vars_df, models=list, colors=None, log=False, size=(9, 4), box=(1.35, 1), save=False
) -> None:
    vars_df = vars_df.reset_index()
    # ensure index column is named 'ordinal_conc'
    if vars_df.columns[0] != "ordinal_conc":
        vars_df = vars_df.rename(columns={vars_df.columns[0]: "ordinal_conc"})

    x_vals = vars_df.pop("ordinal_conc").values
    vars_df = vars_df.iloc[:, : len(models)]

    if colors is None:
        colors = sns.color_palette("tab20", n_colors=len(models))
    else:
        colors = colors

    plt.figure(figsize=size)
    for i, model in enumerate(vars_df.columns[:]):
        var_df_subset = vars_df.xs(model, axis=1)
        plt.plot(
            x_vals,
            var_df_subset,
            marker="o",
            markersize=8,
            mec="gray",
            mew=1,
            label=f"{model} PC1",
            color=colors[i],
            # increase line width:
            lw=2,
        )

    plt.xlabel("Ordinal concentration", fontsize=15)
    plt.ylabel("Variance (Z-score)", fontsize=15)
    plt.title("Variance by method across concentrations", fontsize=15)
    plt.legend(
        title="Model",
        loc="best",
        labels=models,
        bbox_to_anchor=box,
    )
    plt.xticks(x_vals, fontsize=12)
    plt.yticks(fontsize=12)
    if log:
        plt.yscale("log")
    plt.tight_layout()

    if save:
        plt.savefig(
            SAVE_DIR / "pc1_model_variance.pdf",
            format="pdf",
            bbox_inches="tight",
        )

    plt.show()