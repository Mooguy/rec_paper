# import numpy as np
# import scipy.optimize as opt
# import scipy.stats as sps


# def residuals_fixed_sigma_regularized(
#     params, data, fixed_sigma, empirical_mean, reg_strength
# ):
#     loc = params[0]
#     normal_dist = sps.norm(loc=loc, scale=fixed_sigma)

#     # Standard fitting error
#     edges = np.array([0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5])
#     cdf_vals = normal_dist.cdf(edges)
#     binned_model = np.diff(np.concatenate([[0], cdf_vals, [1]]))

#     fit_error = data - binned_model

#     # --- THE FIX: ADD A PENALTY ---
#     # If the fitted 'loc' drifts far from 'empirical_mean', this error grows.
#     # We multiply by sqrt(strength) because least_squares will square it.
#     penalty = np.sqrt(reg_strength) * (loc - empirical_mean)

#     # Append penalty to the error array
#     return np.append(fit_error, penalty)


# def predict_regularized(data, fixed_sigma, reg_strength):
#     # Calculate Empirical Mean (Center of Mass)
#     # This serves as the "Anchor" for our regularization
#     indices = np.arange(8)
#     empirical_mean = np.sum(data * indices)

#     res = opt.least_squares(
#         residuals_fixed_sigma_regularized,
#         x0=[empirical_mean],  # Start searching at the data center
#         args=(data, fixed_sigma, empirical_mean, reg_strength),
#     )

#     fitted_mean = res.x[0]
#     return np.exp(fitted_mean * np.log(5) + np.log(0.0064))


# def find_global_sigma(df):
#     valid_sigmas = []
#     for i in range(df.shape[0]):
#         row_data = df.iloc[i, 0:8].values.astype(float)
#         # Important: Normalize even for calibration
#         if np.sum(row_data) > 0:
#             row_data = row_data / np.sum(row_data)

#         peak_bin = np.argmax(row_data)

#         if 2 <= peak_bin <= 5:
#             res = opt.least_squares(
#                 lambda p, d: (
#                     d
#                     - np.diff(
#                         np.concatenate(
#                             [
#                                 [0],
#                                 sps.norm(p[0], p[1]).cdf(
#                                     [0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5]
#                                 ),
#                                 [1],
#                             ]
#                         )
#                     )
#                 ),
#                 x0=[3.5, 1.0],
#                 args=(row_data,),
#                 bounds=([-np.inf, 0.1], [np.inf, 5]),
#             )
#             valid_sigmas.append(res.x[1])

#     global_sigma = np.mean(valid_sigmas) if valid_sigmas else 1.0
#     print(f"Using Global Sigma: {global_sigma:.4f}")
#     return global_sigma


# def reconstruct_conc(df, reg_strength=0.05):
#     df=df.copy()
#     global_sigma = find_global_sigma(df)
#     df["rc_pseudo_conc"] = 0.0
#     for i in range(df.shape[0]):
#         row_data = df.iloc[i, 0:8].values.astype(float)

#         # --- CRITICAL: UNCOMMENTED NORMALIZATION ---
#         # The solver will malfunction if data doesn't sum to 1.0
#         total = np.sum(row_data)
#         if total > 0:
#             row_data = row_data / total
#         else:
#             # Handle empty rows if necessary
#             continue

#         df.iloc[i, -1] = predict_regularized(
#             row_data, global_sigma, reg_strength=reg_strength
#         )

#     df["rc_exp_conc"] = np.log(df["rc_pseudo_conc"])
#     results_df = df[["rc_pseudo_conc", "rc_exp_conc"]]
#     return results_df

import numpy as np
import scipy.optimize as opt
import scipy.stats as sps

# Bin edges between integer rating bins 1–7
BIN_EDGES = np.array([0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5])


def compute_binned_probabilities(loc: float, sigma: float) -> np.ndarray:
    """
    Compute the probability mass in each of 8 bins (0–7) for a normal
    distribution with the given mean (loc) and standard deviation (sigma).

    Bins 0 and 7 absorb the left and right tails respectively.
    """
    cdf_at_edges = sps.norm(loc=loc, scale=sigma).cdf(BIN_EDGES)
    # Prepend 0 and append 1 so np.diff gives probabilities for all 8 bins
    return np.diff(np.concatenate([[0], cdf_at_edges, [1]]))


def fitting_residuals(
    params: list[float],
    observed: np.ndarray,
    sigma: float,
    anchor_mean: float,
    reg_strength: float,
) -> np.ndarray:
    """
    Residuals for the least-squares fitter, combining:
      1. Fit error: how well the normal model matches the observed bin counts.
      2. Regularization penalty: discourages the fitted mean from drifting
         far from the empirical (data-weighted) mean.

    The penalty is scaled by sqrt(reg_strength) because least_squares
    squares all residuals internally — squaring restores the intended weight.

    Args:
        params:        [loc] — the fitted mean being optimized.
        observed:      Normalized bin counts (must sum to 1).
        sigma:         Fixed standard deviation (determined globally).
        anchor_mean:   Empirical mean used as the regularization target.
        reg_strength:  How strongly to penalize deviations from anchor_mean.

    Returns:
        A residual array of length 9 (8 fit errors + 1 penalty term).
    """
    loc = params[0]
    model_probs = compute_binned_probabilities(loc, sigma)
    fit_errors = observed - model_probs
    regularization_penalty = np.sqrt(reg_strength) * (loc - anchor_mean)

    return np.append(fit_errors, regularization_penalty)


def fit_mean_from_bins(
    observed: np.ndarray,
    sigma: float,
    reg_strength: float,
) -> float:
    """
    Fit the mean of a normal distribution to observed bin counts,
    with regularization anchoring the result near the empirical mean.

    Args:
        observed:     Normalized bin counts (must sum to 1).
        sigma:        Fixed standard deviation to use during fitting.
        reg_strength: Regularization strength (higher = stays closer to anchor).

    Returns:
        The fitted mean (continuous value on the bin scale 0–7).
    """
    # Weighted average bin index — serves as both the starting guess
    # and the regularization anchor
    bin_indices = np.arange(8)
    empirical_mean = np.sum(observed * bin_indices)

    result = opt.least_squares(
        fitting_residuals,
        x0=[empirical_mean],
        args=(observed, sigma, empirical_mean, reg_strength),
    )

    return result.x[0]


def fitted_mean_to_concentration(fitted_mean: float) -> float:
    """
    Convert a fitted bin mean to a pseudo-concentration value.

    The formula assumes a log-linear relationship between bin index and
    concentration, calibrated so that bin 5 maps to a reference concentration.
    Adjust the constants (log base 5, scale 0.0064) to match your calibration.

    Args:
        fitted_mean: Fitted mean on the 0–7 bin scale.

    Returns:
        Pseudo-concentration as a positive float.
    """
    return np.exp(fitted_mean * np.log(5) + np.log(0.0064))


def estimate_global_sigma(df) -> float:
    """
    Estimate a single shared sigma (spread) across all rows by fitting
    a free normal distribution to rows whose peak bin is well-centered (2–5).

    Rows peaking at the extremes (0, 1, 6, 7) are excluded because edge
    effects make sigma estimation unreliable there.

    Args:
        df: DataFrame whose first 8 columns are bin counts per sample.

    Returns:
        The mean sigma across all well-centered rows, or 1.0 as a fallback.
    """
    fitted_sigmas = []

    for i in range(df.shape[0]):
        bin_counts = df.iloc[i, 0:8].values.astype(float)
        total = bin_counts.sum()
        if total <= 0:
            continue
        bin_counts /= total  # Normalize

        peak_bin = np.argmax(bin_counts)
        if not (2 <= peak_bin <= 5):
            continue  # Skip edge-dominated rows

        def residuals_free_fit(params, observed):
            loc, sigma = params
            return observed - compute_binned_probabilities(loc, sigma)

        result = opt.least_squares(
            residuals_free_fit,
            x0=[3.5, 1.0],  # Start near the center with a moderate spread
            args=(bin_counts,),
            bounds=([-np.inf, 0.1], [np.inf, 5]),
        )
        fitted_sigmas.append(result.x[1])

    global_sigma = np.mean(fitted_sigmas) if fitted_sigmas else 1.0
    print(f"Estimated global sigma: {global_sigma:.4f}")
    return global_sigma


def reconstruct_conc(df, reg_strength: float = 0.05):
    """
    Reconstruct pseudo-concentrations for each row in the DataFrame by:
      1. Estimating a shared sigma across all rows.
      2. Fitting a regularized normal distribution mean to each row's bin counts.
      3. Converting fitted means to pseudo-concentrations.

    Args:
        df:           DataFrame whose first 8 columns are per-bin counts.
        reg_strength: Regularization strength for mean fitting (default 0.05).

    Returns:
        A DataFrame with two new columns:
          - 'rc_pseudo_conc': concentration on the original scale.
          - 'rc_log_conc':    natural log of pseudo-concentration.
    """
    df = df.copy()
    global_sigma = estimate_global_sigma(df)
    pseudo_concentrations = np.zeros(df.shape[0])

    for i in range(df.shape[0]):
        bin_counts = df.iloc[i, 0:8].values.astype(float)
        total = bin_counts.sum()
        if total <= 0:
            continue  # Leave as 0 for empty rows
        bin_counts /= total  # Normalize so counts sum to 1

        fitted_mean = fit_mean_from_bins(bin_counts, global_sigma, reg_strength)
        pseudo_concentrations[i] = fitted_mean_to_concentration(fitted_mean)

    df["rc_pseudo_conc"] = pseudo_concentrations
    df["rc_log_conc"] = np.log(df["rc_pseudo_conc"])

    return df[["rc_pseudo_conc", "rc_log_conc"]]