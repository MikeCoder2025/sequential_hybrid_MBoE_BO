from __future__ import annotations

import time

"""
FULL STANDALONE GOOED FOR METHANE OXIDATION
============================================

This file is completely self-contained. It does NOT import benchmark_common.py.

Workflow
--------
1. Read ONLY Experiments 1-12 from Factorial_Data.csv.
2. Calibrate all six MVK kinetic parameters using weighted multi-response
   nonlinear least squares for yCH4, yO2 and yCO2.
3. The experimental response standard deviations are
       SIGMA_Y = [0.00043, 0.00202, 0.00051]
   and the response covariance is therefore
       R_y = diag(SIGMA_Y**2).
   Dividing residuals by SIGMA_Y is exactly equivalent to weighting by R_y^-1.
4. For Experiment 13 onward, GOOED selects the operating condition that
   maximizes mechanistic methane conversion using the CURRENT fitted parameters.
5. BEFORE the in-silico experiment, propagate parameter covariance to methane
   conversion. Two estimates are saved:
       - local delta-method uncertainty;
       - nonlinear Monte-Carlo parameter-propagation uncertainty.
   Monte Carlo is used for the publication-style uncertainty because the model
   is nonlinear and the parameter uncertainty can be large.
6. Run a virtual physical experiment with the MVK ground-truth parameters.
   Gaussian measurement noise is added to yCH4, yO2 and yCO2 using SIGMA_Y as
   STANDARD DEVIATIONS.
7. Append the noisy response and recalibrate all six kinetic parameters.
8. Recompute methane conversion and predictive uncertainty AFTER recalibration.
9. Parameter significance is assessed with the 95% two-sided Student-t critical
   value using the current residual degrees of freedom.
10. Repeat until Experiment 20 (fixed budget).

Important uncertainty point
---------------------------
The conversion derivative / Monte-Carlo propagation uses RAW (unclipped)
model conversion. Clipping conversion to [0,1] inside the uncertainty
calculation can artificially drive the gradient and hence the propagated
uncertainty to zero near complete conversion.

The reported/visualized central conversion is physically clipped to [0,1],
but uncertainty is computed before clipping. This prevents the false 0.0%
uncertainty problem seen in the earlier GOOED script.
"""

import argparse
import math
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from scipy.optimize import least_squares, differential_evolution
from scipy.stats import t as student_t


# =============================================================================
# USER SETTINGS
# =============================================================================

N_INITIAL = 12
MAX_EXPERIMENTS = 20
RANDOM_SEED = 42

# Parameter significance uses a dynamic 95% two-sided Student-t critical value
# based on the current residual degrees of freedom.

# Multistart parameter estimation
N_STARTS_INITIAL = 60
N_STARTS_SEQUENTIAL = 30

# GOOED continuous design-space optimisation
DE_POPSIZE = 12
DE_MAXITER = 40

# Monte-Carlo propagation of parameter uncertainty.
# Increase to 2000-5000 for final publication runs if runtime is acceptable.
MC_SAMPLES = 1000

# Numerical derivative step for delta-method diagnostic uncertainty.
UNCERTAINTY_RELATIVE_STEP = 1e-5

# Physical design bounds
TEMPERATURE_RANGE = (250.0, 350.0)       # degC
MASSFLOW_RANGE = (20.0, 30.0)            # Nml/min
MOLERATIO_RANGE = (2.0, 4.0)             # O2/CH4
METHANE_RANGE = (0.005, 0.025)            # inlet CH4 mole fraction

PINLET_DEFAULT = 1.75
POUTLET_DEFAULT = 1.30

# =============================================================================
# MVK REACTOR MODEL
# =============================================================================

R = 8.314
T_REF_C = 320.0
CATALYST_MASS_G = 0.01

PARAMETER_NAMES = [
    "theta_1", "theta_2", "theta_3",
    "theta_4", "theta_5", "theta_6",
]

# Experimental RESPONSE STANDARD DEVIATIONS.
SIGMA_Y = np.array([0.00043, 0.00202, 0.00051], dtype=float)

# Experimental response covariance matrix and inverse.
R_Y = np.diag(SIGMA_Y ** 2)
R_Y_INV = np.diag(1.0 / (SIGMA_Y ** 2))

# Used ONLY to emulate the physical/in-silico experiment.
THETA_TRUE = np.array([5.77, 6.72, 5.87, 9.51, 10.17, 7.98], dtype=float)

# Neutral estimation initial guess; intentionally not THETA_TRUE.
INITIAL_GUESS = np.array([6.0, 4.5, 6.0, 0.2, 10.0, 7.0], dtype=float)
LOWER_BOUNDS = np.zeros(6, dtype=float)
# UPPER_BOUNDS = np.full(6, 30.0, dtype=float)
UPPER_BOUNDS = np.array([30.0, 30.0, 30.0, 15, 30.0, 30.0], dtype=float)

MODEL_X_COLS = [
    "Temperature",
    "Massflow",
    "Inlet conc of Methane",
    "Moleratio",
    "Poutlet",
    "Pinlet",
]


def _mvk_ode(x, w, u, theta, pavg):
    """Packed-bed MVK model used in the switching implementation."""
    temperature_c = float(u[0])
    massflow = float(u[1])

    y_ch4 = max(float(x[0]), 1e-60)
    y_o2 = max(float(x[1]), 1e-60)

    t1, t2, t3, t4, t5, t6 = np.asarray(theta, dtype=float)

    T0 = 20.0
    P0 = 1.0

    k1 = math.exp(
        -t1
        - (t2 * 1e4 / R)
        * ((1.0 / (temperature_c + 273.15))
           - (1.0 / (T_REF_C + 273.15)))
    )

    k2 = math.exp(
        -t3
        - (t4 * 1e4 / R)
        * ((1.0 / (temperature_c + 273.15))
           - (1.0 / (T_REF_C + 273.15)))
    )

    k3 = math.exp(
        -t5
        - (t6 * 1e4 / R)
        * ((1.0 / (temperature_c + 273.15))
           - (1.0 / (T_REF_C + 273.15)))
    )

    numerator = k1 * k2 * (pavg ** 2) * y_ch4 * y_o2

    denominator = (
        k1 * pavg * y_o2
        + 2.0 * k2 * pavg * y_ch4
        + (k1 * k2 / max(k3, 1e-60))
        * (pavg ** 2) * y_ch4 * y_o2
    )

    r1 = numerator / max(denominator, 1e-60)

    cf = (
        R * (temperature_c + 273.15)
        / (
            pavg
            * 1e5
            * (
                massflow
                * (1e-6 / 60.0)
                * (P0 / pavg)
                * ((temperature_c + 273.15) / (T0 + 273.15))
            )
        )
    )

    return [
        -r1 * cf,
        -2.0 * r1 * cf,
        r1 * cf,
        2.0 * r1 * cf,
    ]


def condition_to_x(condition):
    """
    Design condition order:
        [Temperature, Massflow, Moleratio, inlet_CH4]

    Model X order:
        [Temperature, Massflow, inlet_CH4, Moleratio, Poutlet, Pinlet]
    """
    T, flow, ratio, methane = [float(v) for v in condition]
    return np.array(
        [T, flow, methane, ratio, POUTLET_DEFAULT, PINLET_DEFAULT],
        dtype=float,
    )


def predict_outputs(theta, X):
    """
    Predict [yCH4, yO2, yCO2] from the packed-bed MVK model.
    """
    X = np.asarray(X, dtype=float)
    if X.ndim == 1:
        X = X.reshape(1, -1)

    predictions = []

    for row in X:
        T, flow, methane, ratio, pout, pin = row

        inlet_o2 = methane * ratio
        pavg = 0.5 * (pin + pout)

        x0 = [methane, inlet_o2, 0.0, 1e-10]
        w_span = np.linspace(0.0, CATALYST_MASS_G, 5)

        sol = odeint(
            _mvk_ode,
            x0,
            w_span,
            args=(row, theta, pavg),
            mxstep=5000,
        )

        predictions.append(sol[-1, :3])

    return np.asarray(predictions, dtype=float)


# =============================================================================
# DATA LOADING
# =============================================================================


def load_initial_12(csv_path):
    df = pd.read_csv(csv_path, encoding="utf-8-sig")
    df.columns = [str(c).strip() for c in df.columns]

    required = [
        "Exp",
        "Temperature",
        "Massflow",
        "Moleratio",
        "Inlet conc of Methane",
        "Poutlet",
        "Pinlet",
        "yCH4",
        "yO2",
        "yCO2",
    ]

    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    df12 = (
        df.sort_values("Exp")
        .head(N_INITIAL)
        .copy()
        .reset_index(drop=True)
    )

    X = df12[MODEL_X_COLS].to_numpy(float)
    Y = df12[["yCH4", "yO2", "yCO2"]].to_numpy(float)

    return df12, X, Y


# =============================================================================
# PARAMETER ESTIMATION
# =============================================================================


def weighted_residuals(theta, X, Y):
    """
    Weighted residuals using experimental RESPONSE STANDARD DEVIATIONS.

    Dividing by sigma is statistically equivalent to using the inverse
    response covariance R_y^-1 in weighted least squares because

        r_w^T r_w = r^T diag(1/sigma^2) r.
    """
    Y_pred = predict_outputs(theta, X)
    return ((Y_pred - Y) / SIGMA_Y).reshape(-1)


def _multistart_guesses(center, n_starts, seed):
    rng = np.random.default_rng(seed)

    starts = [
        np.clip(
            np.asarray(center, dtype=float),
            LOWER_BOUNDS,
            UPPER_BOUNDS,
        )
    ]

    deterministic = [
        [6, 6, 6, 8, 10, 7],
        [5, 5, 5, 8, 9, 6],
        [6, 7, 5, 10, 10, 8],
        [6, 8, 4, 12, 10, 6],
        [5, 10, 3, 15, 10, 8],
    ]

    for g in deterministic:
        starts.append(np.asarray(g, dtype=float))

    practical_low = np.array([2, 1, 1, 1, 7, 1], dtype=float)
    practical_high = np.array([10, 18, 10, 25, 14, 18], dtype=float)

    while len(starts) < n_starts:
        g = practical_low + rng.random(6) * (practical_high - practical_low)
        starts.append(np.clip(g, LOWER_BOUNDS, UPPER_BOUNDS))

    return starts[:n_starts]


def fit_parameters(X, Y, initial_guess, n_starts, seed):
    best = None

    for guess in _multistart_guesses(initial_guess, n_starts, seed):
        result = least_squares(
            weighted_residuals,
            guess,
            bounds=(LOWER_BOUNDS, UPPER_BOUNDS),
            args=(X, Y),
            method="trf",
            jac="3-point",
            max_nfev=50000,
            x_scale="jac",
            ftol=1e-10,
            xtol=1e-10,
            gtol=1e-10,
        )

        if best is None or result.cost < best.cost:
            best = result

    theta = np.asarray(best.x, dtype=float)
    Jw = np.asarray(best.jac, dtype=float)

    F = Jw.T @ Jw

    weighted_rss = float(best.fun @ best.fun)
    dof = max(len(best.fun) - len(theta), 1)
    sigma2 = weighted_rss / dof

    # 95% two-sided Student-t critical value for the current residual degrees of freedom.
    t_critical = float(student_t.ppf(0.975, dof))

    covariance = sigma2 * np.linalg.pinv(F, rcond=1e-12)
    covariance = 0.5 * (covariance + covariance.T)

    se = np.sqrt(np.maximum(np.diag(covariance), 0.0))

    t_values = np.divide(
        np.abs(theta),
        se,
        out=np.full_like(theta, np.inf),
        where=se > 0,
    )

    relative_uncertainty_pct = 100.0 * np.divide(
        se,
        np.maximum(np.abs(theta), 1e-12),
    )

    return {
        "theta": theta,
        "jacobian": Jw,
        "fim": F,
        "covariance": covariance,
        "se": se,
        "t_values": t_values,
        "t_critical": t_critical,
        "dof": dof,
        "passed": t_values >= t_critical,
        "relative_uncertainty_pct": relative_uncertainty_pct,
        "weighted_rss": weighted_rss,
        "sigma2": sigma2,
        "fim_condition": float(np.linalg.cond(F)),
    }


def parameter_table(exp_no, fit):
    return pd.DataFrame(
        {
            "Experiment": exp_no,
            "Parameter": PARAMETER_NAMES,
            "Estimate": fit["theta"],
            "Standard_Error": fit["se"],
            "t_Value": fit["t_values"],
            "t_Reference": fit["t_critical"],
            "Pass": fit["passed"],
            "Relative_Parameter_Uncertainty_pct":
                fit["relative_uncertainty_pct"],
        }
    )


def print_parameter_table(exp_no, fit):
    tab = parameter_table(exp_no, fit)

    print("\n" + "=" * 105)
    print(f"PARAMETER ESTIMATION AFTER EXPERIMENT {exp_no}")
    print("=" * 105)
    print(
        tab[
            [
                "Parameter",
                "Estimate",
                "Standard_Error",
                "t_Value",
                "t_Reference",
                "Pass",
                "Relative_Parameter_Uncertainty_pct",
            ]
        ].to_string(index=False, float_format=lambda z: f"{z:.5f}")
    )

    print(f"\nResidual degrees of freedom: {fit['dof']}")
    print(f"95% two-sided Student-t critical value: {fit['t_critical']:.5f}")
    print(f"All parameters pass: {bool(np.all(fit['passed']))}")
    print(f"Weighted RSS: {fit['weighted_rss']:.6f}")
    print(f"Residual variance scale: {fit['sigma2']:.6f}")
    print(f"FIM condition number: {fit['fim_condition']:.6e}")

    return tab


# =============================================================================
# METHANE CONVERSION
# =============================================================================


def raw_model_conversion(theta, condition):
    """
    RAW, UNCLIPPED conversion.

    This function MUST be used for uncertainty propagation. If conversion is
    clipped to 1.0 before numerical differentiation, the derivative becomes
    artificially zero in the saturation region and the propagated parameter
    uncertainty collapses.
    """
    x = condition_to_x(condition)
    y = predict_outputs(theta, x)[0]

    inlet_ch4 = float(x[2])
    outlet_ch4 = float(y[0])

    return (inlet_ch4 - outlet_ch4) / max(inlet_ch4, 1e-12)


def physical_model_conversion(theta, condition):
    """Physical central value for reporting and GOOED optimisation."""
    return float(
        np.clip(
            raw_model_conversion(theta, condition),
            0.0,
            1.0,
        )
    )


# =============================================================================
# GOOED DESIGN OPTIMISATION
# =============================================================================


def design_bounds():
    # condition order: T, flow, ratio, methane
    return [
        TEMPERATURE_RANGE,
        MASSFLOW_RANGE,
        MOLERATIO_RANGE,
        METHANE_RANGE,
    ]


def optimize_gooed(theta, seed):
    """Select next condition by maximum CURRENT mechanistic conversion."""

    def objective(u):
        return -physical_model_conversion(theta, u)

    result = differential_evolution(
        objective,
        bounds=design_bounds(),
        seed=seed,
        popsize=DE_POPSIZE,
        maxiter=DE_MAXITER,
        tol=1e-7,
        polish=True,
        updating="immediate",
        workers=1,
    )

    return np.asarray(result.x, dtype=float), -float(result.fun)


# =============================================================================
# CONVERSION UNCERTAINTY
# =============================================================================


def conversion_gradient_unclipped(theta, condition):
    theta = np.asarray(theta, dtype=float)
    grad = np.zeros(len(theta), dtype=float)

    for j in range(len(theta)):
        h = UNCERTAINTY_RELATIVE_STEP * max(abs(theta[j]), 1.0)

        tp = theta.copy()
        tm = theta.copy()

        tp[j] += h
        tm[j] -= h

        cp = raw_model_conversion(tp, condition)
        cm = raw_model_conversion(tm, condition)

        grad[j] = (cp - cm) / (2.0 * h)

    return grad


def delta_conversion_uncertainty(theta, covariance, condition):
    """Local first-order parameter + CH4 measurement uncertainty."""
    grad = conversion_gradient_unclipped(theta, condition)

    parameter_variance = float(grad.T @ covariance @ grad)
    parameter_variance = max(parameter_variance, 0.0)

    inlet_ch4 = float(np.asarray(condition, float)[3])

    # yCH4 response measurement VARIANCE propagated to conversion.
    measurement_variance = (SIGMA_Y[0] ** 2) / max(inlet_ch4 ** 2, 1e-24)

    total_variance = parameter_variance + measurement_variance

    return {
        "parameter_sd": math.sqrt(parameter_variance),
        "measurement_sd": math.sqrt(measurement_variance),
        "total_sd": math.sqrt(max(total_variance, 0.0)),
        "total_ci95_half": 1.96 * math.sqrt(max(total_variance, 0.0)),
        "gradient": grad,
    }


def _nearest_psd(covariance):
    """Symmetrise and clip tiny negative eigenvalues from numerical round-off."""
    C = 0.5 * (np.asarray(covariance, float) + np.asarray(covariance, float).T)
    vals, vecs = np.linalg.eigh(C)
    vals = np.maximum(vals, 0.0)
    return vecs @ np.diag(vals) @ vecs.T


def monte_carlo_conversion_uncertainty(
    theta,
    covariance,
    condition,
    seed,
    n_samples=MC_SAMPLES,
):
    """
    Nonlinear propagation of parameter covariance to RAW methane conversion.

    Parameter samples are drawn from N(theta_hat, Cov(theta)) and clipped only
    to the parameter bounds. Conversion itself is NOT clipped when computing
    the spread, which avoids the false near-zero uncertainty caused by output
    saturation/clipping.

    The CH4 response measurement variance is then added to the parameter-driven
    variance to obtain total predictive uncertainty.
    """
    rng = np.random.default_rng(seed)

    theta = np.asarray(theta, dtype=float)
    C = _nearest_psd(covariance)

    try:
        samples = rng.multivariate_normal(
            mean=theta,
            cov=C,
            size=n_samples,
            check_valid="ignore",
        )
    except Exception:
        # Fallback via eigen decomposition.
        vals, vecs = np.linalg.eigh(C)
        vals = np.maximum(vals, 0.0)
        z = rng.normal(size=(n_samples, len(theta)))
        samples = theta + z @ (vecs @ np.diag(np.sqrt(vals))).T

    samples = np.clip(samples, LOWER_BOUNDS, UPPER_BOUNDS)

    raw_conversions = []
    failed = 0

    for th in samples:
        try:
            val = raw_model_conversion(th, condition)
            if np.isfinite(val):
                raw_conversions.append(float(val))
            else:
                failed += 1
        except Exception:
            failed += 1

    raw_conversions = np.asarray(raw_conversions, dtype=float)

    if len(raw_conversions) < max(50, int(0.1 * n_samples)):
        raise RuntimeError(
            "Too many Monte-Carlo MVK simulations failed; "
            f"valid={len(raw_conversions)}, failed={failed}."
        )

    # Parameter-driven spread of the raw model prediction.
    parameter_mean = float(np.mean(raw_conversions))
    parameter_sd = float(np.std(raw_conversions, ddof=1))
    parameter_variance = parameter_sd ** 2

    inlet_ch4 = float(np.asarray(condition, float)[3])
    measurement_variance = (SIGMA_Y[0] ** 2) / max(inlet_ch4 ** 2, 1e-24)
    measurement_sd = math.sqrt(measurement_variance)

    total_variance = parameter_variance + measurement_variance
    total_sd = math.sqrt(max(total_variance, 0.0))

    # Predictive percentiles include measurement noise explicitly.
    measurement_noise_conversion = rng.normal(
        loc=0.0,
        scale=measurement_sd,
        size=len(raw_conversions),
    )

    predictive_samples = raw_conversions + measurement_noise_conversion

    q025, q975 = np.percentile(predictive_samples, [2.5, 97.5])

    return {
        "n_valid": int(len(raw_conversions)),
        "n_failed": int(failed),
        "parameter_mean_raw": parameter_mean,
        "parameter_sd": parameter_sd,
        "measurement_sd": measurement_sd,
        "total_sd": total_sd,
        "total_ci95_half_normal": 1.96 * total_sd,
        "predictive_q025_raw": float(q025),
        "predictive_q975_raw": float(q975),
    }


# =============================================================================
# VIRTUAL PHYSICAL EXPERIMENT
# =============================================================================


def run_virtual_experiment(condition, rng, add_noise=True):
    """
    MVK ground-truth response.

    SIGMA_Y contains response STANDARD DEVIATIONS. We therefore draw
        epsilon ~ N(0, SIGMA_Y**2)
    by passing SIGMA_Y as the `scale` argument to numpy.normal.
    """
    x = condition_to_x(condition)
    y_true = predict_outputs(THETA_TRUE, x)[0]

    if add_noise:
        noise = rng.normal(
            loc=0.0,
            scale=SIGMA_Y,
            size=3,
        )
        y_obs = y_true + noise
    else:
        y_obs = y_true.copy()

    y_obs = np.maximum(y_obs, 0.0)

    inlet_ch4 = float(x[2])
    observed_conversion = float(
        np.clip(
            (inlet_ch4 - y_obs[0]) / max(inlet_ch4, 1e-12),
            0.0,
            1.0,
        )
    )

    true_conversion = float(
        np.clip(
            (inlet_ch4 - y_true[0]) / max(inlet_ch4, 1e-12),
            0.0,
            1.0,
        )
    )

    return x, y_obs, y_true, observed_conversion, true_conversion


def append_experiment(X, Y, x_new, y_new):
    return (
        np.vstack([X, x_new]),
        np.vstack([Y, y_new]),
    )


# =============================================================================
# PLOT
# =============================================================================


def save_publication_plot(history, path):
    """
    Plot AFTER-recalibration conversion together with AFTER-recalibration
    nonlinear Monte-Carlo total predictive 1-SD uncertainty.

    This keeps the central value and uncertainty from the SAME calibrated
    model state, avoiding the mismatch in the earlier script.
    """
    if history.empty:
        return

    fig, ax = plt.subplots(figsize=(7.4, 5.3))

    ax.errorbar(
        history["Experiment"],
        history["Conversion_After_pct"],
        yerr=history["MC_Total_SD_After_pct"],
        fmt="s",
        markersize=5.5,
        markerfacecolor="black",
        markeredgecolor="black",
        ecolor="black",
        elinewidth=1.1,
        capsize=4,
        capthick=1.1,
        linestyle="none",
        label="GOOED conversion",
    )

    ax.plot(
        history["Experiment"],
        history["Observed_Conversion_pct"],
        linestyle="none",
        marker="*",
        markersize=10,
        markerfacecolor="white",
        markeredgecolor="red",
        markeredgewidth=1.2,
        color="red",
        label="In-silico conversion",
    )

    ax.set_xlabel("Experiment")
    ax.set_ylabel("Methane conversion (%)")
    ax.set_xticks(history["Experiment"].to_numpy())

    ax.minorticks_off()
    ax.tick_params(
        axis="both",
        which="major",
        direction="in",
        length=6,
        width=1.0,
    )

    ax.grid(False)

    legend = ax.legend(
        loc="lower left",
        frameon=True,
        fontsize=9,
    )
    legend.get_frame().set_edgecolor("black")
    legend.get_frame().set_linewidth(0.8)
    legend.get_frame().set_facecolor("white")

    fig.tight_layout()
    fig.savefig(path, dpi=600, bbox_inches="tight")
    plt.close(fig)


# =============================================================================
# MAIN GOOED LOOP
# =============================================================================


def run(
    data="Factorial_Data.csv",
    outdir="results_GOOED_full",
    add_noise=True,
):
    out = Path(outdir)
    out.mkdir(parents=True, exist_ok=True)

    _, X, Y = load_initial_12(data)
    rng = np.random.default_rng(RANDOM_SEED)

    print("\nUsing ONLY Experiments 1-12 for the initial mechanistic fit.")

    fit = fit_parameters(
        X,
        Y,
        INITIAL_GUESS,
        N_STARTS_INITIAL,
        RANDOM_SEED,
    )

    parameter_history = [print_parameter_table(12, fit)]
    experiment_history = []

    for exp_no in range(13, MAX_EXPERIMENTS + 1):
        theta_before = fit["theta"].copy()
        covariance_before = fit["covariance"].copy()

        # -----------------------------------------------------------------
        # 1) GOOED selects maximum predicted methane conversion.
        # -----------------------------------------------------------------
        condition, predicted_before = optimize_gooed(
            theta_before,
            RANDOM_SEED + exp_no,
        )

        # -----------------------------------------------------------------
        # 2) Predictive uncertainty BEFORE the new experiment.
        # -----------------------------------------------------------------
        delta_before = delta_conversion_uncertainty(
            theta_before,
            covariance_before,
            condition,
        )

        mc_before = monte_carlo_conversion_uncertainty(
            theta_before,
            covariance_before,
            condition,
            seed=RANDOM_SEED + 10000 + exp_no,
        )

        T, flow, ratio, methane = condition

        print("\n" + "#" * 105)
        print(f"EXPERIMENT {exp_no} | GOOED maximum-conversion design")
        print("#" * 105)
        print(f"Temperature                    = {T:.3f} C")
        print(f"Mass flow                      = {flow:.3f} Nml/min")
        print(f"O2/CH4 mole ratio              = {ratio:.3f}")
        print(f"Inlet methane                  = {methane:.6f}")
        print(f"Predicted conversion BEFORE    = {100*predicted_before:.3f}%")
        print(f"Delta parameter SD BEFORE      = {100*delta_before['parameter_sd']:.3f}%")
        print(f"CH4 measurement SD contribution= {100*delta_before['measurement_sd']:.3f}%")
        print(f"Delta TOTAL SD BEFORE          = {100*delta_before['total_sd']:.3f}%")
        print(f"MC parameter SD BEFORE         = {100*mc_before['parameter_sd']:.3f}%")
        print(f"MC TOTAL SD BEFORE             = {100*mc_before['total_sd']:.3f}%")

        # -----------------------------------------------------------------
        # 3) Ground-truth MVK in-silico experiment + response noise.
        # -----------------------------------------------------------------
        (
            x_new,
            y_obs,
            y_true,
            observed_conversion,
            true_conversion,
        ) = run_virtual_experiment(condition, rng, add_noise=add_noise)

        print(f"Ground-truth conversion         = {100*true_conversion:.3f}%")
        print(f"Noisy in-silico conversion      = {100*observed_conversion:.3f}%")
        print(f"Observed yCH4                    = {y_obs[0]:.8f}")

        # -----------------------------------------------------------------
        # 4) Append new noisy response and recalibrate ALL six parameters.
        # -----------------------------------------------------------------
        X, Y = append_experiment(X, Y, x_new, y_obs)

        fit = fit_parameters(
            X,
            Y,
            theta_before,
            N_STARTS_SEQUENTIAL,
            RANDOM_SEED + 1000 + exp_no,
        )

        parameter_history.append(print_parameter_table(exp_no, fit))

        # -----------------------------------------------------------------
        # 5) Central conversion + uncertainty AFTER recalibration.
        #    The central value and uncertainty now come from the SAME fit.
        # -----------------------------------------------------------------
        conversion_after = physical_model_conversion(
            fit["theta"],
            condition,
        )

        delta_after = delta_conversion_uncertainty(
            fit["theta"],
            fit["covariance"],
            condition,
        )

        mc_after = monte_carlo_conversion_uncertainty(
            fit["theta"],
            fit["covariance"],
            condition,
            seed=RANDOM_SEED + 20000 + exp_no,
        )

        print(f"Conversion AFTER recalibration = {100*conversion_after:.3f}%")
        print(f"MC TOTAL SD AFTER              = {100*mc_after['total_sd']:.3f}%")
        print(
            "Maximum parameter uncertainty   = "
            f"{np.max(fit['relative_uncertainty_pct']):.3f}%"
        )
        print(f"Minimum t-value                 = {np.min(fit['t_values']):.4f}")

        experiment_history.append(
            {
                "Experiment": exp_no,
                "Temperature": float(T),
                "Massflow": float(flow),
                "Moleratio": float(ratio),
                "Inlet_CH4": float(methane),
                "Poutlet": POUTLET_DEFAULT,
                "Pinlet": PINLET_DEFAULT,

                "Predicted_Conversion_Before_pct":
                    100.0 * predicted_before,

                "GroundTruth_Conversion_pct":
                    100.0 * true_conversion,

                "Observed_Conversion_pct":
                    100.0 * observed_conversion,

                "Conversion_After_pct":
                    100.0 * conversion_after,

                "Delta_Parameter_SD_Before_pct":
                    100.0 * delta_before["parameter_sd"],

                "Measurement_SD_Contribution_pct":
                    100.0 * delta_before["measurement_sd"],

                "Delta_Total_SD_Before_pct":
                    100.0 * delta_before["total_sd"],

                "Delta_Total_95CI_Before_pct":
                    100.0 * delta_before["total_ci95_half"],

                "MC_Parameter_SD_Before_pct":
                    100.0 * mc_before["parameter_sd"],

                "MC_Total_SD_Before_pct":
                    100.0 * mc_before["total_sd"],

                "MC_Total_95CI_Before_pct":
                    100.0 * mc_before["total_ci95_half_normal"],

                "MC_Predictive_Q025_Before_pct":
                    100.0 * mc_before["predictive_q025_raw"],

                "MC_Predictive_Q975_Before_pct":
                    100.0 * mc_before["predictive_q975_raw"],

                "Delta_Total_SD_After_pct":
                    100.0 * delta_after["total_sd"],

                "MC_Parameter_SD_After_pct":
                    100.0 * mc_after["parameter_sd"],

                "MC_Total_SD_After_pct":
                    100.0 * mc_after["total_sd"],

                "MC_Total_95CI_After_pct":
                    100.0 * mc_after["total_ci95_half_normal"],

                "Maximum_Parameter_Uncertainty_pct":
                    float(np.max(fit["relative_uncertainty_pct"])),

                "Minimum_t_value":
                    float(np.min(fit["t_values"])),

                "t_Critical_95pct_TwoSided":
                    float(fit["t_critical"]),

                "Residual_Degrees_of_Freedom":
                    int(fit["dof"]),

                "All_t_values_pass":
                    bool(np.all(fit["passed"])),
            }
        )

    experiment_df = pd.DataFrame(experiment_history)
    parameter_df = pd.concat(parameter_history, ignore_index=True)

    experiment_df.to_csv(
        out / "GOOED_experiment_history.csv",
        index=False,
    )

    parameter_df.to_csv(
        out / "GOOED_parameter_history.csv",
        index=False,
    )

    save_publication_plot(
        experiment_df,
        out / "GOOED_conversion_and_uncertainty.png",
    )

    print("\n" + "=" * 105)
    print("FINISHED STANDALONE GOOED")
    print("=" * 105)
    print("Initial fit used Experiments 1-12 only.")
    print("GOOED objective: maximum mechanistic methane conversion.")
    print("Each selected condition is evaluated by the MVK ground-truth model.")
    print("Response noise uses SIGMA_Y as SD; response variance is SIGMA_Y**2.")
    print("All six kinetic parameters are recalibrated after every new experiment.")
    print("Uncertainty uses unclipped conversion propagation.")
    print("Parameter significance uses the 95% two-sided Student-t threshold based on current residual DOF.")
    print("Publication plot uses AFTER-recalibration MC total predictive ±1 SD.")
    print(f"Results saved to: {out.resolve()}")

    return experiment_df, parameter_df


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--data",
        default="Factorial_Data.csv",
    )

    parser.add_argument(
        "--outdir",
        default="results_GOOED_full",
    )

    parser.add_argument(
        "--no-noise",
        action="store_true",
    )

    parser.add_argument(
        "--mc-samples",
        type=int,
        default=MC_SAMPLES,
        help="Monte-Carlo samples used for nonlinear uncertainty propagation.",
    )

    args = parser.parse_args()

    # Allow command-line override without changing the code above.
    MC_SAMPLES = int(args.mc_samples)

    N_TIMING_RUNS = 3
    runtimes = []

    for timing_run in range(1, N_TIMING_RUNS + 1):
        print("\n" + "#" * 105)
        print(f"TIMING RUN {timing_run}/{N_TIMING_RUNS} - GOOED")
        print("#" * 105)

        # Use a separate output folder for each timing run so files are not overwritten.
        timing_outdir = f"{args.outdir}_timing_run_{timing_run}"

        start_time = time.perf_counter()
        run(
            data=args.data,
            outdir=timing_outdir,
            add_noise=not args.no_noise,
        )
        elapsed_time = time.perf_counter() - start_time
        runtimes.append(elapsed_time)

        print(f"GOOED computational time, run {timing_run}: {elapsed_time:.2f} s "
              f"({elapsed_time/60.0:.2f} min)")

    mean_runtime = float(np.mean(runtimes))
    std_runtime = float(np.std(runtimes, ddof=1)) if len(runtimes) > 1 else 0.0

    print("\n" + "=" * 105)
    print("GOOED COMPUTATIONAL-TIME SUMMARY")
    print("=" * 105)
    for i, runtime in enumerate(runtimes, start=1):
        print(f"Run {i}: {runtime:.2f} s ({runtime/60.0:.2f} min)")
    print(f"Mean computational time: {mean_runtime:.2f} s ({mean_runtime/60.0:.2f} min)")
    print(f"Sample SD across 3 runs: {std_runtime:.2f} s")
    print("=" * 105)
