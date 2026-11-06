from __future__ import annotations

import time

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.stats import norm
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, RBF

from benchmark_common import (
    load_initial_12,
    conversion_from_xy,
    virtual_experiment,
    observed_conversion,
    append_experiment,
    fit_parameters,
    parameter_table,
    RANDOM_SEED,
    INITIAL_GUESS,
    SIGMA_Y,
)

MAX_EXPERIMENTS = 20
GP_N_RESTARTS = 10
PI_XI = 0.01

TEMPERATURE_RANGE = (250.0, 350.0)
MASSFLOW_RANGE = (20.0, 30.0)
METHANE_RANGE = (0.005, 0.025)
MOLERATIO_RANGE = (2.0, 4.0)
CANDIDATE_LEVELS = 5

MAX_PHYSICAL_CONVERSION = 0.99
MAX_DISPLAY_SD_PCT = 20.0
DISPLAY_LOWER_PCT = 0.0
DISPLAY_UPPER_PCT = 100.0


def gp_feature_from_x(x_row):
    x_row = np.asarray(x_row, dtype=float).reshape(-1)
    return np.array([x_row[0], x_row[1], x_row[2], x_row[3]], dtype=float)


def gp_feature_from_candidate(candidate):
    return np.array(
        [
            candidate["Temperature"],
            candidate["Massflow"],
            candidate["Inlet conc of Methane"],
            candidate["Moleratio"],
        ],
        dtype=float,
    )


def normalize_gp_inputs(X_gp):
    X_gp = np.asarray(X_gp, dtype=float)
    lower = np.array(
        [TEMPERATURE_RANGE[0], MASSFLOW_RANGE[0], METHANE_RANGE[0], MOLERATIO_RANGE[0]],
        dtype=float,
    )
    upper = np.array(
        [TEMPERATURE_RANGE[1], MASSFLOW_RANGE[1], METHANE_RANGE[1], MOLERATIO_RANGE[1]],
        dtype=float,
    )
    return (X_gp - lower) / (upper - lower)


def generate_candidate_grid():
    temperatures = np.linspace(*TEMPERATURE_RANGE, CANDIDATE_LEVELS)
    massflows = np.linspace(*MASSFLOW_RANGE, CANDIDATE_LEVELS)
    methane_values = np.linspace(*METHANE_RANGE, CANDIDATE_LEVELS)
    moleratios = np.linspace(*MOLERATIO_RANGE, CANDIDATE_LEVELS)

    candidates = []
    for T in temperatures:
        for methane in methane_values:
            for ratio in moleratios:
                for flow in massflows:
                    candidates.append(
                        {
                            "Temperature": float(T),
                            "Massflow": float(flow),
                            "Inlet conc of Methane": float(methane),
                            "Moleratio": float(ratio),
                        }
                    )
    return candidates


def condition_array_for_virtual_experiment(candidate):
    return np.array(
        [
            candidate["Temperature"],
            candidate["Massflow"],
            candidate["Moleratio"],
            candidate["Inlet conc of Methane"],
        ],
        dtype=float,
    )


def same_gp_condition(candidate, feature_row, atol=1e-12):
    return np.allclose(
        gp_feature_from_candidate(candidate),
        np.asarray(feature_row, dtype=float),
        atol=atol,
        rtol=0.0,
    )


def remove_existing_candidates(candidates, X_gp):
    return [
        c for c in candidates
        if not any(same_gp_condition(c, row) for row in X_gp)
    ]


def conversion_noise_std_from_inlet_ch4(inlet_ch4):
    inlet_ch4 = np.asarray(inlet_ch4, dtype=float)
    sigma_ych4 = float(SIGMA_Y[0])
    return sigma_ych4 / np.maximum(inlet_ch4, 1e-12)


def alpha_from_gp_inputs(X_gp):
    sigma_conv = conversion_noise_std_from_inlet_ch4(X_gp[:, 2])
    return np.maximum(sigma_conv**2, 1e-8)


def fit_gp_model(X_gp, y_gp, random_state):
    X_gp = np.asarray(X_gp, dtype=float)
    y_gp = np.asarray(y_gp, dtype=float).reshape(-1)
    Xn = normalize_gp_inputs(X_gp)
    alpha_vec = alpha_from_gp_inputs(X_gp)

    kernel = (
        ConstantKernel(1.0, constant_value_bounds=(1e-3, 1e3))
        * RBF(
            length_scale=np.ones(Xn.shape[1]),
            length_scale_bounds=(1e-2, 1e2),
        )
    )

    gp = GaussianProcessRegressor(
        kernel=kernel,
        alpha=alpha_vec,
        normalize_y=True,
        n_restarts_optimizer=GP_N_RESTARTS,
        random_state=random_state,
    )
    gp.fit(Xn, y_gp)
    return gp, alpha_vec


def gp_predict_candidate(gp, candidate):
    x = gp_feature_from_candidate(candidate).reshape(1, -1)
    xn = normalize_gp_inputs(x)
    raw_mean, raw_std = gp.predict(xn, return_std=True)
    mean = float(np.clip(raw_mean[0], 0.0, MAX_PHYSICAL_CONVERSION))
    return mean, float(raw_std[0])


def probability_of_improvement(mean, std, best_observed, xi=PI_XI):
    mean = float(mean)
    std = float(std)
    best_observed = float(best_observed)
    if std <= 1e-12:
        return 1.0 if mean > (best_observed + xi) else 0.0
    z = (mean - best_observed - xi) / std
    return float(norm.cdf(z))


def score_bo_candidates(gp, candidates, best_observed):
    scored = []
    for candidate in candidates:
        mean, std = gp_predict_candidate(gp, candidate)
        pi = probability_of_improvement(mean, std, best_observed, xi=PI_XI)
        scored.append(
            {
                "candidate": candidate,
                "criterion_value": pi,
                "gp_mean": mean,
                "gp_std": std,
            }
        )
    scored.sort(key=lambda item: item["criterion_value"], reverse=True)
    return scored


def initial_gp_dataset(X, Y):
    X_gp = np.vstack([gp_feature_from_x(row) for row in X])
    y_gp = conversion_from_xy(X, Y)
    y_gp = np.clip(np.asarray(y_gp, dtype=float), 0.0, MAX_PHYSICAL_CONVERSION)
    return X_gp, y_gp


def run(data="Factorial_Data.csv", outdir="results_BO_PI", add_noise=True):
    out = Path(outdir)
    out.mkdir(parents=True, exist_ok=True)

    _, X, Y = load_initial_12(data)
    rng = np.random.default_rng(RANDOM_SEED)

    X_gp, y_gp = initial_gp_dataset(X, Y)
    gp, alpha_vec = fit_gp_model(X_gp, y_gp, random_state=RANDOM_SEED)

    print("\nInitial GP fitted using Experiments 1-12 only.")
    print(f"Initial optimized kernel = {gp.kernel_}")
    print(
        "Initial propagated conversion-noise SD range = "
        f"{100*np.sqrt(alpha_vec.min()):.3f}% to "
        f"{100*np.sqrt(alpha_vec.max()):.3f}%"
    )

    diagnostic_fit = fit_parameters(X, Y, INITIAL_GUESS, 40, RANDOM_SEED)
    parameter_history = parameter_table(12, diagnostic_fit)
    if isinstance(parameter_history, pd.DataFrame):
        parameter_history = [parameter_history]

    candidates = remove_existing_candidates(generate_candidate_grid(), X_gp)
    experiment_history = []

    for exp_no in range(13, MAX_EXPERIMENTS + 1):
        if not candidates:
            break

        best_observed = float(np.max(y_gp))
        ranked = score_bo_candidates(gp, candidates, best_observed)
        selected = ranked[0]
        candidate = selected["candidate"]

        mean_before = float(selected["gp_mean"])
        std_before = float(selected["gp_std"])
        pi_value = float(selected["criterion_value"])

        print("\n" + "=" * 100)
        print(f"EXPERIMENT {exp_no} | GP-BO-PI")
        print("=" * 100)
        print(f"Temperature        = {candidate['Temperature']:.2f} C")
        print(f"Massflow           = {candidate['Massflow']:.2f} Nml/min")
        print(f"Inlet CH4          = {candidate['Inlet conc of Methane']:.6f}")
        print(f"O2/CH4             = {candidate['Moleratio']:.2f}")
        print(f"Best observed      = {100*best_observed:.3f}%")
        print(f"GP mean BEFORE     = {100*mean_before:.3f}%")
        print(f"GP SD BEFORE       = {100*std_before:.3f}%")
        print(f"Probability Improvement = {pi_value:.8f}")

        u = condition_array_for_virtual_experiment(candidate)
        x_new, y_new = virtual_experiment(u, rng, add_noise)
        observed = observed_conversion(x_new, y_new)

        print(f"In-silico observed conversion = {100*observed:.3f}%")

        X_gp = np.vstack([X_gp, gp_feature_from_candidate(candidate)])
        y_gp = np.append(y_gp, observed)
        X, Y = append_experiment(X, Y, x_new, y_new)

        gp, alpha_vec = fit_gp_model(
            X_gp,
            y_gp,
            random_state=RANDOM_SEED + exp_no,
        )

        mean_after, std_after = gp_predict_candidate(gp, candidate)

        print(f"Optimized kernel AFTER = {gp.kernel_}")
        print(f"GP mean AFTER          = {100*mean_after:.3f}%")
        print(f"GP SD AFTER            = {100*std_after:.3f}%")

        diagnostic_fit = fit_parameters(
            X,
            Y,
            diagnostic_fit["theta"],
            25,
            RANDOM_SEED + 1000 + exp_no,
        )

        ptab = parameter_table(exp_no, diagnostic_fit)
        if isinstance(ptab, pd.DataFrame):
            parameter_history.append(ptab)
        else:
            parameter_history.extend(ptab)

        mean_pct = 100.0 * mean_before
        raw_sd_pct = 100.0 * std_before
        display_sd_pct = min(raw_sd_pct, MAX_DISPLAY_SD_PCT)
        lower_pct = max(DISPLAY_LOWER_PCT, mean_pct - display_sd_pct)
        upper_pct = min(DISPLAY_UPPER_PCT, mean_pct + display_sd_pct)

        experiment_history.append(
            {
                "Experiment": exp_no,
                "Temperature": candidate["Temperature"],
                "Massflow": candidate["Massflow"],
                "Inlet_CH4": candidate["Inlet conc of Methane"],
                "Moleratio": candidate["Moleratio"],
                "Best_Observed_Before_pct": 100.0 * best_observed,
                "Probability_of_Improvement": pi_value,
                "GP_Mean_Before_pct": mean_pct,
                "GP_SD_Before_Raw_pct": raw_sd_pct,
                "GP_SD_Before_Display_pct": display_sd_pct,
                "GP_Display_Lower_pct": lower_pct,
                "GP_Display_Upper_pct": upper_pct,
                "Observed_Conversion_pct": 100.0 * observed,
                "GP_Mean_After_pct": 100.0 * mean_after,
                "GP_SD_After_pct": 100.0 * std_after,
                "GP_Kernel_After": str(gp.kernel_),
                "Diagnostic_Minimum_t_value": float(np.min(diagnostic_fit["t_values"])),
            }
        )

        candidates = [c for c in candidates if c is not candidate]

    history_df = pd.DataFrame(experiment_history)

    if parameter_history:
        if isinstance(parameter_history[0], pd.DataFrame):
            parameter_df = pd.concat(parameter_history, ignore_index=True)
        else:
            parameter_df = pd.DataFrame(parameter_history)
    else:
        parameter_df = pd.DataFrame()

    history_df.to_csv(out / "BO_PI_experiment_history.csv", index=False)
    parameter_df.to_csv(out / "BO_PI_parameter_diagnostics.csv", index=False)

    if not history_df.empty:
        mean = history_df["GP_Mean_Before_pct"].to_numpy(float)
        lower = history_df["GP_Display_Lower_pct"].to_numpy(float)
        upper = history_df["GP_Display_Upper_pct"].to_numpy(float)
        yerr = np.vstack([mean - lower, upper - mean])

        fig, ax = plt.subplots(figsize=(9, 6))
        ax.errorbar(
            history_df["Experiment"],
            mean,
            yerr=yerr,
            fmt="s",
            capsize=4,
            label="BO-PI GP prediction",
        )
        ax.set_xlabel("Experiment")
        ax.set_ylabel("Methane conversion (%)")
        ax.set_ylim(50, 100)
        ax.grid(alpha=0.2)
        ax.legend()
        fig.tight_layout()
        fig.savefig(out / "BO_PI_published_style.png", dpi=300)
        plt.close(fig)

    print("\n" + "=" * 100)
    print("STAND-ALONE BO-PI FINISHED")
    print("=" * 100)
    print("Initial GP data: Experiments 1-12 only.")
    print("Acquisition: Probability of Improvement.")
    print(f"PI xi = {PI_XI}.")
    print("Observation noise is propagated from yCH4 measurement uncertainty.")
    print("No artificial cap is applied to the raw GP mean beyond the physical 0-99% bound.")
    print("GP is retrained after each ground-truth BO experiment.")
    print(f"Final GP kernel = {gp.kernel_}")
    print(f"Results saved to: {out.resolve()}")

    return history_df, parameter_df


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="Factorial_Data.csv")
    parser.add_argument("--outdir", default="results_BO_PI")
    parser.add_argument("--no-noise", action="store_true")
    args = parser.parse_args()

    N_TIMING_RUNS = 3
    runtimes = []

    for timing_run in range(1, N_TIMING_RUNS + 1):
        print("\n" + "#" * 100)
        print(f"TIMING RUN {timing_run}/{N_TIMING_RUNS} - BO")
        print("#" * 100)

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

        print(f"BO computational time, run {timing_run}: {elapsed_time:.2f} s "
              f"({elapsed_time/60.0:.2f} min)")

    mean_runtime = float(np.mean(runtimes))
    std_runtime = float(np.std(runtimes, ddof=1)) if len(runtimes) > 1 else 0.0

    print("\n" + "=" * 100)
    print("BO COMPUTATIONAL-TIME SUMMARY")
    print("=" * 100)
    for i, runtime in enumerate(runtimes, start=1):
        print(f"Run {i}: {runtime:.2f} s ({runtime/60.0:.2f} min)")
    print(f"Mean computational time: {mean_runtime:.2f} s ({mean_runtime/60.0:.2f} min)")
    print(f"Sample SD across 3 runs: {std_runtime:.2f} s")
    print("=" * 100)
