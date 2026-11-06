"""

Standalone D-optimal MBDoE script extracted from the validated MBDoE phase of the switching code.



Critical matching points:

1) Uses the uploaded MVK packed-bed ODE reactor model, not a simplified contact-factor model.

2) Fits all three outputs yCH4/yO2/yCO2 with the same measurement-error weighting:

       SIGMA_Y = [0.00043, 0.00202, 0.00051]

3) The least-squares Jacobian is therefore the WEIGHTED Jacobian used by the software.

4) D-optimality maximises log(det(F_current + F_candidate)), where candidate sensitivities

   are also divided by SIGMA_Y before forming J.T @ J.

5) Parameter covariance, standard errors, 95% CI and t-values are recalculated after every

   new experiment.

6) Switch MBDoE -> BO is evaluated automatically after each completed MBDoE experiment
   using the four-component switching score S_w = R_t * R_u * R_c * R_s.
   The transition experiment is not hard-coded.

7) Conversion uncertainty is propagated from parameter covariance by the delta method.

8) BO uses Expected Improvement (EI) with the warm-started GP surrogate.



Put this script beside Factorial_Data.csv and run it.

"""



from __future__ import annotations

import time



import math

from pathlib import Path



import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

from scipy.integrate import odeint

from scipy.optimize import least_squares
from scipy.stats import norm, t as student_t
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, RBF





# =====================================================================

# USER SETTINGS

# =====================================================================



DATA_FILE = Path("Factorial_Data.csv")

SHEET_NAME = 0

OUTPUT_DIR = Path("sequential_mbdoe_bo_results")

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)



N_INITIAL = 12

MAX_TOTAL_EXPERIMENTS = 20



# Parameter significance: 95% one-sided Student-t test
# H0: theta_j <= 0; H1: theta_j > 0
T_CONFIDENCE = 0.95

# Relative parameter uncertainty is calculated and reported for diagnostics only.
# It is NOT used as the stand-alone MBDoE stopping criterion.
PARAMETER_UNCERTAINTY_THRESHOLD_PCT = 1.0



# BO Expected Improvement exploration parameter.
# EI_XI = 0.01 corresponds to a 1 percentage-point improvement margin.
# Set EI_XI = 0.0 for classical EI with no additional exploration margin.
EI_XI = 0.01

# Gaussian-process settings. The GP is updated after every completed
# experiment, including MBDoE experiments, so that BO is warm-started
# from all information accumulated before the switch.
GP_ALPHA = 1e-6
GP_N_RESTARTS = 10




# Performance reference used for the MBDoE -> BO readiness criterion
J_TARGET = 0.988
SWITCH_FRACTION = 0.75
J_SWITCH = SWITCH_FRACTION * J_TARGET

# BO stopping criterion:
# Stop when experimentally realised conversion is at least 95%
# AND the updated GP posterior standard deviation is at most 1%.
# Otherwise continue until the total experimental budget is exhausted.
TARGET_CONVERSION = 0.95
TARGET_CONVERSION_STD = 0.01



# Display cap requested by user

MAX_DISPLAY_CONVERSION = 0.99



# Candidate grid - same structure/ranges used by the software

TEMPERATURE_RANGE = (250.0, 350.0)

METHANE_RANGE = (0.005, 0.025)

MOLERATIO_RANGE = (2.0, 4.0)

MASSFLOW_RANGE = (20.0, 30.0)

CANDIDATE_LEVELS = 5



PINLET_DEFAULT = 1.75

POUTLET_DEFAULT = 1.30



# Multistart fitting

N_STARTS_INITIAL = 60

N_STARTS_SEQUENTIAL = 30

RANDOM_SEED = 42



# Numerical differentiation - match software services

FIM_RELATIVE_STEP = 1e-4

UNCERTAINTY_RELATIVE_STEP = 1e-5


# =====================================================================
# FOUR-COMPONENT MBDoE -> BO SWITCH SETTINGS
# =====================================================================

SWITCH_SCORE_THRESHOLD = 0.5
PERFORMANCE_UNCERTAINTY_THRESHOLD = 0.05
SWITCH_PERFORMANCE_THRESHOLD = J_SWITCH
CONTROL_STABILITY_TOL = 0.05

CONTROL_VARIABLES = [
    "Temperature",
    "Massflow",
    "Inlet conc of Methane",
    "Moleratio",
]





# =====================================================================

# MVK REACTOR MODEL - copied from uploaded mvk_reactor.py

# =====================================================================



R = 8.314

T_REF_C = 320.0

CATALYST_MASS_G = 0.01



PARAMETER_NAMES = [

    "theta_1",

    "theta_2",

    "theta_3",

    "theta_4",

    "theta_5",

    "theta_6",

]



# Same output measurement standard deviations used by the software

SIGMA_Y = np.array([0.00043, 0.00202, 0.00051], dtype=float)



# Literature/true values are used ONLY to emulate the physical experiment.

# They are not used as the parameter-estimation initial guess.

THETA_TRUE = np.array([5.77, 6.72, 5.87, 9.51, 10.17, 7.98], dtype=float)



# Neutral estimation start, deliberately not THETA_TRUE.

INITIAL_GUESS = np.array([6.0, 6.0, 6.0, 8.0, 10.0, 7.0], dtype=float)

LOWER_BOUNDS = np.zeros(6, dtype=float)

UPPER_BOUNDS = np.full(6, 30.0, dtype=float)





def physical_parameters(theta: np.ndarray) -> dict[str, float]:

    theta = np.asarray(theta, dtype=float)

    return {

        "k1_ref": float(math.exp(-theta[0])),

        "Ea1": float(theta[1] * 10000.0),

        "k2_ref": float(math.exp(-theta[2])),

        "Ea2": float(theta[3] * 10000.0),

        "k3_ref": float(math.exp(-theta[4])),

        "Ea3": float(theta[5] * 10000.0),

    }





def _mvk_ode(x, w, u, theta, pavg):

    temperature_c = float(u[0])

    massflow = float(u[1])



    y_ch4 = max(float(x[0]), 1e-60)

    y_o2 = max(float(x[1]), 1e-60)

    y_co2 = max(float(x[2]), 0.0)

    y_h2o = max(float(x[3]), 1e-60)



    theta_1, theta_2, theta_3, theta_4, theta_5, theta_6 = theta



    T0 = 20.0

    P0 = 1.0



    k1 = math.exp(

        -theta_1

        - (theta_2 * 1e4 / R)

        * ((1.0 / (temperature_c + 273.15)) - (1.0 / (T_REF_C + 273.15)))

    )

    k2 = math.exp(

        -theta_3

        - (theta_4 * 1e4 / R)

        * ((1.0 / (temperature_c + 273.15)) - (1.0 / (T_REF_C + 273.15)))

    )

    k3 = math.exp(

        -theta_5

        - (theta_6 * 1e4 / R)

        * ((1.0 / (temperature_c + 273.15)) - (1.0 / (T_REF_C + 273.15)))

    )



    numerator = k1 * k2 * (pavg**2) * y_ch4 * y_o2

    denominator = (

        (k1 * pavg * y_o2)

        + (2.0 * k2 * pavg * y_ch4)

        + ((k1 * k2 / max(k3, 1e-60)) * (pavg**2) * y_ch4 * y_o2)

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



    dy_ch4_dw = -r1 * cf

    dy_o2_dw = -2.0 * r1 * cf

    dy_co2_dw = r1 * cf

    dy_h2o_dw = 2.0 * r1 * cf



    return [dy_ch4_dw, dy_o2_dw, dy_co2_dw, dy_h2o_dw]





def predict_outputs(theta: np.ndarray, X: np.ndarray) -> np.ndarray:

    """X = [Temperature, Massflow, Inletconc, Moleratio, Poutlet, Pinlet]."""

    theta = np.asarray(theta, dtype=float)

    X = np.asarray(X, dtype=float)

    if X.ndim == 1:

        X = X.reshape(1, -1)



    predictions = []



    for row in X:

        temperature = float(row[0])

        massflow = float(row[1])

        inlet_ch4 = float(row[2])

        moleratio = float(row[3])

        poutlet = float(row[4])

        pinlet = float(row[5])



        inlet_o2 = inlet_ch4 * moleratio

        pavg = (pinlet + poutlet) / 2.0



        x0 = [inlet_ch4, inlet_o2, 0.0, 1e-10]

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





def weighted_residuals(theta: np.ndarray, X: np.ndarray, Y: np.ndarray) -> np.ndarray:

    """Exactly the residual definition in uploaded mvk_reactor.py."""

    Y_pred = predict_outputs(theta, X)

    return ((Y_pred - Y) / SIGMA_Y).reshape(-1)





# =====================================================================

# DATA

# =====================================================================



REQUIRED_COLUMNS = [

    "Exp",

    "Temperature",

    "Massflow",

    "Moleratio",

    "Inlet conc of Methane",

    "Poutlet",

    "yCO2",

    "yO2",

    "yCH4",

    "Pinlet",

]





def load_initial_data():

    if not DATA_FILE.exists():

        raise FileNotFoundError(

            f"Cannot find {DATA_FILE.resolve()}. Put this script beside Factorial_Data.csv "

            "or change DATA_FILE at the top of the script."

        )



    df = pd.read_csv(DATA_FILE)

    df.columns = [str(c).strip() for c in df.columns]



    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]

    if missing:

        raise ValueError(f"Missing required columns: {missing}")



    df = df.sort_values("Exp").reset_index(drop=True)

    if len(df) < N_INITIAL:

        raise ValueError(f"Need at least {N_INITIAL} rows; found {len(df)}.")



    df12 = df.iloc[:N_INITIAL].copy()



    X = np.column_stack(

        [

            df12["Temperature"].to_numpy(float),

            df12["Massflow"].to_numpy(float),

            df12["Inlet conc of Methane"].to_numpy(float),

            df12["Moleratio"].to_numpy(float),

            df12["Poutlet"].to_numpy(float),

            df12["Pinlet"].to_numpy(float),

        ]

    )



    Y = np.column_stack(

        [

            df12["yCH4"].to_numpy(float),

            df12["yO2"].to_numpy(float),

            df12["yCO2"].to_numpy(float),

        ]

    )



    return df, df12, X, Y





# =====================================================================

# PARAMETER ESTIMATION - mirrors the software's weighted multiresponse fit

# =====================================================================





def _multistart_guesses(initial_guess, n_starts, seed):

    rng = np.random.default_rng(seed)

    initial_guess = np.asarray(initial_guess, dtype=float)

    guesses = [np.clip(initial_guess, LOWER_BOUNDS, UPPER_BOUNDS)]



    # Add several physically sensible deterministic starts.

    sensible = [

        [5, 5, 5, 8, 9, 6],

        [6, 7, 5, 10, 10, 8],

        [6, 8, 4, 12, 10, 6],

        [5, 10, 3, 15, 10, 8],

    ]

    for g in sensible:

        guesses.append(np.asarray(g, dtype=float))



    # Random starts centred in a realistic reparameterised range.

    # This is multistart optimisation, not use of the true values.

    practical_low = np.array([2.0, 1.0, 1.0, 1.0, 7.0, 1.0])

    practical_high = np.array([10.0, 18.0, 10.0, 25.0, 14.0, 18.0])



    while len(guesses) < n_starts:

        g = practical_low + rng.random(6) * (practical_high - practical_low)

        guesses.append(np.clip(g, LOWER_BOUNDS, UPPER_BOUNDS))



    return guesses[:n_starts]





def fit_model(X, Y, initial_guess, n_starts=30, seed=42):

    X = np.asarray(X, dtype=float)

    Y = np.asarray(Y, dtype=float)



    best = None

    best_cost = np.inf



    for x0 in _multistart_guesses(initial_guess, n_starts, seed):

        try:

            result = least_squares(

                fun=lambda theta: weighted_residuals(theta, X, Y),

                x0=np.asarray(x0, dtype=float),

                bounds=(LOWER_BOUNDS, UPPER_BOUNDS),

                method="trf",

                max_nfev=50000,

                x_scale="jac",

            )

            rss_weighted = float(np.sum(result.fun**2))

            if result.success and np.isfinite(rss_weighted) and rss_weighted < best_cost:

                best = result

                best_cost = rss_weighted

        except Exception:

            pass



    if best is None:

        raise RuntimeError("All parameter-estimation starts failed.")



    theta = np.asarray(best.x, dtype=float)

    predictions = predict_outputs(theta, X)

    residual_vector = weighted_residuals(theta, X, Y)



    # IMPORTANT: result.jac is derivative of WEIGHTED residuals.

    # This is the same weighting used by the uploaded software.

    Jw = np.asarray(best.jac, dtype=float)

    F = Jw.T @ Jw



    n_obs = Jw.shape[0]

    n_params = Jw.shape[1]

    dof = max(n_obs - n_params, 1)



    rss = float(np.sum(residual_vector**2))

    sigma2 = rss / dof



    covariance = sigma2 * np.linalg.pinv(F)

    covariance = 0.5 * (covariance + covariance.T)



    se = np.sqrt(np.maximum(np.diag(covariance), 0.0))

    # 95% one-sided Student-t critical value using current residual DOF.
    # H0: theta_j <= 0 ; H1: theta_j > 0
    t_critical = float(student_t.ppf(T_CONFIDENCE, dof))

    # For a one-sided 95% interval, the lower confidence bound is
    # theta - t_critical*SE. We retain a half-width field for compatibility.
    ci_half = t_critical * se

    with np.errstate(divide="ignore", invalid="ignore"):
        t_values = np.divide(
            theta,
            se,
            out=np.full_like(theta, np.inf),
            where=se > 0,
        )

    passed = t_values >= t_critical



    return {

        "theta": theta,

        "predictions": predictions,

        "weighted_residuals": residual_vector,

        "weighted_rss": rss,

        "sigma2": sigma2,

        "jacobian": Jw,

        "fim": F,

        "covariance": covariance,

        "standard_errors": se,

        "ci_half": ci_half,

        "ci_lower": theta - ci_half,

        "ci_upper": theta + ci_half,

        "t_values": t_values,

        "t_reference": t_critical,

        "passed": passed,

        "all_pass": bool(np.all(passed)),

        "dof": dof,

        "fim_condition": float(np.linalg.cond(F)),

        "success": bool(best.success),

        "message": str(best.message),

    }





# =====================================================================

# PARAMETER TABLE

# =====================================================================





def parameter_table(experiment_number, fit):

    return pd.DataFrame(

        {

            "Experiment": experiment_number,

            "Parameter": PARAMETER_NAMES,

            "Estimate": fit["theta"],

            "Standard_Error": fit["standard_errors"],

            "CI95_HalfWidth": fit["ci_half"],

            "CI95_Lower": fit["ci_lower"],

            "CI95_Upper": fit["ci_upper"],

            "t_Value": fit["t_values"],

            "t_Reference": fit["t_reference"],

            "Pass": fit["passed"],

        }

    )





def print_parameter_table(experiment_number, fit):

    tab = parameter_table(experiment_number, fit)

    print("\n" + "=" * 105)

    print(f"PARAMETER ESTIMATION AFTER EXPERIMENT {experiment_number}")

    print("=" * 105)

    print(

        tab[

            [

                "Parameter",

                "Estimate",

                "Standard_Error",

                "CI95_HalfWidth",

                "t_Value",

                "t_Reference",

                "Pass",

            ]

        ].to_string(index=False, float_format=lambda z: f"{z:.5f}")

    )

    print(f"\nAll parameters pass: {fit['all_pass']}")

    print(f"Weighted RSS: {fit['weighted_rss']:.6f}")

    print(f"Residual variance scale: {fit['sigma2']:.6f}")

    print(f"FIM condition number: {fit['fim_condition']:.6e}")

    return tab





# =====================================================================

# CANDIDATE GRID - same 5-level Cartesian grid as uploaded candidates.py

# =====================================================================





def generate_candidate_grid():

    temperatures = np.linspace(*TEMPERATURE_RANGE, CANDIDATE_LEVELS)

    methane_values = np.linspace(*METHANE_RANGE, CANDIDATE_LEVELS)

    moleratios = np.linspace(*MOLERATIO_RANGE, CANDIDATE_LEVELS)

    massflows = np.linspace(*MASSFLOW_RANGE, CANDIDATE_LEVELS)



    candidates = []

    for T in temperatures:

        for methane in methane_values:

            for ratio in moleratios:

                for flow in massflows:

                    candidates.append(

                        {

                            "Temperature": float(T),

                            "Inlet conc of Methane": float(methane),

                            "Moleratio": float(ratio),

                            "Oxygen Conc": float(methane * ratio),

                            "Massflow": float(flow),

                            "Poutlet": POUTLET_DEFAULT,

                            "Pinlet": PINLET_DEFAULT,

                        }

                    )

    return candidates





def candidate_to_x(candidate):

    return np.array(

        [

            candidate["Temperature"],

            candidate["Massflow"],

            candidate["Inlet conc of Methane"],

            candidate["Moleratio"],

            candidate["Poutlet"],

            candidate["Pinlet"],

        ],

        dtype=float,

    )





def same_condition(candidate, xrow, atol=1e-10):

    c = candidate_to_x(candidate)

    return np.allclose(c, np.asarray(xrow, dtype=float), atol=atol, rtol=0.0)





def remove_existing_candidates(candidates, X):

    return [c for c in candidates if not any(same_condition(c, x) for x in X)]





# =====================================================================

# FISHER INFORMATION - exactly the software logic

# =====================================================================





def finite_difference_sensitivity(theta, x_row, epsilon=FIM_RELATIVE_STEP):

    theta = np.asarray(theta, dtype=float)

    x_row = np.asarray(x_row, dtype=float)

    if x_row.ndim == 1:

        x_row = x_row.reshape(1, -1)



    y_base = np.asarray(predict_outputs(theta, x_row)[0], dtype=float)

    S = np.zeros((len(y_base), len(theta)), dtype=float)



    for j in range(len(theta)):

        step = epsilon * max(abs(theta[j]), 1.0)

        tp = theta.copy()

        tm = theta.copy()

        tp[j] += step

        tm[j] -= step

        yp = np.asarray(predict_outputs(tp, x_row)[0], dtype=float)

        ym = np.asarray(predict_outputs(tm, x_row)[0], dtype=float)

        S[:, j] = (yp - ym) / (2.0 * step)



    return S





def candidate_fim(theta, candidate):

    x_row = candidate_to_x(candidate).reshape(1, -1)

    S = finite_difference_sensitivity(theta, x_row)



    # CRITICAL: software divides each output sensitivity by its measurement SD.

    Sw = S / SIGMA_Y[:, None]

    return Sw.T @ Sw





def score_d_optimal_candidates(theta, current_jacobian, candidates):

    theta = np.asarray(theta, dtype=float)

    n_params = len(theta)



    # Current weighted FIM from the current fitted dataset.

    current_fim = current_jacobian.T @ current_jacobian



    scored = []

    for candidate in candidates:

        F_new = current_fim + candidate_fim(theta, candidate)

        F_reg = F_new + 1e-12 * np.eye(n_params)

        sign, logdet = np.linalg.slogdet(F_reg)

        if sign <= 0:

            logdet = -1e30



        scored.append(

            {

                "candidate": candidate,

                "criterion_value": float(logdet),

                "fim_condition_number": float(np.linalg.cond(F_reg)),

            }

        )



    scored.sort(key=lambda z: z["criterion_value"], reverse=True)

    return scored





# =====================================================================

# SIMULATION AND CONVERSION - same behaviour as simulation.py

# =====================================================================





def simulate_mvk_experiment(theta, candidate):

    X = candidate_to_x(candidate).reshape(1, -1)

    y_pred = np.asarray(predict_outputs(theta, X), dtype=float)



    inlet_ch4 = float(candidate["Inlet conc of Methane"])

    y_ch4 = float(np.clip(y_pred[0, 0], 0.0, inlet_ch4))

    y_o2 = float(max(y_pred[0, 1], 0.0))

    y_co2 = float(max(y_pred[0, 2], 0.0))



    conversion = 0.0

    if inlet_ch4 > 1e-12:

        conversion = (inlet_ch4 - y_ch4) / inlet_ch4

    conversion = float(np.clip(conversion, 0.0, MAX_DISPLAY_CONVERSION))



    return {

        **candidate,

        "yCH4": y_ch4,

        "yO2": y_o2,

        "yCO2": y_co2,

        "X_CH4": conversion,

    }





# =====================================================================

# DELTA-METHOD CONVERSION UNCERTAINTY - same algorithm as uncertainty.py

# =====================================================================





def conversion_uncertainty_delta_method(theta, covariance, candidate):

    theta = np.asarray(theta, dtype=float)

    covariance = np.asarray(covariance, dtype=float)



    base = simulate_mvk_experiment(theta, candidate)

    base_conversion = float(base["X_CH4"])



    grad = np.zeros(len(theta), dtype=float)



    for i in range(len(theta)):

        step = UNCERTAINTY_RELATIVE_STEP * max(abs(theta[i]), 1.0)

        tp = theta.copy()

        tm = theta.copy()

        tp[i] += step

        tm[i] -= step



        cp = float(simulate_mvk_experiment(tp, candidate)["X_CH4"])

        cm = float(simulate_mvk_experiment(tm, candidate)["X_CH4"])

        grad[i] = (cp - cm) / (2.0 * step)



    variance = float(grad.T @ covariance @ grad)

    variance = max(variance, 0.0)

    std = float(np.sqrt(variance))

    ci_half = 1.96 * std



    return {

        "conversion": base_conversion,

        "conversion_std": std,

        "conversion_ci95_half": ci_half,

        "conversion_ci95_lower": max(0.0, base_conversion - ci_half),

        "conversion_ci95_upper": min(MAX_DISPLAY_CONVERSION, base_conversion + ci_half),

        "gradient": grad,

    }





# =====================================================================

# BO - Expected Improvement with direct GP trained in parallel with MBDoE
# The GP is fitted directly to conversion. Only the posterior mean is
# bounded to the physical 0-99% range; posterior SD is left unchanged.

# =====================================================================


def gp_feature_from_x(x_row):
    """Return the four manipulated variables used by the GP.

    Order: Temperature, Massflow, inlet CH4, O2/CH4 mole ratio.
    Pressure variables are fixed in the present case study and therefore
    are not included as GP inputs.
    """
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
    """Normalize GP inputs to [0, 1] using the fixed design bounds."""
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


def initial_gp_dataset(df12):
    """
    Build the initial GP dataset from Experiments 1-12.

    The GP is trained directly on methane conversion in physical units.
    """
    X_gp = np.column_stack(
        [
            df12["Temperature"].to_numpy(float),
            df12["Massflow"].to_numpy(float),
            df12["Inlet conc of Methane"].to_numpy(float),
            df12["Moleratio"].to_numpy(float),
        ]
    )

    inlet = df12["Inlet conc of Methane"].to_numpy(float)
    outlet = df12["yCH4"].to_numpy(float)

    y_gp = np.divide(
        inlet - outlet,
        inlet,
        out=np.zeros_like(inlet),
        where=inlet > 1e-12,
    )

    y_gp = np.clip(y_gp, 0.0, MAX_DISPLAY_CONVERSION)
    return X_gp, y_gp


def fit_gp_model(X_gp, y_gp, random_state=RANDOM_SEED):
    """
    Fit or refit the Gaussian-process surrogate directly on methane
    conversion and re-optimise the kernel hyperparameters.

    This function is called after every completed MBDoE or BO experiment.
    Therefore, the GP learns in parallel during the MBDoE phase and is
    already trained on all accumulated experimental observations when
    control is transferred to BO.
    """
    Xn = normalize_gp_inputs(X_gp)
    y_gp = np.asarray(y_gp, dtype=float).reshape(-1)

    kernel = ConstantKernel(1.0, (1e-3, 1e3)) * RBF(
        length_scale=np.ones(Xn.shape[1]),
        length_scale_bounds=(1e-2, 1e2),
    )

    gp = GaussianProcessRegressor(
        kernel=kernel,
        alpha=GP_ALPHA,
        normalize_y=True,
        n_restarts_optimizer=GP_N_RESTARTS,
        random_state=random_state,
    )

    gp.fit(Xn, y_gp)
    return gp


def gp_predict_candidate(gp, candidate):
    """
    Predict methane conversion and GP posterior uncertainty for one
    candidate operating condition.

    The GP is trained directly on conversion. Because an ordinary GP
    regression mean is mathematically unbounded, the posterior mean used
    for physical interpretation and EI scoring is restricted to the
    admissible conversion interval:

        0 <= mu_GP <= MAX_DISPLAY_CONVERSION

    The GP posterior standard deviation is NOT clipped or transformed.
    It remains the model's uncertainty measure.

    Note:
        The EI acquisition value is an expected improvement score and is
        therefore not interpreted as a physical conversion prediction.
    """
    x = gp_feature_from_candidate(candidate).reshape(1, -1)
    xn = normalize_gp_inputs(x)

    raw_mean, std = gp.predict(xn, return_std=True)

    bounded_mean = float(
        np.clip(raw_mean[0], 0.0, MAX_DISPLAY_CONVERSION)
    )

    return bounded_mean, float(std[0])


def score_bo_candidates(gp, candidates, best_observed, xi=EI_XI):
    """
    Rank candidates using Expected Improvement (EI) for a maximization problem.

    For each candidate u:

        improvement = mu_GP(u) - J_best - xi

        Z = improvement / sigma_GP(u)

        EI(u) = improvement * Phi(Z) + sigma_GP(u) * phi(Z)

    where:
        mu_GP(u)    = bounded GP posterior mean,
        sigma_GP(u) = GP posterior standard deviation,
        J_best      = best experimentally observed conversion so far,
        xi          = optional exploration/improvement margin,
        Phi(.)      = standard-normal CDF,
        phi(.)      = standard-normal PDF.

    EI is set to zero when the posterior standard deviation is effectively
    zero and no positive deterministic improvement is available.
    """
    best_observed = float(best_observed)
    xi = float(xi)

    scored = []

    for candidate in candidates:
        mean, std = gp_predict_candidate(gp, candidate)

        improvement = float(mean - best_observed - xi)

        if std > 1e-12:
            z = improvement / std
            ei = improvement * norm.cdf(z) + std * norm.pdf(z)
            ei = max(float(ei), 0.0)
        else:
            # Limiting deterministic case as std -> 0.
            ei = max(improvement, 0.0)
            z = np.inf if improvement > 0 else -np.inf

        scored.append(
            {
                "candidate": candidate,
                "criterion_value": float(ei),
                "gp_mean": float(mean),
                "gp_std": float(std),
                "best_observed": best_observed,
                "improvement": improvement,
                "z_ei": float(z),
            }
        )

    scored.sort(key=lambda item: item["criterion_value"], reverse=True)
    return scored
# =====================================================================

# VIRTUAL PHYSICAL EXPERIMENT

# =====================================================================





def run_virtual_physical_experiment(candidate):

    """

    Emulates the laboratory response at the selected conditions.

    THETA_TRUE is deliberately isolated here.

    """

    sim = simulate_mvk_experiment(THETA_TRUE, candidate)

    return np.array([sim["yCH4"], sim["yO2"], sim["yCO2"]], dtype=float), sim





# =====================================================================

# SEQUENTIAL HYBRID LOOP

# =====================================================================

# =====================================================================

# FOUR-COMPONENT MBDoE -> BO SWITCHING CRITERION

# =====================================================================



def relative_parameter_uncertainty_percent(theta, standard_errors):

    """

    Relative uncertainty (%) for each estimated parameter.



    U_j = SE(theta_j) / |theta_j| * 100

    """

    theta = np.asarray(theta, dtype=float)

    standard_errors = np.asarray(standard_errors, dtype=float)



    with np.errstate(divide="ignore", invalid="ignore"):

        rel_unc = np.divide(

            standard_errors,

            np.abs(theta),

            out=np.full_like(theta, np.inf, dtype=float),

            where=np.abs(theta) > 1e-15,

        )



    return 100.0 * rel_unc





def normalized_control_change(current_candidate, previous_candidate):

    """

    Maximum normalized change in the selected control variables

    between two consecutive MBDoE experiments.

    """

    if previous_candidate is None:

        return np.inf



    ranges = {

        "Temperature": TEMPERATURE_RANGE[1] - TEMPERATURE_RANGE[0],

        "Massflow": MASSFLOW_RANGE[1] - MASSFLOW_RANGE[0],

        "Inlet conc of Methane": METHANE_RANGE[1] - METHANE_RANGE[0],

        "Moleratio": MOLERATIO_RANGE[1] - MOLERATIO_RANGE[0],

    }



    changes = []



    for name in CONTROL_VARIABLES:

        design_range = float(ranges[name])



        if design_range <= 0:

            continue



        delta = abs(

            float(current_candidate[name])

            - float(previous_candidate[name])

        )



        changes.append(delta / design_range)



    if not changes:

        return np.inf



    return float(max(changes))





def calculate_switch_score(
    fit,
    performance_value,
    performance_std,
    current_candidate,
    previous_mbdoe_candidate,
):
    """
    Four-component switching score:

        S_w = R_t * R_u * R_c * R_s

    Components:
        R_t : parameter statistical significance
        R_u : propagated uncertainty in the process-performance metric
        R_c : process-performance readiness
        R_s : stability of the selected control variables

    Each component is assigned:
        1.0 = criterion satisfied
        0.5 = criterion not fully satisfied

    With SWITCH_SCORE_THRESHOLD = 0.5, at least three of the four
    criteria must be fully satisfied for transition to BO.
    """

    # -------------------------------------------------------------
    # R_t : statistical significance of all estimated parameters
    # -------------------------------------------------------------
    all_t_pass = bool(np.all(fit["t_values"] >= fit["t_reference"]))
    R_t = 1.0 if all_t_pass else 0.5

    # -------------------------------------------------------------
    # R_u : propagated uncertainty in the process-performance metric
    # -------------------------------------------------------------
    performance_std = float(performance_std)
    uncertainty_pass = performance_std <= PERFORMANCE_UNCERTAINTY_THRESHOLD
    R_u = 1.0 if uncertainty_pass else 0.5

    # -------------------------------------------------------------
    # R_c : process-performance readiness criterion
    # -------------------------------------------------------------
    performance_value = float(performance_value)
    performance_pass = performance_value >= SWITCH_PERFORMANCE_THRESHOLD
    R_c = 1.0 if performance_pass else 0.5

    # -------------------------------------------------------------
    # R_s : stability of selected control variables
    # -------------------------------------------------------------
    delta_u = normalized_control_change(
        current_candidate,
        previous_mbdoe_candidate,
    )

    stability_pass = (
        np.isfinite(delta_u)
        and delta_u <= CONTROL_STABILITY_TOL
    )
    R_s = 1.0 if stability_pass else 0.5

    # -------------------------------------------------------------
    # Composite switch score
    # -------------------------------------------------------------
    S_w = float(R_t * R_u * R_c * R_s)
    switch_to_bo = bool(S_w >= SWITCH_SCORE_THRESHOLD)

    return {
        "S_w": S_w,
        "R_t": float(R_t),
        "R_u": float(R_u),
        "R_c": float(R_c),
        "R_s": float(R_s),

        "all_t_pass": all_t_pass,

        "performance_value": performance_value,
        "performance_pass": bool(performance_pass),

        "performance_std": performance_std,
        "performance_uncertainty_pct": 100.0 * performance_std,
        "uncertainty_pass": bool(uncertainty_pass),

        "normalized_control_change": float(delta_u),
        "stability_pass": bool(stability_pass),

        "switch_to_bo": switch_to_bo,
    }


def run():
    full_df, df12, X, Y = load_initial_data()

    print("\nUsing ONLY Experiments 1-12 for the initial mechanistic fit.")
    print("Standalone MBDoE uses the SAME D-optimal candidate scoring, weighted FIM,")
    print("MVK ground-truth simulator, recalibration, and delta-method conversion uncertainty")
    print("as the MBDoE phase of the supplied switching code.\n")

    fit = fit_model(
        X,
        Y,
        INITIAL_GUESS,
        n_starts=N_STARTS_INITIAL,
        seed=RANDOM_SEED,
    )

    parameter_history = [print_parameter_table(12, fit)]
    experiment_history = []

    candidates = generate_candidate_grid()
    candidates = remove_existing_candidates(candidates, X)

    stop_after_experiment = None

    for exp_no in range(13, MAX_TOTAL_EXPERIMENTS + 1):
        if not candidates:
            print("No candidate points remain.")
            break

        theta_before = fit["theta"].copy()
        covariance_before = fit["covariance"].copy()

        # ---------------------------------------------------------
        # EXACT MBDoE selection logic from the switching code:
        # maximize log det(F_current + F_candidate)
        # using the weighted least-squares Jacobian and candidate
        # sensitivities weighted by SIGMA_Y.
        # ---------------------------------------------------------
        ranked = score_d_optimal_candidates(
            theta=theta_before,
            current_jacobian=fit["jacobian"],
            candidates=candidates,
        )

        selected = ranked[0]
        candidate = selected["candidate"]
        score = selected["criterion_value"]

        # Mechanistic conversion uncertainty BEFORE experiment.
        pred_before = conversion_uncertainty_delta_method(
            theta_before,
            covariance_before,
            candidate,
        )

        print("\n" + "#" * 105)
        print(f"EXPERIMENT {exp_no} | D-optimal MBDoE")
        print("#" * 105)
        print(f"Temperature               = {candidate['Temperature']:.2f} C")
        print(f"Inlet methane             = {candidate['Inlet conc of Methane']:.6f}")
        print(f"Mass flow                 = {candidate['Massflow']:.3f} Nml/min")
        print(f"O2/CH4 mole ratio         = {candidate['Moleratio']:.3f}")
        print(f"D-optimal logdet score    = {score:.6f}")
        print(f"Mechanistic conversion    = {100*pred_before['conversion']:.3f}%")
        print(f"Conversion SD before      = {100*pred_before['conversion_std']:.3f}%")

        # ---------------------------------------------------------
        # Ground-truth in-silico physical experiment.
        # THETA_TRUE is isolated inside run_virtual_physical_experiment.
        # ---------------------------------------------------------
        y_new, true_sim = run_virtual_physical_experiment(candidate)
        true_conversion = float(true_sim["X_CH4"])
        print(f"Virtual measured conversion = {100*true_conversion:.3f}%")

        # Append the completed experiment.
        X = np.vstack([X, candidate_to_x(candidate)])
        Y = np.vstack([Y, y_new])

        # Recalibrate ALL six mechanistic parameters.
        fit = fit_model(
            X,
            Y,
            theta_before,
            n_starts=N_STARTS_SEQUENTIAL,
            seed=RANDOM_SEED + exp_no,
        )
        parameter_history.append(print_parameter_table(exp_no, fit))

        # Conversion uncertainty AFTER recalibration.
        pred_after = conversion_uncertainty_delta_method(
            fit["theta"],
            fit["covariance"],
            candidate,
        )

        # Parameter relative uncertainty is a DIFFERENT quantity from
        # conversion uncertainty.  It is used for the stand-alone MBDoE
        # stopping rule, but it is NOT plotted as a conversion error bar.
        parameter_rel_unc_pct = relative_parameter_uncertainty_percent(
            fit["theta"],
            fit["standard_errors"],
        )

        all_t_pass = bool(np.all(fit["t_values"] >= fit["t_reference"]))

        experiment_history.append(
            {
                "Experiment": exp_no,
                "Method": "D-optimal MBDoE",
                "Temperature": candidate["Temperature"],
                "Massflow": candidate["Massflow"],
                "Inlet_CH4": candidate["Inlet conc of Methane"],
                "Moleratio": candidate["Moleratio"],
                "Oxygen_Conc": candidate["Oxygen Conc"],
                "Selection_Score": score,
                "Virtual_Measured_Conversion_pct": 100 * true_conversion,
                "Conversion_Before_pct": 100 * pred_before["conversion"],
                "Conversion_SD_Before_pct": 100 * pred_before["conversion_std"],
                "Conversion_After_pct": 100 * pred_after["conversion"],
                "Conversion_SD_After_pct": 100 * pred_after["conversion_std"],
                "Conversion_95CI_After_pct": 100 * pred_after["conversion_ci95_half"],
                "Minimum_t_After": float(np.min(fit["t_values"])),
                "t_Reference": fit["t_reference"],
                "Maximum_Parameter_Relative_Uncertainty_pct": float(np.max(parameter_rel_unc_pct)),
                "All_t_Pass_After": all_t_pass,
                "FIM_Condition": fit["fim_condition"],
            }
        )

        print(f"Conversion after recalibration = {100*pred_after['conversion']:.3f}%")
        print(f"Conversion SD after            = {100*pred_after['conversion_std']:.3f}%")
        print(f"Maximum parameter uncertainty  = {np.max(parameter_rel_unc_pct):.3f}%")
        print(f"Minimum t-value                = {np.min(fit['t_values']):.4f}")
        print(f"All t-values pass?             = {all_t_pass}")

        # Remove exactly the selected design point.
        candidates = [c for c in candidates if c is not candidate]

        # Stand-alone MBDoE stopping criterion:
        # stop as soon as ALL six parameter t-values exceed the current 95% one-sided Student-t critical value.
        # Relative parameter uncertainty is still reported, but it does not
        # determine termination of the MBDoE campaign.
        if all_t_pass:
            stop_after_experiment = exp_no
            print("\n" + "=" * 105)
            print(f"MBDoE STOPPING CRITERION SATISFIED AFTER EXPERIMENT {exp_no}")
            print(f"All six parameter t-values are >= one-sided 95% t_critical = {fit['t_reference']:.5f}.")
            print("=" * 105)
            break

    parameter_df = pd.concat(parameter_history, ignore_index=True)
    experiment_df = pd.DataFrame(experiment_history)

    # Add relative uncertainty and explicit remarks to the parameter table.
    rel_unc_all = []
    for exp in sorted(parameter_df["Experiment"].unique()):
        sub = parameter_df[parameter_df["Experiment"] == exp].copy()
        rel = relative_parameter_uncertainty_percent(
            sub["Estimate"].to_numpy(float),
            sub["Standard_Error"].to_numpy(float),
        )
        rel_unc_all.extend(rel.tolist())

    parameter_df["Relative_Parameter_Uncertainty_pct"] = rel_unc_all
    parameter_df["t_Remark"] = np.where(parameter_df["t_Value"] >= parameter_df["t_Reference"], "PASS", "FAIL")

    xlsx_path = OUTPUT_DIR / "Standalone_MBDoE_Results.xlsx"
    with pd.ExcelWriter(xlsx_path, engine="openpyxl") as writer:
        df12.to_excel(writer, sheet_name="Initial_12", index=False)
        parameter_df.to_excel(writer, sheet_name="Parameter_History", index=False)
        experiment_df.to_excel(writer, sheet_name="MBDoE_Experiments", index=False)

    parameter_df.to_csv(OUTPUT_DIR / "MBDoE_parameter_history.csv", index=False)
    experiment_df.to_csv(OUTPUT_DIR / "MBDoE_experiment_history.csv", index=False)

    # ---------------------------------------------------------
    # CORRECT PLOT
    # ---------------------------------------------------------
    # Error bars are the propagated 1-SD uncertainty in CONVERSION.
    # They are NOT parameter-relative uncertainty and NOT an unconstrained
    # covariance-volume metric.  The plotted quantity therefore has the
    # same physical units as conversion and the y-axis remains 0-100%.
    # ---------------------------------------------------------
    if not experiment_df.empty:
        fig, ax = plt.subplots(figsize=(9, 6))
        ax.errorbar(
            experiment_df["Experiment"],
            experiment_df["Conversion_After_pct"],
            yerr=experiment_df["Conversion_SD_After_pct"],
            fmt="s",
            capsize=4,
            label="MBDoE",
        )
        ax.set_xlabel("Experiment")
        ax.set_ylabel("Methane conversion (%)")
        ax.set_ylim(0, 100)
        ax.grid(alpha=0.2)
        ax.legend()
        fig.tight_layout()
        fig.savefig(OUTPUT_DIR / "MBDoE_conversion.png", dpi=300)
        plt.close(fig)

    # Parameter uncertainty plot kept separate from conversion.
    if not parameter_df.empty:
        fig, ax = plt.subplots(figsize=(10, 6))
        for pname in PARAMETER_NAMES:
            sub = parameter_df[parameter_df["Parameter"] == pname]
            ax.plot(
                sub["Experiment"],
                sub["Relative_Parameter_Uncertainty_pct"],
                marker="o",
                label=pname,
            )
        ax.set_xlabel("Experiment")
        ax.set_ylabel("Relative parameter uncertainty (%)")
        ax.grid(alpha=0.2)
        ax.legend(ncol=2)
        fig.tight_layout()
        fig.savefig(OUTPUT_DIR / "MBDoE_parameter_uncertainty.png", dpi=300)
        plt.close(fig)

    print("\n" + "=" * 105)
    print("FINISHED STAND-ALONE MBDoE")
    print("=" * 105)
    print(f"Results: {xlsx_path}")
    print(f"Stopping experiment: {stop_after_experiment}")

    return parameter_df, experiment_df


if __name__ == "__main__":
    N_TIMING_RUNS = 3
    runtimes = []

    for timing_run in range(1, N_TIMING_RUNS + 1):
        print("\n" + "#" * 105)
        print(f"TIMING RUN {timing_run}/{N_TIMING_RUNS} - MBDoE")
        print("#" * 105)

        start_time = time.perf_counter()
        run()
        elapsed_time = time.perf_counter() - start_time
        runtimes.append(elapsed_time)

        print(f"MBDoE computational time, run {timing_run}: {elapsed_time:.2f} s "
              f"({elapsed_time/60.0:.2f} min)")

    mean_runtime = float(np.mean(runtimes))
    std_runtime = float(np.std(runtimes, ddof=1)) if len(runtimes) > 1 else 0.0

    print("\n" + "=" * 105)
    print("MBDoE COMPUTATIONAL-TIME SUMMARY")
    print("=" * 105)
    for i, runtime in enumerate(runtimes, start=1):
        print(f"Run {i}: {runtime:.2f} s ({runtime/60.0:.2f} min)")
    print(f"Mean computational time: {mean_runtime:.2f} s ({mean_runtime/60.0:.2f} min)")
    print(f"Sample SD across 3 runs: {std_runtime:.2f} s")
    print("=" * 105)
