# =============================================================================
# Author : Xiaoyan Wu
# Date   : June 2026
# Description: Model fitting — maximum likelihood estimation for motive-cocktail models M1–M8 on TPI AI data
# =============================================================================

"""
Model fitting for AI-generated TPP data.

For logprobs data (p_yes continuous ∈ [0,1]):
    lik += p_yes * log(p_model) + (1 - p_yes) * log(1 - p_model)

This is the expected log-likelihood (cross-entropy), the correct
adaptation of binary log-likelihood for soft probability observations.

For repeated-sampling data (intervene ∈ {0,1}):
    lik += action * log(p_model) + (1 - action) * log(1 - p_model)

Variable mapping (MATLAB → Python):
    data.block     → block:  1=punish, 2=help
    data.ratio     → ratio:  1.5 or 3.0
    data.violator  → x1:     allocator tokens
    data.victim    → x2:     receiver tokens
    data.cost      → cost
    data.action    → p_yes (logprobs) or intervene (repeated sampling)
"""

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

TP = 50  # third-party starting tokens

# ─── Utility functions (translated from MATLAB) ───────────────────────────────

def compute_p_model(x, row):
    """
    Returns p_model = P(intervene=Yes) for a given model and trial.
    Dispatched by model index stored in x['model_id'].
    """
    block = row["block"]   # 1=punish, 2=help
    ratio = row["ratio"]
    x1    = row["x1"]
    x2    = row["x2"]
    cost  = row["cost"]
    mid   = row["model_id"]

    x3 = TP
    x3s = x3 - cost

    if block == 1:   # punish
        x1s = x1 - cost * ratio
        x2s = x2
    else:            # help
        x1s = x1
        x2s = x2 + ratio * cost

    # Softmax helper
    def softmax_yes(Uyes, Uno, lam):
        return 1.0 / (1.0 + np.exp(np.clip(lam * (Uno - Uyes), -500, 500)))

    if mid == 1:
        # m1: fixed p
        p = x[0]
        return np.clip(p, 1e-6, 1 - 1e-6)

    elif mid == 2:
        # m2: SI only
        lam = x[0]
        Uyes = x3s
        Uno  = x3
        return softmax_yes(Uyes, Uno, lam)

    elif mid == 3:
        # m3: SI + SCI
        envy, guilt, lam = x[0], x[1], x[2]
        disad = max(x1s - x3s, 0) + max(x2s - x3s, 0)
        ad    = max(x3s - x1s, 0) + max(x3s - x2s, 0)
        Uyes = x3s - envy*disad - guilt*ad
        Uno  = x3  - envy*(max(x1-x3,0)+max(x2-x3,0)) \
                   - guilt*(max(x3-x1,0)+max(x3-x2,0))
        return softmax_yes(Uyes, Uno, lam)

    elif mid == 4:
        # m4: SI + SCI + VCI
        gama, envy, guilt, lam = x[0], x[1], x[2], x[3]
        disad = max(x1s-x3s,0)+max(x2s-x3s,0)
        ad    = max(x3s-x1s,0)+max(x3s-x2s,0)
        inqua = max(x1s-x2s,0)
        Uyes = x3s - envy*disad - guilt*ad - gama*inqua
        Uno  = x3  - envy*(max(x1-x3,0)+max(x2-x3,0)) \
                   - guilt*(max(x3-x1,0)+max(x3-x2,0)) \
                   - gama*max(x1-x2,0)
        return softmax_yes(Uyes, Uno, lam)

    elif mid == 5:
        # m5: SI + SCI + VCI + EC
        gama, envy, guilt, lam, omega = x[0], x[1], x[2], x[3], x[4]
        disad = max(x1s-x3s,0)+max(x2s-x3s,0)
        ad    = max(x3s-x1s,0)+max(x3s-x2s,0)
        inqua = max(x1s-x2s,0)
        EP    = x1s + x2s
        Uyes = x3s - envy*disad - guilt*ad - gama*inqua + omega*EP
        Uno  = x3  - envy*(max(x1-x3,0)+max(x2-x3,0)) \
                   - guilt*(max(x3-x1,0)+max(x3-x2,0)) \
                   - gama*max(x1-x2,0) + omega*(x1+x2)
        return softmax_yes(Uyes, Uno, lam)

    elif mid == 6:
        # m6: SI + SCI + VCI + EC + RP  ← motive cocktail
        gama, envy, guilt, lam, omega, kapa = x[0], x[1], x[2], x[3], x[4], x[5]
        disad = max(x1s-x3s,0)+max(x2s-x3s,0)
        ad    = max(x3s-x1s,0)+max(x3s-x2s,0)
        inqua = max(x1s-x2s,0)
        EP    = x1s + x2s
        RP    = max(x2s-x1s,0)
        Uyes = x3s - envy*disad - guilt*ad - gama*inqua + omega*EP + kapa*RP
        Uno  = x3  - envy*(max(x1-x3,0)+max(x2-x3,0)) \
                   - guilt*(max(x3-x1,0)+max(x3-x2,0)) \
                   - gama*max(x1-x2,0) + omega*(x1+x2) + kapa*max(x2-x1,0)
        return softmax_yes(Uyes, Uno, lam)

    elif mid == 7:
        # m7: m6 + II (inaction inequality attention)
        gama, envy, guilt, lam, omega, kapa, etak, etaa = x[0],x[1],x[2],x[3],x[4],x[5],x[6],x[7]
        disad = max(x1s-x3s,0)+max(x2s-x3s,0)
        ad    = max(x3s-x1s,0)+max(x3s-x2s,0)
        inqua = max(x1s-x2s,0)
        EP    = x1s + x2s
        RP    = max(x2s-x1s,0)
        IIk   = 2.0 / (1.0 + np.exp(np.clip(etak*(cost/50), -500, 500)))
        IIa   = 2.0 / (1.0 + np.exp(np.clip(etaa*(cost/50), -500, 500)))
        Uyes = x3s - envy*disad - guilt*ad - gama*inqua*IIa + omega*EP + kapa*RP
        Uno  = x3  - envy*(max(x1-x3,0)+max(x2-x3,0)) \
                   - guilt*(max(x3-x1,0)+max(x3-x2,0)) \
                   - gama*(max(x1-x2,0)*IIk) + omega*(x1+x2) + kapa*max(x2-x1,0)
        return softmax_yes(Uyes, Uno, lam)

    elif mid == 8:
        # m8: m7 + lapse rate
        gama, envy, guilt, lam, omega, kapa, etak, etaa, minp, maxp = x[0],x[1],x[2],x[3],x[4],x[5],x[6],x[7],x[8],x[9]
        disad = max(x1s-x3s,0)+max(x2s-x3s,0)
        ad    = max(x3s-x1s,0)+max(x3s-x2s,0)
        inqua = max(x1s-x2s,0)
        EP    = x1s + x2s
        RP    = max(x2s-x1s,0)
        IIk   = 2.0 / (1.0 + np.exp(np.clip(etak*(cost/50), -500, 500)))
        IIa   = 2.0 / (1.0 + np.exp(np.clip(etaa*(cost/50), -500, 500)))
        Uyes = x3s - envy*disad - guilt*ad - gama*inqua*IIa + omega*EP + kapa*RP
        Uno  = x3  - envy*(max(x1-x3,0)+max(x2-x3,0)) \
                   - guilt*(max(x3-x1,0)+max(x3-x2,0)) \
                   - gama*(max(x1-x2,0)*IIk) + omega*(x1+x2) + kapa*max(x2-x1,0)
        pact = softmax_yes(Uyes, Uno, lam)
        pact = minp + (1 - maxp - minp) * pact
        return np.clip(pact, 1e-6, 1-1e-6)

    elif mid == 9:
        # m9: heuristic
        lam, bs, bi, bc, br, maxp, minp = x[0],x[1],x[2],x[3],x[4],x[5],x[6]
        block_sign = 1 if block == 1 else -1
        inequa = max(x1 - x2, 0)
        p = 1.0 / (1.0 + np.exp(np.clip(lam*(bs*block_sign + bi*inequa + bc*cost + br*ratio), -500, 500)))
        p = minp + (1 - maxp - minp) * p
        return np.clip(p, 1e-6, 1-1e-6)


# ─── Negative log-likelihood ──────────────────────────────────────────────────

def neg_loglik_logprobs(params, model_id, trials):
    """
    For logprobs data: p_yes is continuous ∈ [0,1].
    lik contribution = p_yes * log(p_model) + (1-p_yes) * log(1-p_model)
    """
    lik = 0.0
    for row in trials:
        row["model_id"] = model_id
        p_model = compute_p_model(list(params), row)
        p_model = np.clip(p_model, 1e-9, 1-1e-9)
        p_yes   = row["p_yes"]
        lik += p_yes * np.log(p_model) + (1 - p_yes) * np.log(1 - p_model)
    return -lik


def neg_loglik_binary(params, model_id, trials):
    """
    For repeated-sampling data: intervene ∈ {0,1}.
    """
    lik = 0.0
    for row in trials:
        row["model_id"] = model_id
        p_model = compute_p_model(list(params), row)
        p_model = np.clip(p_model, 1e-9, 1-1e-9)
        action  = row["p_yes"]
        lik += action * np.log(p_model) + (1 - action) * np.log(1 - p_model)
    return -lik


# ─── Model specs ──────────────────────────────────────────────────────────────

MODEL_SPECS = {
    1: {"n_params": 1, "bounds": [(1e-6, 1-1e-6)],
        "x0_ranges": [(0.1, 0.9)]},
    2: {"n_params": 1, "bounds": [(0.001, 10)],
        "x0_ranges": [(0.01, 5)]},
    3: {"n_params": 3, "bounds": [(-5,5),(-5,5),(0.001,10)],
        "x0_ranges": [(-2,2),(-2,2),(0.01,5)]},
    4: {"n_params": 4, "bounds": [(-5,5),(-5,5),(-5,5),(0.001,10)],
        "x0_ranges": [(-2,2),(-2,2),(-2,2),(0.01,5)]},
    5: {"n_params": 5, "bounds": [(-5,5),(-5,5),(-5,5),(0.001,10),(-5,5)],
        "x0_ranges": [(-2,2),(-2,2),(-2,2),(0.01,5),(-2,2)]},
    6: {"n_params": 6, "bounds": [(-5,5),(-5,5),(-5,5),(0.001,10),(-5,5),(-5,5)],
        "x0_ranges": [(-2,2),(-2,2),(-2,2),(0.01,5),(-2,2),(-2,2)]},
    7: {"n_params": 8, "bounds": [(-5,5),(-5,5),(-5,5),(0.001,10),(-5,5),(-5,5),(-10,10),(-10,10)],
        "x0_ranges": [(-2,2),(-2,2),(-2,2),(0.01,5),(-2,2),(-2,2),(-5,5),(-5,5)]},
    8: {"n_params": 10,"bounds": [(-5,5),(-5,5),(-5,5),(0.001,10),(-5,5),(-5,5),(-10,10),(-10,10),(0,0.49),(0,0.49)],
        "x0_ranges": [(-2,2),(-2,2),(-2,2),(0.01,5),(-2,2),(-2,2),(-5,5),(-5,5),(0,0.3),(0,0.3)]},
    9: {"n_params": 7, "bounds": [(0.001,10),(-5,5),(-5,5),(-5,5),(-5,5),(0,0.49),(0,0.49)],
        "x0_ranges": [(0.01,5),(-2,2),(-2,2),(-2,2),(-2,2),(0,0.3),(0,0.3)]},
}

PARAM_NAMES = {
    1: ["p"],
    2: ["lambda"],
    3: ["envy","guilt","lambda"],
    4: ["gamma","envy","guilt","lambda"],
    5: ["gamma","envy","guilt","lambda","omega"],
    6: ["gamma","envy","guilt","lambda","omega","kappa"],
    7: ["gamma","envy","guilt","lambda","omega","kappa","etak","etaa"],
    8: ["gamma","envy","guilt","lambda","omega","kappa","etak","etaa","minp","maxp"],
    9: ["lambda","bs","bi","bc","br","maxp","minp"],
}


# ─── Fit one model to one subject ─────────────────────────────────────────────

def fit_model(model_id, trials, data_type="logprobs", n_restarts=20, seed=42):
    """
    Fit one model to one subject's trials.
    data_type: "logprobs" or "binary"
    Returns dict with params, negloglik, BIC, AIC.
    """
    rng    = np.random.default_rng(seed)
    spec   = MODEL_SPECS[model_id]
    bounds = spec["bounds"]
    ranges = spec["x0_ranges"]
    k      = spec["n_params"]
    n      = len(trials)

    nll_fn = neg_loglik_logprobs if data_type == "logprobs" else neg_loglik_binary

    best_nll = np.inf
    best_x   = None

    for _ in range(n_restarts):
        x0 = [rng.uniform(lo, hi) for (lo, hi) in ranges]
        try:
            res = minimize(
                nll_fn, x0,
                args=(model_id, trials),
                method="L-BFGS-B",
                bounds=bounds,
                options={"maxiter": 2000, "ftol": 1e-10},
            )
            if res.fun < best_nll:
                best_nll = res.fun
                best_x   = res.x
        except Exception:
            pass

    if best_x is None:
        best_nll = np.nan
        best_x   = [np.nan] * k

    bic = 2 * best_nll + k * np.log(n)
    aic = 2 * best_nll + 2 * k

    result = {"model_id": model_id, "negloglik": best_nll, "BIC": bic, "AIC": aic, "n_trials": n}
    for name, val in zip(PARAM_NAMES[model_id], best_x):
        result[name] = val
    return result


# ─── Load and prepare data ────────────────────────────────────────────────────

def load_logprobs(csv_path):
    df = pd.read_csv(csv_path)
    df["block"] = df["block_type"].map({"punish": 1, "help": 2})
    return df


def load_binary(csv_path):
    """For repeated-sampling data: aggregate per condition → mean p_yes."""
    df = pd.read_csv(csv_path)
    df["block"] = df["block_type"].map({"punish": 1, "help": 2})
    # Average across repetitions per condition
    agg = (df.groupby(["country","gender","cond_id","block_type","block","x1","x2","cost","ratio"])
             ["intervene"].mean().reset_index().rename(columns={"intervene":"p_yes"}))
    return agg


def df_to_trials(df):
    """Convert DataFrame rows to list of dicts (fast iteration)."""
    return df[["block","ratio","x1","x2","cost","p_yes"]].to_dict("records")


# ─── Main fitting loop ────────────────────────────────────────────────────────

def run_fitting(csv_path, data_type="logprobs", models=range(1, 10),
                out_dir=None, label=None):
    """
    Fit all models × all (country, gender) combinations.

    data_type: "logprobs" or "binary"
    """
    print(f"\nLoading: {csv_path}")
    if data_type == "logprobs":
        df = load_logprobs(csv_path)
    else:
        df = load_binary(csv_path)

    if label is None:
        label = Path(csv_path).stem
    if out_dir is None:
        out_dir = Path(csv_path).parent

    results = []
    groups  = df.groupby(["country", "gender"])
    total   = len(groups) * len(list(models))
    done    = 0

    for (country, gender), grp in groups:
        trials = df_to_trials(grp)
        for mid in models:
            res = fit_model(mid, trials, data_type=data_type)
            res.update({"ai_model": label, "country": country, "gender": gender})
            results.append(res)
            done += 1
            print(f"  [{done}/{total}] {country} {gender} m{mid} — "
                  f"NLL={res['negloglik']:.3f}  BIC={res['BIC']:.3f}")

    out_df   = pd.DataFrame(results)
    out_path = out_dir / f"fit_{label}.csv"
    out_df.to_csv(out_path, index=False)
    print(f"\nSaved → {out_path}")
    return out_df


# ─── Entry point ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    DATA_DIR = Path("/Users/wuxiaoyan/Desktop/TPP_culture_AI/ai_study_data")
    OUT_DIR  = DATA_DIR / "ModelFitting" / "results"
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # ── GPT-4o-mini (logprobs, complete) ──────────────────────────────────────
    gpt_file = DATA_DIR / "gpt4o_mini_logprobs_20260610_131216.csv"
    run_fitting(gpt_file, data_type="logprobs", label="gpt4o_mini", out_dir=OUT_DIR)

    # ── Llama 3.3-70B (logprobs, add when complete) ───────────────────────────
    # llama3_files = sorted(DATA_DIR.glob("llama3_70b_logprobs_*.csv"))
    # if llama3_files:
    #     run_fitting(llama3_files[-1], data_type="logprobs", label="llama3_70b", out_dir=OUT_DIR)

    # ── Llama 4 Scout (logprobs, add when complete) ───────────────────────────
    # llama4_files = sorted(DATA_DIR.glob("llama4_scout_logprobs_*.csv"))
    # if llama4_files:
    #     run_fitting(llama4_files[-1], data_type="logprobs", label="llama4_scout", out_dir=OUT_DIR)

    # ── Claude Haiku (repeated sampling → aggregate to p_yes) ─────────────────
    # claude_files = sorted(DATA_DIR.glob("claude_haiku_r10_*.csv"))
    # if claude_files:
    #     run_fitting(claude_files[-1], data_type="binary", label="claude_haiku", out_dir=OUT_DIR)

    # ── Gemini 2.0 Flash (repeated sampling → aggregate to p_yes) ─────────────
    # gemini_files = sorted(DATA_DIR.glob("gemini_flash_r10_*.csv"))
    # if gemini_files:
    #     run_fitting(gemini_files[-1], data_type="binary", label="gemini_flash", out_dir=OUT_DIR)
