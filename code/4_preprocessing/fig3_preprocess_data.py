# =============================================================================
# Author : Xiaoyan Wu
# Date   : June 2026
# Description: Preprocessing — computes RSA/cultural similarity matrices and generates fig3_data.mat for Figure 3
# =============================================================================

"""
Figure3 — preprocess_data.py
Generates data/fig3_data.mat for the three MATLAB panel scripts.

Run once:
    cd /Users/wuxiaoyan/Desktop/TPP_culture_AI/Figure3
    python preprocess_data.py

Output: data/fig3_data.mat containing:
  Panel A (human country variation):
    h_country_names    — cell (8×1) sorted by mean descending
    h_country_means    — (8×1) mean intervention rate
    h_country_se       — (8×1) SE of participant-level means
    h_country_n        — (8×1) number of participants
    h_participant_mat  — (8 × max_n) participant-level means, NaN-padded

  Panel B (F-statistic):
    all_labels         — cell (n_models+1) × 1, first = 'Human'
    all_Fstat          — (n_models+1) × 1 F-values
    all_eta2           — (n_models+1) × 1 eta² values
    all_sig            — (n_models+1) × 1 significance string ('***','**','*','ns')
    F_critical         — scalar (α=.05, df_between=7)
    h_Fstat            — scalar human F
    h_eta2             — scalar human η²

  Panel C (Spearman rank correlation):
    rho_labels         — cell (n_models×1) model names
    rho_vals           — (n_models×1) Spearman ρ
    rho_pvals          — (n_models×1) p-values
    rho_sig            — cell (n_models×1) significance strings
    h_country_rank     — (8×1) human country rank (1=highest)
"""

import numpy as np
import pandas as pd
import scipy.io as sio
import scipy.stats as stats
from pathlib import Path

BASE = Path('/Users/wuxiaoyan/Desktop/TPP_culture_AI')
OUT  = Path(__file__).parent / 'data'
OUT.mkdir(exist_ok=True)

# ── 1. Load human data ────────────────────────────────────────────────────────
human = pd.read_csv(BASE / 'human_data.csv')
human['action'] = pd.to_numeric(human['action'], errors='coerce')
human = human.dropna(subset=['action', 'country'])

# Participant-level means (aggregate trials → 1 value per participant)
part_means = human.groupby(['subid', 'country'])['action'].mean().reset_index()
part_means.columns = ['subid', 'country', 'mean_action']

# Country summary sorted by mean descending
country_summary = (part_means.groupby('country')['mean_action']
                   .agg(['mean', 'sem', 'count'])
                   .sort_values('mean', ascending=False))

countries_sorted = country_summary.index.tolist()
n_ctry = len(countries_sorted)
h_country_means = country_summary['mean'].values
h_country_se    = country_summary['sem'].values
h_country_n     = country_summary['count'].values.astype(float)

# Build NaN-padded participant matrix (n_ctry × max_n)
max_n = int(h_country_n.max())
h_participant_mat = np.full((n_ctry, max_n), np.nan)
for i, ctry in enumerate(countries_sorted):
    vals = part_means.loc[part_means['country'] == ctry, 'mean_action'].values
    h_participant_mat[i, :len(vals)] = vals

# ── 2. One-way ANOVA helper ───────────────────────────────────────────────────
def compute_anova(df_sub, group_col='country', val_col='action'):
    """Returns (F, eta2, p_value)."""
    groups = [g[val_col].values for _, g in df_sub.groupby(group_col)]
    F, p = stats.f_oneway(*groups)
    grand  = df_sub[val_col].mean()
    ss_b   = sum(len(g) * (g.mean() - grand)**2 for g in groups)
    ss_t   = ((df_sub[val_col] - grand)**2).sum()
    eta2   = ss_b / ss_t if ss_t > 0 else 0.0
    return float(F), float(eta2), float(p)

def sig_stars(p):
    if p < 0.001: return '***'
    if p < 0.01:  return '**'
    if p < 0.05:  return '*'
    return 'ns'

h_F, h_eta2, h_p = compute_anova(human, 'country', 'action')
print(f'Human: F={h_F:.1f}, η²={h_eta2:.4f}, p={h_p:.2e}')

# F-critical: α=.05, df1=7 (8 countries-1), df2 very large
df1 = n_ctry - 1
# approximate df2 from human data
df2_human = len(human) - n_ctry
F_critical = float(stats.f.ppf(0.95, df1, df2_human))
print(f'F_critical (df1={df1}, df2={df2_human}) = {F_critical:.3f}')

# ── 3. Load AI data ───────────────────────────────────────────────────────────
amat   = sio.loadmat(str(BASE / 'ModelFitting/data_ai_all.mat'),
                     squeeze_me=True, struct_as_record=False)
data_ai = amat['data_ai']

ai_rows = []
for s in data_ai:
    try:
        model   = str(s.model).strip()
        method  = str(s.Method).strip()
        country = str(s.CountryOfBirth).strip()
        actions = pd.to_numeric(np.array(s.action).flatten(), errors='coerce')
        n = len(actions)
        ai_rows.append(pd.DataFrame({
            'action':  actions,
            'country': [country] * n,
            'label':   [f'{model}_{method}'] * n,
        }))
    except Exception:
        continue

ai = pd.concat(ai_rows, ignore_index=True).dropna()

# All unique model labels (sorted) — include both prob and logprobs
all_labels_ai = sorted(ai['label'].unique())
print('AI models (all formats):', all_labels_ai)

# ── 4. Compute F, η² for all AI models ────────────────────────────────────────
ai_F_list    = []
ai_eta2_list = []
ai_sig_list  = []

for lbl in all_labels_ai:
    sub = ai[ai['label'] == lbl]
    if sub.empty or sub['country'].nunique() < 2:
        ai_F_list.append(np.nan)
        ai_eta2_list.append(np.nan)
        ai_sig_list.append('ns')
        continue
    F_ai, eta2_ai, p_ai = compute_anova(sub, 'country', 'action')
    ai_F_list.append(F_ai)
    ai_eta2_list.append(eta2_ai)
    ai_sig_list.append(sig_stars(p_ai))
    print(f'  {lbl}: F={F_ai:.2f}, η²={eta2_ai:.4f}, p={p_ai:.4f} {sig_stars(p_ai)}')

# Full arrays for Panel B: Human first, then AI sorted by F descending
ai_F_arr    = np.array(ai_F_list)
ai_eta2_arr = np.array(ai_eta2_list)
sort_idx    = np.argsort(ai_F_arr)[::-1]  # sort by F descending

all_labels_B  = ['Human'] + [all_labels_ai[i] for i in sort_idx]
all_Fstat_B   = np.array([h_F]    + [ai_F_list[i]    for i in sort_idx])
all_eta2_B    = np.array([h_eta2] + [ai_eta2_list[i] for i in sort_idx])
all_sig_B     = ['***'] + [ai_sig_list[i] for i in sort_idx]

# ── 5. Spearman rank correlation (Panel C) ────────────────────────────────────
h_ctry_mean = (part_means.groupby('country')['mean_action'].mean()
               .reindex(countries_sorted))

rho_labels = []
rho_vals   = []
rho_pvals  = []
rho_sig    = []

for lbl in all_labels_ai:
    sub = ai[ai['label'] == lbl]
    if sub.empty:
        continue
    ai_ctry = sub.groupby('country')['action'].mean().reindex(countries_sorted)
    valid = ~ai_ctry.isna()
    if valid.sum() < 4:
        continue
    rho, p = stats.spearmanr(h_ctry_mean[valid], ai_ctry[valid])
    rho_labels.append(lbl)
    rho_vals.append(float(rho))
    rho_pvals.append(float(p))
    rho_sig.append(sig_stars(p))
    print(f'  ρ {lbl}: rho={rho:.3f}, p={p:.4f} {sig_stars(p)}')

# Sort Panel C by rho descending
rho_sort = np.argsort(rho_vals)[::-1]
rho_labels_s = [rho_labels[i] for i in rho_sort]
rho_vals_s   = np.array([rho_vals[i]  for i in rho_sort])
rho_pvals_s  = np.array([rho_pvals[i] for i in rho_sort])
rho_sig_s    = [rho_sig[i]  for i in rho_sort]

# Human country ranks (1 = highest intervention)
h_country_rank = np.argsort(np.argsort(-h_country_means)) + 1  # 1-based

# ── 6. Pairwise country–condition differences (Panel D) ───────────────────────
# For each of 28 country pairs × 100 conditions = 2800 signed differences
# Conditions: block_type × ratio × cost × x2 (offer), same order as Figure2 heatmap
print('\n=== Computing pairwise country differences (Panel D) ===')

from itertools import combinations
from scipy.stats import pearsonr as _pearsonr

COUNTRIES_ALPHA = sorted(human['country'].unique())   # 8 alphabetical
N_CTRY = len(COUNTRIES_ALPHA)

# Condition axes (human data)
BLOCKS_H = sorted(human['block_type'].unique())       # ['help', 'punish']
RATIOS   = sorted(human['ratio'].unique())             # [1.5, 3.0]
COSTS    = sorted(human['cost'].unique())              # [10,20,30,40,50]
# x2 has ±2 noise around multiples of 10 → round to nearest 10
human['x2_bin'] = (human['x2'] / 10).round().astype(int) * 10
OFFERS   = sorted(human['x2_bin'].unique())            # [10,20,30,40,50]
N_COND   = len(BLOCKS_H) * len(RATIOS) * len(COSTS) * len(OFFERS)  # 100

print(f'Condition axes: {len(BLOCKS_H)} blocks × {len(RATIOS)} ratios × {len(COSTS)} costs × {len(OFFERS)} offers = {N_COND}')

# Human: (N_CTRY × N_COND) mean intervention per country per condition
h_grp = (human.groupby(['country', 'block_type', 'ratio', 'cost', 'x2_bin'])['action']
               .mean().reset_index())

h_cond_country = np.full((N_CTRY, N_COND), np.nan)
cond_idx = 0
for blk in BLOCKS_H:
    for rat in RATIOS:
        for cost in COSTS:
            for off in OFFERS:
                mask = ((h_grp['block_type'] == blk) &
                        (np.abs(h_grp['ratio'] - rat) < 0.01) &
                        (h_grp['cost'] == cost) &
                        (h_grp['x2_bin'] == off))
                for ci, ctry in enumerate(COUNTRIES_ALPHA):
                    row = h_grp[mask & (h_grp['country'] == ctry)]
                    if not row.empty:
                        h_cond_country[ci, cond_idx] = row['action'].iloc[0]
                cond_idx += 1

# AI: rebuild full dataframe with condition labels from data_ai_all.mat
# block: 1=punish(REDUCE), 2=help(INCREASE); ratio; victim=x2 offer
# BLOCKS_H sorted → ['help','punish'], corresponding AI block = [2, 1]
BLOCK_H2AI = {'help': 2, 'punish': 1}

ai_cond_rows = []
for s in data_ai:
    try:
        model_name = str(s.model).strip()
        method     = str(s.Method).strip()
        country    = str(s.CountryOfBirth).strip()
        label      = f'{model_name}_{method}'
        block_arr  = np.array(s.block).flatten().astype(float)
        ratio_arr  = np.array(s.ratio).flatten().astype(float)
        cost_arr   = np.array(s.cost).flatten().astype(float)
        # victim = x2 (beneficiary offer); round to nearest 10
        victim_arr = np.array(s.victim).flatten().astype(float)
        x2_bin     = np.round(victim_arr / 10).astype(int) * 10
        action_arr = pd.to_numeric(np.array(s.action).flatten(), errors='coerce')
        n = len(action_arr)
        ai_cond_rows.append(pd.DataFrame({
            'label':   [label]   * n,
            'country': [country] * n,
            'block':   block_arr,
            'ratio':   ratio_arr,
            'cost':    cost_arr,
            'x2':      x2_bin,
            'action':  action_arr,
        }))
    except Exception as e:
        print(f'  Warning: {e}')
        continue

ai_full = pd.concat(ai_cond_rows, ignore_index=True).dropna(subset=['action'])
ai_labels_D = sorted(ai_full['label'].unique())
n_ai_D = len(ai_labels_D)
print(f'AI models for Panel D: {n_ai_D}')

ai_grp = (ai_full.groupby(['label', 'country', 'block', 'ratio', 'cost', 'x2'])['action']
                 .mean().reset_index())

ai_cond_country = np.full((n_ai_D, N_CTRY, N_COND), np.nan)
for mi, lbl in enumerate(ai_labels_D):
    sub = ai_grp[ai_grp['label'] == lbl]
    cond_idx = 0
    for blk_h in BLOCKS_H:
        blk_ai = float(BLOCK_H2AI[blk_h])
        for rat in RATIOS:
            for cost in COSTS:
                for off in OFFERS:
                    mask = ((sub['block'] == blk_ai) &
                            (np.abs(sub['ratio'] - rat) < 0.01) &
                            (sub['cost'] == cost) &
                            (sub['x2'] == off))
                    for ci, ctry in enumerate(COUNTRIES_ALPHA):
                        row = sub[mask & (sub['country'] == ctry)]
                        if not row.empty:
                            ai_cond_country[mi, ci, cond_idx] = row['action'].iloc[0]
                    cond_idx += 1

# Pairwise differences (28 pairs × 100 conditions = 2800 values)
pairs = list(combinations(range(N_CTRY), 2))  # 28 pairs
n_pairs = len(pairs)

pair_delta_human = np.full(n_pairs * N_COND, np.nan)
for pi, (i, j) in enumerate(pairs):
    pair_delta_human[pi*N_COND:(pi+1)*N_COND] = (
        h_cond_country[i, :] - h_cond_country[j, :])

pair_delta_ai = np.full((n_ai_D, n_pairs * N_COND), np.nan)
for mi in range(n_ai_D):
    for pi, (i, j) in enumerate(pairs):
        pair_delta_ai[mi, pi*N_COND:(pi+1)*N_COND] = (
            ai_cond_country[mi, i, :] - ai_cond_country[mi, j, :])

# Pearson r for each AI model
delta_r_vals = np.full(n_ai_D, np.nan)
delta_p_vals = np.full(n_ai_D, np.nan)
for mi in range(n_ai_D):
    hx = pair_delta_human
    ay = pair_delta_ai[mi]
    valid = ~np.isnan(hx) & ~np.isnan(ay)
    if valid.sum() > 10:
        r, p = _pearsonr(hx[valid], ay[valid])
        delta_r_vals[mi] = r
        delta_p_vals[mi] = p
    print(f'  {ai_labels_D[mi]}: r={delta_r_vals[mi]:.3f}')

# ── RSA: 8×8 country similarity matrices + Mantel test ───────────────────────
print('\n=== RSA: Country similarity matrices + Mantel test ===')
from scipy.stats import spearmanr as _spearmanr

# Human RDM: 8×8 Pearson r between country 100-condition vectors
h_rdm = np.corrcoef(h_cond_country)   # (8×8)

# AI RDMs
ai_rdm = np.full((n_ai_D, N_CTRY, N_CTRY), np.nan)
for mi in range(n_ai_D):
    mat = ai_cond_country[mi]          # (8×100)
    # only compute where no full-NaN rows
    valid_rows = ~np.all(np.isnan(mat), axis=1)
    sub = mat.copy()
    sub[~valid_rows, :] = np.nanmean(mat, axis=0)   # fill missing rows temporarily
    ai_rdm[mi] = np.corrcoef(sub)
    ai_rdm[mi][~valid_rows, :] = np.nan
    ai_rdm[mi][:, ~valid_rows] = np.nan

# Lower triangle indices (28 unique pairs, k=-1 excludes diagonal)
tril_idx = np.tril_indices(N_CTRY, k=-1)
h_lower  = h_rdm[tril_idx]            # (28,)

# Mantel permutation test (permute country labels in AI matrix)
N_PERM_M  = 5000
rng_m     = np.random.default_rng(99)

mantel_r    = np.full(n_ai_D, np.nan)
mantel_p    = np.full(n_ai_D, np.nan)
mantel_null = np.full((n_ai_D, N_PERM_M), np.nan)

print(f'\n{"Model":<32}  {"Mantel ρ":>9}  {"perm_p":>8}  {"percentile":>10}')
print('-' * 65)
for mi in range(n_ai_D):
    ai_lower = ai_rdm[mi][tril_idx]   # (28,)
    valid = ~np.isnan(h_lower) & ~np.isnan(ai_lower)
    if valid.sum() < 10:
        continue
    rho_obs, _ = _spearmanr(h_lower[valid], ai_lower[valid])
    mantel_r[mi] = rho_obs

    null_rs = np.empty(N_PERM_M)
    ai_mat  = ai_rdm[mi].copy()
    for k in range(N_PERM_M):
        perm        = rng_m.permutation(N_CTRY)
        ai_perm     = ai_mat[np.ix_(perm, perm)]
        ai_lp       = ai_perm[tril_idx]
        vp          = ~np.isnan(h_lower) & ~np.isnan(ai_lp)
        null_rs[k]  = _spearmanr(h_lower[vp], ai_lp[vp])[0]

    mantel_null[mi] = null_rs
    mantel_p[mi]    = np.mean(np.abs(null_rs) >= np.abs(rho_obs))
    pct             = np.mean(null_rs <= rho_obs) * 100
    p_str = f'<{1/N_PERM_M:.4f}' if mantel_p[mi] == 0 else f'{mantel_p[mi]:.4f}'
    print(f'  {ai_labels_D[mi]:<30}  {rho_obs:>9.3f}  {p_str:>8}  {pct:>9.1f}%')

# ── Permutation test (5000 shuffles) for all models ──────────────────────────
print('\n=== Permutation test (n_perm=5000) ===')
N_PERM = 5000
rng = np.random.default_rng(42)

perm_null_r   = np.full((n_ai_D, N_PERM), np.nan)   # null distribution per model
delta_p_perm  = np.full(n_ai_D, np.nan)             # permutation p-value

hx_all = pair_delta_human.copy()

for mi in range(n_ai_D):
    ay = pair_delta_ai[mi]
    valid = ~np.isnan(hx_all) & ~np.isnan(ay)
    hx_v = hx_all[valid]
    ay_v = ay[valid]
    obs_r = delta_r_vals[mi]

    null_rs = np.empty(N_PERM)
    for k in range(N_PERM):
        ay_shuf = rng.permutation(ay_v)
        null_rs[k] = np.corrcoef(hx_v, ay_shuf)[0, 1]

    perm_null_r[mi, :] = null_rs
    # two-tailed p-value
    delta_p_perm[mi] = np.mean(np.abs(null_rs) >= np.abs(obs_r))

    pct = np.mean(null_rs <= obs_r) * 100
    print(f'  {ai_labels_D[mi]}: r={obs_r:.3f}  perm_p={delta_p_perm[mi]:.4f}  '
          f'percentile={pct:.1f}%  '
          f'null 95% CI [{np.percentile(null_rs,2.5):.3f}, {np.percentile(null_rs,97.5):.3f}]')

# ── 7. Save .mat ──────────────────────────────────────────────────────────────
out_path = str(OUT / 'fig3_data.mat')

# MATLAB cell arrays via object array
def cell(lst):
    arr = np.empty(len(lst), dtype=object)
    for i, v in enumerate(lst): arr[i] = v
    return arr

sio.savemat(out_path, {
    # Panel A
    'h_country_names':   cell(countries_sorted),
    'h_country_means':   h_country_means,
    'h_country_se':      h_country_se,
    'h_country_n':       h_country_n,
    'h_participant_mat': h_participant_mat,   # (8 × max_n) NaN-padded

    # Panel B
    'all_labels_B':  cell(all_labels_B),
    'all_Fstat_B':   all_Fstat_B,
    'all_eta2_B':    all_eta2_B,
    'all_sig_B':     cell(all_sig_B),
    'F_critical':    np.array([F_critical]),
    'h_Fstat':       np.array([h_F]),
    'h_eta2':        np.array([h_eta2]),

    # Panel C
    'rho_labels':    cell(rho_labels_s),
    'rho_vals':      rho_vals_s,
    'rho_pvals':     rho_pvals_s,
    'rho_sig':       cell(rho_sig_s),
    'h_country_rank': h_country_rank,

    # Panel E (RSA: country similarity matrices + Mantel test)
    'rdm_country_names': cell(COUNTRIES_ALPHA),   # alphabetical
    'h_rdm':             h_rdm,                   # (8×8)
    'ai_rdm':            ai_rdm,                  # (n_ai×8×8)
    'mantel_r':          mantel_r,                # (n_ai,) Spearman ρ
    'mantel_p':          mantel_p,                # (n_ai,) permutation p
    'mantel_null':       mantel_null,             # (n_ai×5000) null distributions

    # Panel D (pairwise country-condition differences)
    'delta_ai_labels':  cell(ai_labels_D),
    'pair_delta_human': pair_delta_human,    # (2800,)
    'pair_delta_ai':    pair_delta_ai,       # (n_ai × 2800)
    'delta_r_vals':     delta_r_vals,        # (n_ai,)
    'delta_p_vals':     delta_p_vals,        # (n_ai,) parametric
    'delta_p_perm':     delta_p_perm,        # (n_ai,) permutation p
    'perm_null_r':      perm_null_r,         # (n_ai × 5000) null distributions
    'n_ai_delta':       np.array([n_ai_D]),
}, do_compression=True)

print(f'\nSaved: {out_path}')
