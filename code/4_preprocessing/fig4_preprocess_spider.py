# =============================================================================
# Author : Xiaoyan Wu
# Date   : June 2026
# Description: Preprocessing — generates fig5_data.mat (parameter distributions) for Figure 4/5
# =============================================================================

"""
Figure5 — preprocess_data.py
Spider plot: Human M8 parameters as reference, AI models projected onto human axis range.

Normalization logic:
  - For each parameter k: axis range = [human_mean[k] - 2*human_SD[k],
                                        human_mean[k] + 2*human_SD[k]]
  - Normalized value = (x - (human_mean - 2*SD)) / (4*SD)
    → Human mean  maps to 0.5
    → Human mean ± 2SD maps to 1.0 / 0.0
    → AI values outside human 95% range will be <0 or >1

Three AI models (all selected M8 Full as best cognitive model):
  - GPT-4o-mini [logprobs]  — 16/16 chose M8
  - GPT-5.4 [prob]          — 14/16 chose M8
  - Gemini-3.5-Flash [prob] — 14/16 chose M8

Output: data/fig5_data.mat
"""

import numpy as np
import scipy.io as sio
import pandas as pd
from pathlib import Path

BASE  = Path('/Users/wuxiaoyan/Desktop/TPP_culture_AI')
MFIT  = BASE / 'ModelFitting'
OUT   = Path(__file__).parent / 'data'
OUT.mkdir(exist_ok=True)

PARAM_NAMES   = ['gama', 'envy', 'guilt', 'lamda', 'oumiga', 'kapa', 'etak', 'etaa']
PARAM_DISPLAY = [r'\gamma (risk)', 'envy', 'guilt', r'\lambda (loss)',
                 r'\omega (SCI)',  r'\kappa (RP)', r'\eta_k', r'\eta_a']

# ── 1. Load raw data ──────────────────────────────────────────────────────
hr    = sio.loadmat(str(MFIT/'results_human_aligned.mat'),
                    squeeze_me=True, struct_as_record=False)['results_aligned']
ar    = sio.loadmat(str(MFIT/'results_ai.mat'),
                    squeeze_me=True, struct_as_record=False)['results_ai']
adata = sio.loadmat(str(MFIT/'data_ai_all.mat'),
                    squeeze_me=True, struct_as_record=False)['data_ai']

meta = pd.DataFrame({
    'model':  [str(s.model)  for s in adata],
    'method': [str(s.Method) for s in adata],
})

# ── 2. Extract M8 parameters (all subjects) ───────────────────────────────
h_x   = np.array(hr[7].x)[:, :8]                                          # (875, 8)
ai_x8 = np.array([np.array(ar[7].x[i]).flatten()[:8] for i in range(192)])  # (192, 8)

# ── 3. Human reference statistics ────────────────────────────────────────
h_mean = h_x.mean(axis=0)
h_std  = h_x.std(axis=0)
h_se   = h_std / np.sqrt(len(h_x))

# Axis range: [mean - 2SD,  mean + 2SD]
ax_lo  = h_mean - 2 * h_std
ax_hi  = h_mean + 2 * h_std
ax_rng = 4 * h_std   # = ax_hi - ax_lo
ax_rng[ax_rng < 1e-9] = 1.0

print("=== Human M8 reference (mean ± 2SD) ===")
for i, p in enumerate(PARAM_NAMES):
    print(f"  {p:<8}: mean={h_mean[i]:6.3f}  SD={h_std[i]:6.3f}  "
          f"axis=[{ax_lo[i]:7.3f}, {ax_hi[i]:7.3f}]")

# Normalization function: 0 = mean-2SD, 0.5 = mean, 1 = mean+2SD
def norm(x):
    return (np.array(x) - ax_lo) / ax_rng

def mean_se_norm(idx):
    raw_m = ai_x8[idx].mean(axis=0)
    raw_se = ai_x8[idx].std(axis=0) / np.sqrt(len(idx))
    return norm(raw_m), raw_se / ax_rng   # SE scaled by same factor

# Human normalized (should give mean≈0.5, ±1SD → ±0.25)
h_norm_mean = norm(h_mean)
h_norm_se   = h_se / ax_rng

print("\n=== Human normalized (expect ~0.5 per axis) ===")
for i, p in enumerate(PARAM_NAMES):
    print(f"  {p:<8}: {h_norm_mean[i]:.3f} ± {h_norm_se[i]:.3f}")

# ── 4. AI models ──────────────────────────────────────────────────────────
AI_MODELS = [
    ('GPT-4o-mini', 'logprobs', 'GPT-4o-mini [lp]'),
    ('GPT-5.4',     'prob',     'GPT-5.4'),
    ('Gemini-3.5-Flash', 'prob', 'Gemini-3.5-Flash'),
]

ai_means = {}
ai_ses   = {}
for model, method, label in AI_MODELS:
    idx = meta.index[(meta['model'] == model) & (meta['method'] == method)].tolist()
    m, se = mean_se_norm(idx)
    ai_means[label] = m
    ai_ses[label]   = se
    print(f"\n=== {label} (N={len(idx)}) normalized ===")
    for i, p in enumerate(PARAM_NAMES):
        flag = ' *** OUTSIDE HUMAN RANGE' if (m[i] < 0 or m[i] > 1) else ''
        print(f"  {p:<8}: {m[i]:6.3f} ± {se[i]:.3f}{flag}")

# ── 5. Save ───────────────────────────────────────────────────────────────
def cell(lst):
    arr = np.empty(len(lst), dtype=object)
    for i, v in enumerate(lst): arr[i] = v
    return arr

sio.savemat(str(OUT / 'fig5_data.mat'), {
    'param_names':    cell(PARAM_NAMES),
    'param_display':  cell(PARAM_DISPLAY),
    'ax_lo':          ax_lo,
    'ax_hi':          ax_hi,
    'h_mean_n':       h_norm_mean,
    'h_se_n':         h_norm_se,
    'h_std_n':        h_std / ax_rng,   # 1SD in normalized units (= 0.25)
    'mini_lp_mean_n': ai_means['GPT-4o-mini [lp]'],
    'mini_lp_se_n':   ai_ses['GPT-4o-mini [lp]'],
    'gpt54_mean_n':   ai_means['GPT-5.4'],
    'gpt54_se_n':     ai_ses['GPT-5.4'],
    'gem35_mean_n':   ai_means['Gemini-3.5-Flash'],
    'gem35_se_n':     ai_ses['Gemini-3.5-Flash'],
}, do_compression=True)
print(f'\nSaved: {OUT}/fig5_data.mat')
