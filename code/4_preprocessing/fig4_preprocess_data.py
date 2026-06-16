# =============================================================================
# Author : Xiaoyan Wu
# Date   : June 2026
# Description: Preprocessing — generates fig4_data.mat (model complexity) for Figure 4
# =============================================================================

"""
Figure4 — preprocess_data.py
Generates data/fig4_data.mat for MATLAB panel scripts.

Run once:
    cd /Users/wuxiaoyan/Desktop/TPP_culture_AI/Figure4
    python preprocess_data.py

Output: data/fig4_data.mat containing:

  Panel A (model complexity stacked bar):
    row_names        — cell (n_rows × 1): 'Human' + 8 AI model names
    best_counts_mat  — (n_rows × 8): count of subjects choosing M1–M8
    n_subjects       — (n_rows × 1): total subjects per row
    is_human         — (n_rows × 1): logical, 1 for human row
    model_labels     — cell (8 × 1): 'M1\nBaseline' … 'M8\nFull'

  Panel B (radar — Human, GPT-5.4, Gemini-3.5):
    param_names      — cell (8 × 1): parameter names
    h_mean_norm      — (8 × 1): human mean normalized
    h_se_norm        — (8 × 1): human SE normalized
    gpt54_mean_norm  — (8 × 1): GPT-5.4 mean
    gpt54_se_norm    — (8 × 1): GPT-5.4 SE
    gem35_mean_norm  — (8 × 1): Gemini-3.5-Flash mean
    gem35_se_norm    — (8 × 1): Gemini-3.5-Flash SE
"""

import numpy as np
import scipy.io as sio
import pandas as pd
from pathlib import Path

BASE   = Path('/Users/wuxiaoyan/Desktop/TPP_culture_AI')
OUT    = Path(__file__).parent / 'data'
MFIT   = BASE / 'ModelFitting'
OUT.mkdir(exist_ok=True)

PARAM_NAMES  = ['gama','envy','guilt','lambda','oumiga','kapa','etak','etaa']
MODEL_LABELS = ['M1 Baseline','M2 SI','M3 SI+SCI','M4 +VCI',
                'M5 +EC','M6 +RP','M7 +II','M8 Full']

# ── 1. Load results ───────────────────────────────────────────────────────────
hr   = sio.loadmat(str(MFIT/'results_human_aligned.mat'),
                   squeeze_me=True, struct_as_record=False)['results_aligned']
ar   = sio.loadmat(str(MFIT/'results_ai.mat'),
                   squeeze_me=True, struct_as_record=False)['results_ai']
adata = sio.loadmat(str(MFIT/'data_ai_all.mat'),
                    squeeze_me=True, struct_as_record=False)['data_ai']

meta = pd.DataFrame({
    'model':  [str(s.model)  for s in adata],
    'method': [str(s.Method) for s in adata],
})

# ── 2. Model complexity counts ────────────────────────────────────────────────

# Human: best model per participant (AICc argmin across M1-M8)
aicc_h  = np.column_stack([np.array(hr[i].aicc).flatten() for i in range(8)])
h_best  = np.argmin(aicc_h, axis=1)             # (875,)
h_count = np.bincount(h_best, minlength=8)       # (8,)
n_human = len(h_best)
print(f'Human best-model distribution: {h_count}  (N={n_human})')

# AI: best model per AI persona (prob method only)
aicc_ai  = np.column_stack([np.array(ar[i].aicc).flatten() for i in range(8)])  # (192,8)
ai_best  = np.argmin(aicc_ai, axis=1)

prob_mask  = (meta['method'] == 'prob').values
meta_prob  = meta[prob_mask].reset_index(drop=True)
ai_best_p  = ai_best[prob_mask]

# All model+method combinations, ordered by company
MODEL_ORDER = [
    ('GPT-4o',            'logprobs'),
    ('GPT-4o',            'prob'),
    ('GPT-4o-mini',       'logprobs'),
    ('GPT-4o-mini',       'prob'),
    ('GPT-5.4',           'prob'),
    ('GPT-5.4-mini',      'prob'),
    ('Gemini-2.5-Flash',  'prob'),
    ('Gemini-3.5-Flash',  'prob'),
    ('DeepSeek-V4-Flash', 'logprobs'),
    ('DeepSeek-V4-Pro',   'logprobs'),
    ('DeepSeek-V4-Pro',   'prob'),
    ('Mistral-Small-4',   'prob'),
]

def make_label(model, method):
    suffix = ' [lp]' if method == 'logprobs' else ''
    return model + suffix

row_names   = ['Human'] + [make_label(m, mt) for m, mt in MODEL_ORDER]
best_counts = []
n_subjects  = []
is_human    = []

# Human row
best_counts.append(h_count)
n_subjects.append(n_human)
is_human.append(1)

for model, method in MODEL_ORDER:
    idx = np.where((meta['model'].values == model) &
                   (meta['method'].values == method))[0]
    if len(idx) == 0:
        bc = np.zeros(8, dtype=int)
        n  = 0
    else:
        bc = np.bincount(ai_best[idx], minlength=8)
        n  = len(idx)
    best_counts.append(bc)
    n_subjects.append(n)
    is_human.append(0)
    print(f'  {make_label(model,method)}: {bc}  (N={n})')

best_counts_mat = np.vstack(best_counts).astype(float)  # (n_rows, 8)
n_subjects_arr  = np.array(n_subjects, dtype=float)
is_human_arr    = np.array(is_human,   dtype=float)

# ── 3. Radar parameters (M8 params for all subjects) ─────────────────────────
# Use M8 (index 7) for EVERYONE — M8 has all 10 params fitted for every subject.
# This ensures fair comparison across models regardless of which model won.
# Parameters: gama, envy, guilt, lamda, oumiga, kapa, etak, etaa (first 8 of 10)
PARAM_NAMES = ['gama', 'envy', 'guilt', 'lamda', 'oumiga', 'kapa', 'etak', 'etaa']

h_x = np.array(hr[7].x)[:, :8]           # (875, 8)
ai_x8 = np.array([np.array(ar[7].x[i]).flatten()[:8] for i in range(192)])  # (192, 8)

# Normalization: per-parameter, shift so min across (human + all AI) = 0,
# then scale so max = 1. This handles kapa which can be negative.
all_x = np.vstack([h_x, ai_x8])          # (875+192, 8)
xmin  = all_x.min(axis=0)
xmax  = all_x.max(axis=0)
xrng  = np.where(xmax - xmin > 0, xmax - xmin, 1.0)

h_norm   = (h_x   - xmin) / xrng         # (875, 8)
ai_norm8 = (ai_x8 - xmin) / xrng         # (192, 8)

def mean_se(mat):
    m  = mat.mean(axis=0)
    se = mat.std(axis=0) / np.sqrt(len(mat))
    return m, se

# Save RAW (un-normalized) means and SEs.
# Per-axis normalization is done in MATLAB so the radar shows meaningful group differences.
h_mean, h_se = mean_se(h_x)
print(f'\nHuman (N={len(h_x)}):')
for i, p in enumerate(PARAM_NAMES):
    print(f'  {p}: mean={h_mean[i]:.3f}, SE={h_se[i]:.3f}')

# GPT-4o-mini logprobs (M8 Full winner, 16/16)
idx_mini_lp = [i for i,(m,mt) in enumerate(zip(meta['model'].values, meta['method'].values))
               if m == 'GPT-4o-mini' and mt == 'logprobs']
mini_lp_mean, mini_lp_se = mean_se(ai_x8[idx_mini_lp])
print(f'\nGPT-4o-mini [lp] (N={len(idx_mini_lp)}):')
for i, p in enumerate(PARAM_NAMES):
    print(f'  {p}: mean={mini_lp_mean[i]:.3f}, SE={mini_lp_se[i]:.3f}')

# GPT-5.4 prob (M8 Full winner, 14/16)
idx54  = [i for i,(m,mt) in enumerate(zip(meta['model'].values, meta['method'].values))
          if m == 'GPT-5.4' and mt == 'prob']
gpt54_mean, gpt54_se = mean_se(ai_x8[idx54])
print(f'\nGPT-5.4 (N={len(idx54)}):')
for i, p in enumerate(PARAM_NAMES):
    print(f'  {p}: mean={gpt54_mean[i]:.3f}, SE={gpt54_se[i]:.3f}')

# Gemini-3.5-Flash prob (M8 Full winner, 14/16)
idxg35 = [i for i,(m,mt) in enumerate(zip(meta['model'].values, meta['method'].values))
          if m == 'Gemini-3.5-Flash' and mt == 'prob']
gem35_mean, gem35_se = mean_se(ai_x8[idxg35])
print(f'\nGemini-3.5-Flash (N={len(idxg35)}):')
for i, p in enumerate(PARAM_NAMES):
    print(f'  {p}: mean={gem35_mean[i]:.3f}, SE={gem35_se[i]:.3f}')

# ── 4. Save ───────────────────────────────────────────────────────────────────
def cell(lst):
    arr = np.empty(len(lst), dtype=object)
    for i, v in enumerate(lst): arr[i] = v
    return arr

out_path = str(OUT / 'fig4_data.mat')
sio.savemat(out_path, {
    # Panel A
    'row_names':        cell(row_names),
    'best_counts_mat':  best_counts_mat,
    'n_subjects':       n_subjects_arr,
    'is_human':         is_human_arr,
    'model_labels':     cell(MODEL_LABELS),

    # Panel B (radar: Human vs GPT-4o-mini[lp] vs GPT-5.4 vs Gemini-3.5-Flash)
    'param_names':       cell(PARAM_NAMES),
    'h_mean_norm':       h_mean,
    'h_se_norm':         h_se,
    'mini_lp_mean_norm': mini_lp_mean,
    'mini_lp_se_norm':   mini_lp_se,
    'gpt54_mean_norm':   gpt54_mean,
    'gpt54_se_norm':     gpt54_se,
    'gem35_mean_norm':   gem35_mean,
    'gem35_se_norm':     gem35_se,
}, do_compression=True)
print(f'\nSaved: {out_path}')
