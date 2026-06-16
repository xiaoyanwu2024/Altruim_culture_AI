# =============================================================================
# Author : Xiaoyan Wu
# Date   : June 2026
# Description: Preprocessing — extracts and summarises M8 parameter distributions for human vs AI
# =============================================================================

"""
analyze_m8_params.py
=====================
Compare M8 Full model parameters between:
  - Human (N=875)
  - GPT-5.4 prob (N=16)
  - Gemini-3.5-Flash prob (N=16)

Parameters (first 8 of 10):
  γ (gama)   — loss aversion / norm sensitivity
  α (envy)   — disadvantageous inequity aversion
  β (guilt)  — advantageous inequity aversion
  λ (lamda)  — inverse temperature (decision noise)
  ω (oumiga) — social norm weight
  κ (kapa)   — reciprocity / punishment sensitivity
  η_no (etak) — learning rate for non-intervention
  η_yes (etaa) — learning rate for intervention

Statistical tests:
  - AI vs Human: Mann-Whitney U (non-parametric, unequal N)
  - GPT-5.4 vs Gemini: Mann-Whitney U (N=16 each)
  - Bonferroni correction across 8 parameters per comparison
"""

import numpy as np
import scipy.io as sio
from scipy.stats import mannwhitneyu, shapiro
import warnings
warnings.filterwarnings('ignore')

# ── Load data ─────────────────────────────────────────────────────────────
BASE = '/Users/wuxiaoyan/Desktop/TPP_culture_AI'
ar   = sio.loadmat(f'{BASE}/ModelFitting/results_ai.mat',           simplify_cells=True)['results_ai']
hr   = sio.loadmat(f'{BASE}/ModelFitting/results_human_aligned.mat',simplify_cells=True)['results_aligned']
da   = sio.loadmat(f'{BASE}/ModelFitting/data_ai_all.mat',          simplify_cells=True)['data_ai']

x8_ai = np.array(ar[7]['x'])    # (192, 10)
x8_hu = np.array(hr[7]['x'])    # (875, 10)

models  = [d['model']  for d in da]
methods = [d['Method'] for d in da]

idx_54  = [i for i,(m,mt) in enumerate(zip(models,methods)) if m=='GPT-5.4'           and mt=='prob']
idx_g35 = [i for i,(m,mt) in enumerate(zip(models,methods)) if m=='Gemini-3.5-Flash'  and mt=='prob']

# Extract first 8 params
hu   = x8_hu[:, :8]
g54  = x8_ai[idx_54,  :8]
g35  = x8_ai[idx_g35, :8]

PARAM_NAMES = ['γ (gama)', 'α (envy)', 'β (guilt)', 'λ (lamda)',
               'ω (omega)', 'κ (kapa)', 'η_no (etak)', 'η_yes (etaa)']
N_PARAMS = 8
ALPHA    = 0.05
ALPHA_BONF = ALPHA / N_PARAMS   # Bonferroni corrected threshold

# ── Descriptive statistics ─────────────────────────────────────────────────
print("=" * 75)
print("M8 Parameter Descriptive Statistics")
print("=" * 75)
print(f"{'Param':<14} {'Human mean±SD':>18} {'GPT-5.4 mean±SD':>20} {'Gemini mean±SD':>20}")
print("-" * 75)
for k in range(N_PARAMS):
    print(f"{PARAM_NAMES[k]:<14} "
          f"{np.mean(hu[:,k]):>8.3f}±{np.std(hu[:,k]):<8.3f} "
          f"{np.mean(g54[:,k]):>8.3f}±{np.std(g54[:,k]):<8.3f} "
          f"{np.mean(g35[:,k]):>8.3f}±{np.std(g35[:,k]):<8.3f}")

# ── Mann-Whitney U tests ───────────────────────────────────────────────────
def mwu(a, b, label):
    stat, p = mannwhitneyu(a, b, alternative='two-sided')
    sig = '***' if p < 0.001 else ('**' if p < 0.01 else ('*' if p < ALPHA_BONF else ('.' if p < 0.05 else 'ns')))
    return p, sig

print("\n" + "=" * 75)
print(f"Mann-Whitney U Tests  (Bonferroni α = {ALPHA_BONF:.4f}, * = significant)")
print("=" * 75)

# GPT-5.4 vs Human
print("\n── GPT-5.4 vs Human ──")
sig_54_hu = []
for k in range(N_PARAMS):
    p, sig = mwu(g54[:,k], hu[:,k], PARAM_NAMES[k])
    d_mean = np.mean(g54[:,k]) - np.mean(hu[:,k])
    print(f"  {PARAM_NAMES[k]:<14}  p={p:.4f}  {sig}   Δmean={d_mean:+.3f}")
    if sig not in ['ns', '.']:
        sig_54_hu.append((k, p, d_mean))

# Gemini-3.5-Flash vs Human
print("\n── Gemini-3.5-Flash vs Human ──")
sig_g35_hu = []
for k in range(N_PARAMS):
    p, sig = mwu(g35[:,k], hu[:,k], PARAM_NAMES[k])
    d_mean = np.mean(g35[:,k]) - np.mean(hu[:,k])
    print(f"  {PARAM_NAMES[k]:<14}  p={p:.4f}  {sig}   Δmean={d_mean:+.3f}")
    if sig not in ['ns', '.']:
        sig_g35_hu.append((k, p, d_mean))

# GPT-5.4 vs Gemini
print("\n── GPT-5.4 vs Gemini-3.5-Flash ──")
sig_54_g35 = []
for k in range(N_PARAMS):
    p, sig = mwu(g54[:,k], g35[:,k], PARAM_NAMES[k])
    d_mean = np.mean(g54[:,k]) - np.mean(g35[:,k])
    print(f"  {PARAM_NAMES[k]:<14}  p={p:.4f}  {sig}   Δmean={d_mean:+.3f}")
    if sig not in ['ns', '.']:
        sig_54_g35.append((k, p, d_mean))

# ── Conclusions based on significant results only ──────────────────────────
print("\n" + "=" * 75)
print("CONCLUSIONS (significant results only, Bonferroni corrected)")
print("=" * 75)

param_short = ['γ', 'α', 'β', 'λ', 'ω', 'κ', 'η_no', 'η_yes']

print("\n[GPT-5.4 vs Human]")
if not sig_54_hu:
    print("  No parameters significantly different from humans after correction.")
else:
    for k, p, d in sig_54_hu:
        direction = "higher" if d > 0 else "lower"
        print(f"  {param_short[k]}: GPT-5.4 is significantly {direction} than humans "
              f"(Δ={d:+.3f}, p={p:.4f})")

print("\n[Gemini-3.5-Flash vs Human]")
if not sig_g35_hu:
    print("  No parameters significantly different from humans after correction.")
else:
    for k, p, d in sig_g35_hu:
        direction = "higher" if d > 0 else "lower"
        print(f"  {param_short[k]}: Gemini-3.5-Flash is significantly {direction} than humans "
              f"(Δ={d:+.3f}, p={p:.4f})")

print("\n[GPT-5.4 vs Gemini-3.5-Flash]")
if not sig_54_g35:
    print("  No significant differences between the two AI models.")
else:
    for k, p, d in sig_54_g35:
        direction = "higher" if d > 0 else "lower"
        print(f"  {param_short[k]}: GPT-5.4 is significantly {direction} than Gemini "
              f"(Δ={d:+.3f}, p={p:.4f})")

print("\n" + "=" * 75)
print("Legend: *** p<0.001  ** p<0.01  * p<Bonferroni threshold  . p<0.05 (uncorrected)  ns not significant")
print(f"Bonferroni threshold = {ALPHA_BONF:.4f} (α=0.05 / {N_PARAMS} params)")
