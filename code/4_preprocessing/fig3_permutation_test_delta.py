# =============================================================================
# Author : Xiaoyan Wu
# Date   : June 2026
# Description: Preprocessing — Mantel permutation test for cross-cultural delta matrices
# =============================================================================

"""
permutation_test_delta.py
=========================
Permutation test for pairwise country-condition delta correlation
(Human ΔP vs AI ΔP, 28 pairs × 100 conditions = 2800 values per model).

For each model:
  - Observed r: Pearson correlation between Human Δ and AI Δ (2800 points)
  - Null distribution: shuffle AI Δ values 5000 times, recompute r each time
  - Permutation p-value (two-tailed): proportion of |r_null| >= |r_obs|

Outputs:
  - Console: summary table
  - Figure: 2×4 grid, one panel per model
              histogram of null distribution + observed r marker + stats
"""

import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path

# ── Config ────────────────────────────────────────────────────────────────────
DATA_PATH  = Path(__file__).parent / 'data' / 'fig3_data.mat'
OUT_DIR    = Path(__file__).parent / 'output'
OUT_DIR.mkdir(exist_ok=True)

N_PERM  = 5000
SEED    = 42
FONT    = 'Arial'

# Model colors — alphabetical order (prob + logprobs)
# 1 DeepSeek-V4-Flash_logprobs  2 DeepSeek-V4-Pro_logprobs   3 DeepSeek-V4-Pro_prob
# 4 GPT-4o-mini_logprobs        5 GPT-4o-mini_prob           6 GPT-4o_logprobs
# 7 GPT-4o_prob                 8 GPT-5.4-mini_prob          9 GPT-5.4_prob
#10 Gemini-2.5-Flash_prob      11 Gemini-3.5-Flash_prob     12 Mistral-Small-4_prob
MODEL_COLORS = [
    (0.45, 0.25, 0.60),   # dark purple [lp]  DeepSeek-V4-Flash [lp]
    (0.72, 0.52, 0.88),   # light purple [lp] DeepSeek-V4-Pro [lp]
    (0.60, 0.40, 0.72),   # medium purple     DeepSeek-V4-Pro
    (0.80, 0.45, 0.05),   # dark orange [lp]  GPT-4o-mini [lp]
    (0.96, 0.63, 0.13),   # orange            GPT-4o-mini
    (0.65, 0.20, 0.10),   # dark coral [lp]   GPT-4o [lp]
    (0.92, 0.35, 0.22),   # coral-red         GPT-4o
    (0.96, 0.63, 0.72),   # pink              GPT-5.4-mini
    (0.60, 0.55, 0.80),   # lavender          GPT-5.4
    (0.20, 0.62, 0.55),   # dark teal         Gemini-2.5-Flash
    (0.94, 0.78, 0.19),   # yellow            Gemini-3.5-Flash
    (0.72, 0.79, 0.41),   # yellow-green      Mistral-Small-4
]

# ── Load data ─────────────────────────────────────────────────────────────────
mat = sio.loadmat(str(DATA_PATH), squeeze_me=True, struct_as_record=False)

pair_delta_human = mat['pair_delta_human'].flatten()   # (2800,)
pair_delta_ai    = mat['pair_delta_ai']                # (8 × 2800)
delta_ai_labels  = [str(x) for x in mat['delta_ai_labels'].flatten()]
n_models = len(delta_ai_labels)

# Pretty label: remove _prob suffix, shorten
def pretty(s):
    s = s.replace('_prob', '').replace('_logprobs', ' [lp]')
    return s

# ── Run permutation test ──────────────────────────────────────────────────────
rng = np.random.default_rng(SEED)

results = []
null_dists = []

print(f"\n{'='*70}")
print(f"Permutation test  (N={N_PERM})  — Human Δ vs AI Δ  (2800 data points)")
print(f"{'='*70}")
print(f"{'Model':<30}  {'r_obs':>7}  {'perm_p':>8}  {'percentile':>10}  {'null_95CI':>18}")
print(f"{'-'*70}")

for mi in range(n_models):
    hx = pair_delta_human
    ay = pair_delta_ai[mi].flatten()
    valid = ~np.isnan(hx) & ~np.isnan(ay)
    hx_v, ay_v = hx[valid], ay[valid]

    # Observed r
    r_obs = np.corrcoef(hx_v, ay_v)[0, 1]

    # Null distribution
    null_r = np.empty(N_PERM)
    for k in range(N_PERM):
        null_r[k] = np.corrcoef(hx_v, rng.permutation(ay_v))[0, 1]

    # Two-tailed permutation p
    p_perm = np.mean(np.abs(null_r) >= np.abs(r_obs))
    pct    = np.mean(null_r <= r_obs) * 100
    ci_lo  = np.percentile(null_r, 2.5)
    ci_hi  = np.percentile(null_r, 97.5)

    if p_perm == 0:
        p_str = f'<{1/N_PERM:.4f}'
    else:
        p_str = f'{p_perm:.4f}'

    print(f"{pretty(delta_ai_labels[mi]):<30}  {r_obs:>7.3f}  {p_str:>8}  "
          f"{pct:>9.1f}%  [{ci_lo:.3f}, {ci_hi:.3f}]")

    results.append({'label': delta_ai_labels[mi], 'r_obs': r_obs,
                    'p_perm': p_perm, 'pct': pct, 'ci_lo': ci_lo, 'ci_hi': ci_hi})
    null_dists.append(null_r)

print(f"{'='*70}\n")

# ── Plot ──────────────────────────────────────────────────────────────────────
N_COLS = 4
N_ROWS = int(np.ceil(n_models / N_COLS))
fig, axes = plt.subplots(N_ROWS, N_COLS,
                          figsize=(N_COLS * 4.2, N_ROWS * 3.8),
                          facecolor='white')
axes = axes.flatten()
# hide unused subplots
for idx in range(n_models, len(axes)):
    axes[idx].set_visible(False)

for mi in range(n_models):
    ax   = axes[mi]
    col  = MODEL_COLORS[mi]
    res  = results[mi]
    null = null_dists[mi]
    r_obs = res['r_obs']
    p_perm = res['p_perm']
    pct    = res['pct']
    ci_lo  = res['ci_lo']
    ci_hi  = res['ci_hi']

    # Histogram of null distribution
    ax.hist(null, bins=60, color=col, alpha=0.55, edgecolor='none', density=True)

    # 95% CI boundaries of null
    ax.axvline(ci_lo, color='gray', lw=0.9, ls='--', alpha=0.7)
    ax.axvline(ci_hi, color='gray', lw=0.9, ls='--', alpha=0.7,
               label='null 95% CI')

    # Zero line
    ax.axvline(0, color='black', lw=0.7, alpha=0.4)

    # Observed r
    ax.axvline(r_obs, color=col, lw=2.2, ls='-',
               label=f'observed r = {r_obs:.3f}')

    # Shading: tail beyond observed r
    tail_mask = null >= abs(r_obs) if r_obs > 0 else null <= -abs(r_obs)
    # shade both tails for two-tailed
    x_range = np.linspace(null.min(), null.max(), 1000)
    from scipy.stats import gaussian_kde
    kde = gaussian_kde(null)
    kde_y = kde(x_range)
    # right tail
    ax.fill_between(x_range, kde_y,
                    where=(x_range >= abs(r_obs)),
                    color=col, alpha=0.35, label='_nolegend_')
    # left tail
    ax.fill_between(x_range, kde_y,
                    where=(x_range <= -abs(r_obs)),
                    color=col, alpha=0.35, label='_nolegend_')

    # KDE overlay
    ax.plot(x_range, kde_y, color=col, lw=1.4, alpha=0.9)

    # Stats annotation
    if p_perm == 0:
        p_str = f'p < {1/N_PERM:.4f}'
    else:
        p_str = f'p = {p_perm:.4f}'

    sig = '***' if (p_perm < 0.001 or p_perm == 0) else ('**' if p_perm < 0.01 else ('*' if p_perm < 0.05 else 'ns'))

    ax.text(0.97, 0.97,
            f'r = {r_obs:.3f}\n{p_str}  {sig}\npercentile: {pct:.1f}%',
            transform=ax.transAxes,
            ha='right', va='top', fontsize=8, fontfamily=FONT,
            color=tuple(c * 0.7 for c in col),
            bbox=dict(facecolor='white', edgecolor='none', alpha=0.8, pad=2))

    # Title
    ax.set_title(pretty(delta_ai_labels[mi]), fontsize=10, fontfamily=FONT,
                 fontweight='bold', color='black', pad=4)

    ax.set_xlabel('Permuted r', fontsize=8.5, fontfamily=FONT)
    ax.set_ylabel('Density', fontsize=8.5, fontfamily=FONT)
    ax.tick_params(labelsize=7.5)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # Legend (only first panel)
    if mi == 0:
        from matplotlib.lines import Line2D
        legend_elements = [
            Line2D([0], [0], color=col, lw=2.2, label=f'Observed r'),
            Line2D([0], [0], color='gray', lw=0.9, ls='--', label='Null 95% CI'),
            mpatches.Patch(facecolor=col, alpha=0.35, label='Tail region'),
        ]
        ax.legend(handles=legend_elements, fontsize=7, loc='upper left',
                  frameon=False, prop={'family': FONT})

fig.suptitle(
    f'Permutation test: Human ΔP vs AI ΔP  (N={N_PERM} permutations, 2800 data points)\n'
    f'Shaded region = tail beyond |r_obs|; dashed lines = null 95% CI',
    fontsize=11, fontfamily=FONT, fontweight='bold', y=1.01
)

plt.tight_layout()

out_pdf = OUT_DIR / 'permutation_test_delta.pdf'
out_png = OUT_DIR / 'permutation_test_delta.png'
fig.savefig(out_pdf, bbox_inches='tight', dpi=200)
fig.savefig(out_png, bbox_inches='tight', dpi=200)
print(f'Saved: {out_pdf}')
print(f'Saved: {out_png}')
plt.show()
