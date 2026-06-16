# =============================================================================
# Author : Xiaoyan Wu
# Date   : June 2026
# Description: Preprocessing — computes MDS configurations and generates mds_data.mat for Figure 3
# =============================================================================

"""
Figure3 MDS — preprocess_mds.py
Compute per-country 100-condition intervention rate vectors,
then run classical MDS (2D) on the country × country distance matrix.

Conditions: block_type(2) × ratio(2) × offer_std(5) × cost(5) = 100
Offer mapping: x1 rounded to nearest of {90,80,70,60,50}

Output: Figure3/data/mds_data.mat
"""

import numpy as np
import pandas as pd
import scipy.io as sio
from pathlib import Path
from sklearn.manifold import MDS
from scipy.spatial.distance import pdist, squareform

BASE  = Path('/Users/wuxiaoyan/Desktop/TPP_culture_AI')
OUT   = BASE / 'Figure3' / 'data'

# ── 1. Load raw data ──────────────────────────────────────────────────────
df = pd.read_csv(str(BASE / 'human_data.csv'))

# ── 2. Map offer to standard levels ───────────────────────────────────────
def map_offer(x1):
    levels = np.array([90, 80, 70, 60, 50])
    return int(levels[np.argmin(np.abs(levels - x1))])

df['offer_std'] = df['x1'].apply(map_offer)

# ── 3. Build condition key ─────────────────────────────────────────────────
BLOCK_TYPES = ['punish', 'help']
RATIOS      = [1.5, 3.0]
OFFERS      = [90, 80, 70, 60, 50]
COSTS       = [10, 20, 30, 40, 50]

# Sorted condition list (100 conditions, fixed order)
conditions = []
for bt in BLOCK_TYPES:
    for ra in RATIOS:
        for of in OFFERS:
            for co in COSTS:
                conditions.append((bt, ra, of, co))

cond_index = {c: i for i, c in enumerate(conditions)}
print(f"Total conditions: {len(conditions)}")

# ── 4. Per-country 100-dim vectors ────────────────────────────────────────
countries = sorted(df['country'].unique())
n_c       = len(countries)
X         = np.full((n_c, 100), np.nan)

for ci, country in enumerate(countries):
    sub = df[df['country'] == country].copy()
    n_subs = sub['subid'].nunique()

    # For each condition: mean intervention rate across all subjects
    vec = np.zeros(100)
    for cond, idx in cond_index.items():
        bt, ra, of, co = cond
        mask = ((sub['block_type'] == bt) &
                (np.abs(sub['ratio'] - ra) < 0.01) &
                (sub['offer_std'] == of) &
                (sub['cost'] == co))
        trials = sub.loc[mask, 'action']
        vec[idx] = trials.mean() if len(trials) > 0 else np.nan

    X[ci, :] = vec
    print(f"  {country:<15} N={n_subs:>3}  "
          f"mean_intervention={vec[~np.isnan(vec)].mean():.3f}  "
          f"nan_conditions={np.isnan(vec).sum()}")

# ── 5. Classical MDS ──────────────────────────────────────────────────────
# Distance matrix: Euclidean on 100-dim vectors
D = squareform(pdist(X, metric='euclidean'))
print(f"\nDistance matrix (Euclidean, {n_c}x{n_c}):")
print(np.round(D, 3))

# Also compute 1 - Pearson r (for comparison)
from scipy.stats import pearsonr
D_corr = np.zeros((n_c, n_c))
for i in range(n_c):
    for j in range(n_c):
        r, _ = pearsonr(X[i], X[j])
        D_corr[i, j] = 1 - r

print(f"\n1-Pearson distance matrix:")
print(np.round(D_corr, 3))

# Run MDS (2D) on Euclidean distance
mds2 = MDS(n_components=2, dissimilarity='precomputed',
           random_state=42, n_init=20, max_iter=1000)
coords2 = mds2.fit_transform(D)
stress2 = mds2.stress_
print(f"\nMDS 2D stress = {stress2:.4f}")

# Run MDS (3D) for comparison
mds3 = MDS(n_components=3, dissimilarity='precomputed',
           random_state=42, n_init=20, max_iter=1000)
coords3 = mds3.fit_transform(D)
stress3 = mds3.stress_
print(f"MDS 3D stress = {stress3:.4f}")

# Normalized stress (Kruskal's stress-1)
# stress_normalized = stress / sqrt(sum(D^2)/2)
denom = np.sqrt(np.sum(D**2) / 2)
s1_2d = np.sqrt(stress2) / denom
s1_3d = np.sqrt(stress3) / denom
print(f"\nKruskal stress-1: 2D = {s1_2d:.4f},  3D = {s1_3d:.4f}")
print("(< 0.05 = excellent, < 0.10 = good, < 0.20 = fair)")

# ── 6. Bootstrap MDS (Procrustes-aligned) ────────────────────────────────
from scipy.spatial import procrustes
import warnings

N_BOOT = 1000
rng_bs = np.random.default_rng(42)

# Pre-compute per-subject 100-dim vectors (done ONCE — not inside the loop)
print("\nPre-computing per-subject condition vectors...")
all_subids = df['subid'].unique()
subid_to_idx = {s: i for i, s in enumerate(all_subids)}
n_subjs = len(all_subids)

# subj_mat[i, j] = intervention rate of subject i in condition j
subj_mat = np.full((n_subjs, 100), np.nan)
for subid, sdf in df.groupby('subid'):
    si = subid_to_idx[subid]
    for cond, idx in cond_index.items():
        bt, ra, of, co = cond
        mask = ((sdf['block_type'] == bt) &
                (np.abs(sdf['ratio'] - ra) < 0.01) &
                (sdf['offer_std'] == of) &
                (sdf['cost'] == co))
        vals = sdf.loc[mask, 'action'].values
        if len(vals) > 0:
            subj_mat[si, idx] = vals.mean()

print(f"  Done. Subject matrix shape: {subj_mat.shape}")

# Per-country: list of row indices into subj_mat
country_rows = {}
for ci, country in enumerate(countries):
    sids = df[df['country'] == country]['subid'].unique()
    country_rows[ci] = np.array([subid_to_idx[s] for s in sids])

boot_coords = np.zeros((N_BOOT, n_c, 2))

print(f"Running {N_BOOT} bootstrap iterations...")
for b in range(N_BOOT):
    X_boot = np.zeros((n_c, 100))
    for ci in range(n_c):
        rows = country_rows[ci]
        # Resample with replacement (handles duplicates correctly)
        sampled = rng_bs.choice(rows, size=len(rows), replace=True)
        # nanmean across resampled subjects
        X_boot[ci] = np.nanmean(subj_mat[sampled, :], axis=0)

    D_boot = squareform(pdist(X_boot, metric='euclidean'))
    mds_b = MDS(n_components=2, dissimilarity='precomputed',
                random_state=b, n_init=4, max_iter=500,
                init='random')
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        try:
            c_boot = mds_b.fit_transform(D_boot)
        except Exception:
            boot_coords[b] = coords2
            continue

    _, c_aligned, _ = procrustes(coords2, c_boot)
    boot_coords[b] = c_aligned

    if (b + 1) % 100 == 0:
        print(f"  Bootstrap {b+1}/{N_BOOT}")

# Per-country: mean and covariance of 2D positions
boot_mean = boot_coords.mean(axis=0)   # (8,2)
boot_cov  = np.zeros((n_c, 2, 2))
for ci in range(n_c):
    boot_cov[ci] = np.cov(boot_coords[:, ci, :].T)

print("\nBootstrap complete.")
print("Per-country positional SD (Dim1, Dim2):")
for ci, country in enumerate(countries):
    sd1 = np.sqrt(boot_cov[ci,0,0])
    sd2 = np.sqrt(boot_cov[ci,1,1])
    print(f"  {country:<15}: SD1={sd1:.4f}  SD2={sd2:.4f}")

def cell(lst):
    arr = np.empty(len(lst), dtype=object)
    for i, v in enumerate(lst): arr[i] = v
    return arr

sio.savemat(str(OUT / 'mds_data.mat'), {
    'countries':    cell(countries),
    'X':            X,
    'D':            D,
    'D_corr':       D_corr,
    'coords2':      coords2,
    'coords3':      coords3,
    'stress2':      stress2,
    'stress3':      stress3,
    'stress1_2d':   s1_2d,
    'stress1_3d':   s1_3d,
    # Bootstrap results
    'boot_coords':  boot_coords,    # (1000, 8, 2)
    'boot_mean':    boot_mean,      # (8, 2) Procrustes-aligned mean
    'boot_cov':     boot_cov,       # (8, 2, 2) covariance per country
    'n_boot':       float(N_BOOT),
}, do_compression=True)
print(f"\nSaved: {OUT}/mds_data.mat")
