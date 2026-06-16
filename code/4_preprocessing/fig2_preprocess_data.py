# =============================================================================
# Author : Xiaoyan Wu
# Date   : June 2026
# Description: Preprocessing — loads raw CSV/MAT files and generates dg_data.mat and tpp_data.mat for Figure 2
# =============================================================================

"""
preprocess_data.py
==================
Loads raw data from CSV and MAT files, computes summary statistics,
and saves clean data matrices for MATLAB scripts.

Outputs (in ./data/):
  dg_data.mat   — Dictator Game data
  tpp_data.mat  — Third-Party Punishment data

Run once before any MATLAB script:
    python preprocess_data.py
"""

import numpy as np
import pandas as pd
import scipy.io as sio
from scipy.stats import pearsonr
import os

BASE  = '/Users/wuxiaoyan/Desktop/TPP_culture_AI'
OUT   = os.path.join(os.path.dirname(__file__), 'data')
os.makedirs(OUT, exist_ok=True)

# ─────────────────────────────────────────────────────────────────────────────
# 1. DICTATOR GAME
# ─────────────────────────────────────────────────────────────────────────────
print('=== Processing Dictator Game ===')

# Human DG
hdf = pd.read_csv(f'{BASE}/human_data.csv')
h_dg_subj = hdf.drop_duplicates('subid')[['subid','country','gender','DG']]

COUNTRIES = sorted(['South Africa','Italy','Mexico','Poland','Portugal','Greece','Spain','Chile'])
h_dg_mean = np.array([h_dg_subj[h_dg_subj['country']==c]['DG'].mean() for c in COUNTRIES])
h_dg_se   = np.array([h_dg_subj[h_dg_subj['country']==c]['DG'].sem()  for c in COUNTRIES])

# AI DG — load all CSV files from Task1_DG_AI_data
dg_dir = f'{BASE}/Task1_DG_AI_data'
dg_dfs = []
for f in os.listdir(dg_dir):
    if f.endswith('.csv') and f != 'task1_combined_final.csv':
        dg_dfs.append(pd.read_csv(f'{dg_dir}/{f}'))
ai_dg_all = pd.concat(dg_dfs, ignore_index=True)

# Pretty name map for DG models
DG_NAME_MAP = {
    'gpt4o':                  'GPT-4o',
    'gpt4o_mini':             'GPT-4o-mini',
    'gpt_5.4':                'GPT-5.4',
    'gpt_5.4_mini':           'GPT-5.4-mini',
    'gpt_5.5':                'GPT-5.5',
    'gpt_oss_120b':           'GPT-OSS-120B',
    'deepseek_v4pro':         'DeepSeek-V4-Pro',
    'deepseek_v4pro_thinking':'DeepSeek-V4-Pro-Think',
    'deepseek_v4flash':       'DeepSeek-V4-Flash',
    'gemini_2.5_flash':       'Gemini-2.5-Flash',
    'gemma4_31b':             'Gemma-4-31B',
    'mistral_small4':         'Mistral-Small-4',
    'llama3_70b':             'LLaMA-3-70B',
    'llama4_scout':           'LLaMA-4-Scout',
}

dg_model_list = sorted(ai_dg_all['model'].unique())
n_dg_models   = len(dg_model_list)

ai_dg_mean  = np.full((n_dg_models, len(COUNTRIES)), np.nan)
ai_dg_mae   = np.full(n_dg_models, np.nan)
ai_dg_names = []

for i, m in enumerate(dg_model_list):
    sub = ai_dg_all[ai_dg_all['model'] == m]
    means = np.array([sub[sub['country']==c]['allocation'].mean() for c in COUNTRIES])
    ai_dg_mean[i, :] = means
    ai_dg_mae[i]     = np.nanmean(np.abs(means - h_dg_mean))
    ai_dg_names.append(DG_NAME_MAP.get(m, m))

# Sort by MAE
sort_idx      = np.argsort(ai_dg_mae)
top3_dg_idx   = sort_idx[:3]

print(f'  DG top 3 models (by MAE):')
for i in top3_dg_idx:
    print(f'    {ai_dg_names[i]}: MAE={ai_dg_mae[i]:.3f}')

# ─────────────────────────────────────────────────────────────────────────────
# 2. THIRD-PARTY PUNISHMENT (TPP)
# ─────────────────────────────────────────────────────────────────────────────
print('\n=== Processing TPP ===')

# Human TPP condition means
hdf['action'] = pd.to_numeric(hdf['action'], errors='coerce')
hdf['offer']  = hdf['x2'].apply(lambda v: int(round(v/10)*10))
hdf['block']  = hdf['block_type'].map({'punish':1,'help':2})

OFFERS = [10, 20, 30, 40, 50]   # recipient share (10=90:10 unfair, 50=50:50 fair)
COSTS  = [10, 20, 30, 40, 50]
BLOCKS = [1, 2]
RATIOS = [1.5, 3.0]
COND   = ['block','offer','cost','ratio']

h_cond_df = hdf.groupby(COND)['action'].mean().reset_index().rename(columns={'action':'h'})

# Build condition label matrix: each row = [block, offer, cost, ratio]
cond_rows = []
for blk in BLOCKS:
    for rat in RATIOS:
        for c in COSTS:
            for o in OFFERS:
                cond_rows.append([blk, o, c, rat])
cond_matrix = np.array(cond_rows, dtype=float)   # 100 × 4: [block,offer,cost,ratio]

# Get human values in same order
def get_h_vec(cond_df, blk, rat, costs, offers):
    """Return 5×5 matrix (rows=cost, cols=offer) for a given block×ratio"""
    mat = np.full((5,5), np.nan)
    sub = cond_df[(cond_df['block']==blk) & (cond_df['ratio']==rat)]
    for ri,c in enumerate(costs):
        for ci,o in enumerate(offers):
            v = sub[(sub['cost']==c)&(sub['offer']==o)]['h']
            if len(v): mat[ri,ci] = v.values[0]
    return mat

# h_vec: 100-element vector aligned with cond_matrix
h_vec = []
for row in cond_rows:
    blk, o, c, rat = row
    sub = h_cond_df[(h_cond_df['block']==blk)&(h_cond_df['offer']==o)&
                    (h_cond_df['cost']==c)&(h_cond_df['ratio']==rat)]['h']
    h_vec.append(sub.values[0] if len(sub) else np.nan)
h_vec = np.array(h_vec)

# AI TPP
OFFER_MAP = {'50:50':50,'60:40':40,'70:30':30,'80:20':20,'90:10':10}
amat = sio.loadmat(f'{BASE}/ModelFitting/data_ai_all.mat', squeeze_me=True, struct_as_record=False)

TPP_NAME_MAP = {
    'GPT-4o_prob':           'GPT-4o',
    'GPT-4o-mini_prob':      'GPT-4o-mini',
    'GPT-5.4_prob':          'GPT-5.4',
    'GPT-5.4-mini_prob':     'GPT-5.4-mini',
    'Gemini-2.5-Flash_prob': 'Gemini-2.5-Flash',
    'Gemini-3.5-Flash_prob': 'Gemini-3.5-Flash',
    'DeepSeek-V4-Pro_prob':  'DeepSeek-V4-Pro',
    'Mistral-Small-4_prob':  'Mistral-Small-4',
    'GPT-4o_logprobs':           'GPT-4o (logprobs)',
    'GPT-4o-mini_logprobs':      'GPT-4o-mini (logprobs)',
    'DeepSeek-V4-Flash_logprobs':'DeepSeek-V4-Flash (logprobs)',
    'DeepSeek-V4-Pro_logprobs':  'DeepSeek-V4-Pro (logprobs)',
}

# Collect per-persona results, then average per model
persona_results = {}
for s in amat['data_ai']:
    lbl = f'{s.model}_{s.Method}'
    if lbl not in TPP_NAME_MAP: continue
    blk   = np.array(s.block).flatten().astype(int)
    offer = np.array(s.offer).flatten()
    cost  = np.array(s.cost).flatten().astype(int)
    ratio = np.array(s.ratio).flatten().astype(float)
    act   = pd.to_numeric(np.array(s.action).flatten(), errors='coerce')
    oint  = np.array([OFFER_MAP.get(str(o).strip(), np.nan) for o in offer])
    aidf  = pd.DataFrame({'block':blk,'offer':oint,'cost':cost,'ratio':ratio,'action':act})

    ai_vec = []
    for row in cond_rows:
        b,o,c,r = row
        sub = aidf[(aidf['block']==b)&(aidf['offer']==o)&
                   (aidf['cost']==c)&(aidf['ratio']==r)]['action']
        ai_vec.append(sub.mean() if len(sub) else np.nan)
    ai_vec = np.array(ai_vec)

    if lbl not in persona_results:
        persona_results[lbl] = []
    persona_results[lbl].append(ai_vec)

# Average across personas (countries×genders)
tpp_model_keys  = sorted(persona_results.keys())
n_tpp_models    = len(tpp_model_keys)
ai_tpp_mean     = np.full((n_tpp_models, 100), np.nan)   # model × condition
ai_tpp_r        = np.full(n_tpp_models, np.nan)
ai_tpp_mae      = np.full(n_tpp_models, np.nan)
ai_tpp_names    = []

for i, lbl in enumerate(tpp_model_keys):
    vecs  = np.array(persona_results[lbl])          # n_personas × 100
    avg   = np.nanmean(vecs, axis=0)
    ai_tpp_mean[i, :] = avg
    mask  = ~np.isnan(avg) & ~np.isnan(h_vec)
    r, _  = pearsonr(h_vec[mask], avg[mask])
    ai_tpp_r[i]   = r
    ai_tpp_mae[i] = np.nanmean(np.abs(avg - h_vec))
    ai_tpp_names.append(TPP_NAME_MAP[lbl])

# Rank by combined (lower MAE + higher r)
mae_rank = pd.Series(ai_tpp_mae).rank()
r_rank   = pd.Series(ai_tpp_r).rank(ascending=False)
combined = mae_rank + r_rank
top3_tpp_idx = combined.argsort().values[:3]

print(f'  TPP top 3 models (by combined MAE+r rank):')
for i in top3_tpp_idx:
    print(f'    {ai_tpp_names[i]}: MAE={ai_tpp_mae[i]:.3f}, r={ai_tpp_r[i]:.3f}')

# ─────────────────────────────────────────────────────────────────────────────
# 3. BUILD HEATMAP ARRAYS  (n_models × 2blocks × 2ratios × 5costs × 5offers)
# ─────────────────────────────────────────────────────────────────────────────
def build_heatmap(vec_100, cond_rows, costs, offers, blocks, ratios):
    """vec_100: 100-element condition vector → (2,2,5,5) array [blk,rat,cost,offer]"""
    arr = np.full((2, 2, 5, 5), np.nan)
    for idx, row in enumerate(cond_rows):
        blk, o, c, rat = row
        bi  = blocks.index(blk)
        ri  = ratios.index(rat)
        ci  = costs.index(int(c))
        oi  = offers.index(int(o))
        arr[bi, ri, ci, oi] = vec_100[idx]
    return arr

h_heatmap  = build_heatmap(h_vec, cond_rows, COSTS, OFFERS, BLOCKS, RATIOS)
ai_heatmap = np.stack([build_heatmap(ai_tpp_mean[i], cond_rows, COSTS, OFFERS, BLOCKS, RATIOS)
                       for i in range(n_tpp_models)], axis=0)  # (n_models,2,2,5,5)

# ─────────────────────────────────────────────────────────────────────────────
# 4. TPP COUNTRY-LEVEL MEANS (for Panel B country bar chart)
# ─────────────────────────────────────────────────────────────────────────────
# Human TPP country means (participant level)
hdf_tpp = hdf.copy()
h_tpp_part = hdf_tpp.groupby(['subid','country'])['action'].mean().reset_index()
h_tpp_country_mean = np.array([h_tpp_part[h_tpp_part['country']==c]['action'].mean()
                                for c in COUNTRIES])
h_tpp_country_se   = np.array([h_tpp_part[h_tpp_part['country']==c]['action'].sem()
                                for c in COUNTRIES])

# AI TPP country means (per model, averaged over all conditions and personas)
# persona_results stores per-persona 100-condition vectors; we need per-country averages
# Re-collect country info from data_ai struct
ai_tpp_country_mean = np.full((n_tpp_models, len(COUNTRIES)), np.nan)
ai_tpp_country_se   = np.full((n_tpp_models, len(COUNTRIES)), np.nan)

# Build per-model per-country lists
for i, lbl in enumerate(tpp_model_keys):
    ctry_means = {c: [] for c in COUNTRIES}
    for s in amat['data_ai']:
        slbl = f'{s.model}_{s.Method}'
        if slbl != lbl:
            continue
        ctry = str(s.CountryOfBirth).strip()
        if ctry not in ctry_means:
            continue
        act = pd.to_numeric(np.array(s.action).flatten(), errors='coerce')
        ctry_means[ctry].append(np.nanmean(act))
    for j, c in enumerate(COUNTRIES):
        vals = ctry_means[c]
        if vals:
            ai_tpp_country_mean[i, j] = np.nanmean(vals)
            ai_tpp_country_se[i, j]   = np.nanstd(vals) / np.sqrt(len(vals)) if len(vals) > 1 else 0

print(f'\n  TPP country means (human): {np.round(h_tpp_country_mean, 3)}')

# ─────────────────────────────────────────────────────────────────────────────
# 5. PER-PERSONA vectors for scatter plot (stack all personas for top models)
# ─────────────────────────────────────────────────────────────────────────────
# For scatter: use averaged vector (already in ai_tpp_mean)

# ─────────────────────────────────────────────────────────────────────────────
# 5. SAVE .mat files
# ─────────────────────────────────────────────────────────────────────────────

# DG
sio.savemat(f'{OUT}/dg_data.mat', {
    'countries':    np.array(COUNTRIES, dtype=object),
    'h_dg_mean':   h_dg_mean,
    'h_dg_se':     h_dg_se,
    'ai_dg_mean':  ai_dg_mean,           # (n_models × n_countries)
    'ai_dg_mae':   ai_dg_mae,            # (n_models,)
    'ai_dg_names': np.array(ai_dg_names, dtype=object),
    'top3_dg_idx': top3_dg_idx + 1,      # 1-indexed for MATLAB
    'n_dg_models': n_dg_models,
}, do_compression=True)

# TPP
sio.savemat(f'{OUT}/tpp_data.mat', {
    'h_vec':        h_vec,               # (100,)
    'h_heatmap':    h_heatmap,           # (2,2,5,5)  [blk,rat,cost,offer]
    'cond_matrix':  cond_matrix,         # (100,4)    [block,offer,cost,ratio]
    'ai_tpp_mean':  ai_tpp_mean,         # (n_models,100)
    'ai_heatmap':   ai_heatmap,          # (n_models,2,2,5,5)
    'ai_tpp_r':     ai_tpp_r,
    'ai_tpp_mae':   ai_tpp_mae,
    'ai_tpp_names': np.array(ai_tpp_names, dtype=object),
    'top3_tpp_idx': top3_tpp_idx + 1,   # 1-indexed for MATLAB
    'n_tpp_models': n_tpp_models,
    'offers':       np.array(OFFERS),
    'costs':        np.array(COSTS),
    'blocks':       np.array(BLOCKS),
    'ratios':       np.array(RATIOS),
    # Country-level TPP means (for Panel B bar chart)
    'countries':              np.array(COUNTRIES, dtype=object),
    'h_tpp_country_mean':     h_tpp_country_mean,   # (8,)
    'h_tpp_country_se':       h_tpp_country_se,     # (8,)
    'ai_tpp_country_mean':    ai_tpp_country_mean,  # (n_models,8)
    'ai_tpp_country_se':      ai_tpp_country_se,    # (n_models,8)
}, do_compression=True)

print(f'\nSaved to {OUT}/')
print(f'  dg_data.mat  — {n_dg_models} DG models, {len(COUNTRIES)} countries')
print(f'  tpp_data.mat — {n_tpp_models} TPP models, 100 conditions')
print(f'\nTop 3 DG  (indices in MATLAB 1-based): {top3_dg_idx+1}')
print(f'Top 3 TPP (indices in MATLAB 1-based): {top3_tpp_idx+1}')
print('Done.')
