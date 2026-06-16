# Data and Code — Behavioral Fidelity Without Cultural Fidelity

**Author:** Xiaoyan Wu  
**Affiliation:** Department of Adult Psychiatry and Psychotherapy, University of Zurich  
**Contact:** Xiaoyan.psych@gmail.com  
**Date:** June 2026

---

## Overview

This repository contains data and analysis code for the manuscript:

> **"Behavioral Fidelity Without Cultural Fidelity: How Large Language Models Simulate Cross-Cultural Prosocial Behavior"**  
> Xiaoyan Wu, *PNAS* (submitted 2026)

The study administered a **Dictator Game (DG)** and a **Third-Party Intervention (TPI)** task to 875 human participants across 8 countries, and to a representative set of state-of-the-art LLMs from four developers (OpenAI, Google, DeepSeek, Mistral).

---

## Directory Structure

```
data_code/
│
├── data/
│   ├── human/
│   │   └── human_data.csv              Human behavioral data (875 participants, 8 countries)
│   ├── ai_tpi/
│   │   ├── tpp_data.mat                Processed TPI data (human + AI), used by Figure 2–4 scripts
│   │   └── raw/                        Raw AI TPI response CSVs (per-model, timestamped)
│   ├── ai_dg/
│   │   ├── dg_data.mat                 Processed DG data (human + AI), used by Figure 2 scripts
│   │   └── raw/                        Raw AI DG response CSVs
│   ├── model_fitting/
│   │   ├── data_ai_all.mat             All AI subjects formatted for model fitting
│   │   ├── data_human_filtered.mat     Human subjects formatted for model fitting
│   │   ├── results_human_aligned.mat   Human model fitting results (M1–M8)
│   │   ├── results_ai.mat              AI model fitting results (M1–M8, all models)
│   │   └── Results_<Model>.mat         Per-model fitting results
│   └── figure_data/
│       ├── fig3_data.mat               RSA / cultural similarity matrices (Figure 3)
│       ├── mds_data.mat                MDS configurations (Figure 3)
│       ├── fig4_data.mat               Model complexity distributions (Figure 4)
│       └── fig5_data.mat               M8 parameter distributions (Figure 4/5)
│
├── code/
│   ├── 1_data_collection/
│   │   ├── tpi_collect_*.py            TPI data collection scripts (one per model/provider)
│   │   └── dg_collect_*.py             DG data collection scripts (one per model/provider)
│   ├── 2_model_fitting/
│   │   ├── FittingModelAI.m            Main fitting script for AI data
│   │   ├── FittingModelExp2.m          Main fitting script for human data
│   │   ├── IndividualFitting.m         Individual-level parameter estimation
│   │   ├── ForModelComparison.m        AICc/BIC computation for model comparison
│   │   ├── ModelComparision.m          Bayesian model comparison (PXP)
│   │   ├── optimizeAllsubs.m           Batch optimisation across subjects
│   │   ├── optimizeParallelVersion.m   Parallel optimisation
│   │   └── m1.m – m9.m                Motive-cocktail model definitions (M1–M9)
│   ├── 3_figures/
│   │   ├── figure2/                    MATLAB scripts for Figure 2 panels (A–D)
│   │   ├── figure3/                    MATLAB scripts for Figure 3 panels (A–F, MDS, Procrustes)
│   │   ├── figure4/                    MATLAB scripts for Figure 4 panels (A–B, spider, kappa)
│   │   ├── figureS2/                   Supplementary Figure S2 (TPI bar + scatter, all models)
│   │   ├── figureS4/                   Supplementary Figure S4 (TPI directional sensitivity)
│   │   └── figureS5/                   Supplementary Figure S5 (M8 parameter distributions)
│   └── 4_preprocessing/
│       ├── fig2_preprocess_data.py     Generates dg_data.mat and tpp_data.mat
│       ├── fig3_preprocess_data.py     Generates fig3_data.mat (RSA matrices)
│       ├── fig3_preprocess_mds.py      Generates mds_data.mat (MDS configurations)
│       ├── fig3_permutation_test_delta.py  Mantel permutation tests
│       ├── fig4_preprocess_data.py     Generates fig4_data.mat (model complexity)
│       ├── fig4_preprocess_spider.py   Generates fig5_data.mat (parameter distributions)
│       ├── fig4_analyze_m8_params.py   Extracts M8 parameter summaries
│       └── model_fitting.py            Python model fitting (MLE for AI TPI data)
│
└── README.md
```

---

## Recommended Workflow

```
1. Raw API responses (data/ai_tpi/raw/, data/ai_dg/raw/)
        ↓
2. Preprocessing (code/4_preprocessing/fig2_preprocess_data.py)
        ↓
3. Processed .mat files (data/ai_tpi/tpp_data.mat, data/ai_dg/dg_data.mat)
        ↓
4. Model fitting (code/2_model_fitting/FittingModelAI.m)
        ↓
5. Figure generation (code/3_figures/figure*/panel_*.m)
```

---

## Software Requirements

- **MATLAB** R2021b or later (with Statistics and Machine Learning Toolbox)
- **Python** 3.9+ with: `numpy`, `pandas`, `scipy`, `openai`, `anthropic`, `google-generativeai`

---

## Data Description

### human_data.csv
Behavioral data from 875 human participants across 8 countries (Chile, Greece, Italy, Mexico, Poland, Portugal, South Africa, Spain). Each row is one trial. Key columns: `country`, `gender`, `scenario` (Help/Punish), `inequality`, `ratio`, `cost`, `intervene` (0/1).

### tpp_data.mat
Processed TPI data for MATLAB. Contains: `h_tpp_cond_mean` (human condition means, 100×1), `ai_tpp_mean` (AI condition means, n_models×100), `ai_tpp_r` (correlation with human), `countries`, `ai_tpp_names`, etc.

### dg_data.mat
Processed DG data. Contains allocation amounts by country and model.

### data_ai_all.mat / data_human_filtered.mat
Subject-level trial data formatted for model fitting (M1–M8).

---

## Citation

If you use this data or code, please cite:

> Wu, X. (2026). Behavioral Fidelity Without Cultural Fidelity: How Large Language Models Simulate Cross-Cultural Prosocial Behavior. *Proceedings of the National Academy of Sciences*.

---

## License

This code is released under the MIT License. Data are released under CC BY 4.0.
