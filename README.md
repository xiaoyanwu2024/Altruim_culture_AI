# Data and Code - Behavioral Fidelity Without Cultural Fidelity

**Author:** Xiaoyan Wu  
**Affiliation:** Department of Adult Psychiatry and Psychotherapy, University of Zurich  
**Contact:** Xiaoyan.psych@gmail.com  
**Date:** June 2026

---

## Overview

This repository contains data and analysis code for the manuscript:

> **"Behavioral Fidelity Without Cultural Fidelity: How Large Language Models Simulate Cross-Cultural Prosocial Behavior"**  
> Xiaoyan Wu, *PNAS* (submitted 2026)

The study administered a **Dictator Game (DG)** and a **Third-Party Intervention (TPI)** task to 875 human participants across 8 countries (Chile, Greece, Italy, Mexico, Poland, Portugal, South Africa, Spain), and to a representative set of 14 state-of-the-art LLMs from four developers (OpenAI, Google, DeepSeek, Mistral) on the DG, and 9 models on the TPI.

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
│   │   └── raw/                        Raw AI DG response CSVs (per-model, timestamped)
│   ├── model_fitting/
│   │   ├── data_ai_all.mat             All AI subjects formatted for model fitting
│   │   ├── data_human_filtered.mat     Human subjects formatted for model fitting
│   │   ├── results_human_aligned.mat   Human model fitting results (M1–M8)
│   │   ├── results_ai.mat              AI model fitting results (M1–M8, all models)
│   │   └── Results_<Model>.mat         Per-model fitting results (one file per LLM)
│   └── figure_data/
│       ├── fig3_data.mat               RSA / cultural similarity matrices (Figure 3)
│       ├── mds_data.mat                MDS configurations (Figure 3)
│       ├── procrustes_results.mat      Procrustes analysis results (Figure 3)
│       ├── fig4_data.mat               Model complexity distributions (Figure 4)
│       └── fig5_data.mat               M8 parameter distributions (Figure 4/S5)
│
├── code/
│   ├── 1_data_collection/
│   │   ├── tpi_collect_*.py            TPI data collection (one script per model/provider)
│   │   └── dg_collect_*.py             DG data collection (one script per model/provider)
│   ├── 2_model_fitting/
│   │   ├── FittingModelAI.m            Main fitting script for AI TPI data
│   │   ├── IndividualFitting.m         Individual-level parameter estimation
│   │   ├── ForModelComparison.m        AICc/BIC computation for model comparison
│   │   ├── ModelComparision.m          Bayesian model comparison (PXP)
│   │   ├── optimizeAllsubs.m           Batch optimisation across all subjects
│   │   └── m1.m – m9.m                Motive-cocktail model definitions (M1–M9)
│   ├── 3_figures/
│   │   ├── figure2/                    Figure 2 panels (A: DG bar, B: TPI bar, C: heatmap, D: scatter)
│   │   ├── figure3/                    Figure 3 panels (A–F: RSA, MDS, Procrustes, Mantel)
│   │   ├── figure4/                    Figure 4 panels (A: model complexity, B: radar, kappa, spider)
│   │   ├── figureS1/                   Supplementary Figure S1 — DG bar charts, all 14 AI models
│   │   ├── figureS2/                   Supplementary Figure S2 — TPI bar + scatter, all 9 AI models
│   │   ├── figureS3/                   Supplementary Figure S3 — TPI heatmaps, all 9 AI models
│   │   ├── figureS4/                   Supplementary Figure S4 — TPI directional sensitivity
│   │   └── figureS5/                   Supplementary Figure S5 — M8 parameter distributions
│   └── 4_preprocessing/
│       ├── fig2_preprocess_data.py     Generates dg_data.mat and tpp_data.mat
│       ├── fig3_preprocess_data.py     Generates fig3_data.mat (RSA / cultural similarity matrices)
│       ├── fig3_preprocess_mds.py      Generates mds_data.mat (MDS configurations)
│       ├── fig3_permutation_test_delta.py  Mantel permutation tests for cross-cultural delta matrices
│       ├── fig4_preprocess_data.py     Generates fig4_data.mat (model complexity)
│       ├── fig4_preprocess_spider.py   Generates fig5_data.mat (M8 parameter distributions)
│       ├── fig4_analyze_m8_params.py   Extracts and summarises M8 parameters (Human vs AI)
│       └── model_fitting.py            Python MLE fitting for AI TPI data
│
└── README.md
```

---

## Recommended Workflow

```
1. Raw API responses (data/ai_tpi/raw/,  data/ai_dg/raw/)
        ↓  [code/1_data_collection/]
2. Preprocessing (code/4_preprocessing/fig2_preprocess_data.py, etc.)
        ↓
3. Processed .mat files (data/ai_tpi/tpp_data.mat, data/ai_dg/dg_data.mat,
                         data/figure_data/*.mat)
        ↓  [code/2_model_fitting/FittingModelAI.m]
4. Model fitting results (data/model_fitting/results_ai.mat, Results_<Model>.mat)
        ↓  [code/3_figures/figure*/panel_*.m]
5. Figures (Figure 2–4, Supplementary Figures S1–S5)
```

---

## Software Requirements

- **MATLAB** R2021b or later (Statistics and Machine Learning Toolbox required)
- **Python** 3.9+ with packages: `numpy`, `pandas`, `scipy`, `openai`, `anthropic`, `google-generativeai`, `mistralai`

---

## Data Description

### human_data.csv
Behavioral data from 875 human participants across 8 countries. Each row is one trial. Key columns: `country`, `gender`, `scenario` (Help/Punish), `inequality`, `ratio`, `cost`, `intervene` (0/1).

### tpp_data.mat
Processed TPI data for MATLAB. Key variables: `h_tpp_cond_mean` (human condition means, 100×1), `ai_tpp_mean` (AI condition means, n_models×100), `ai_tpp_r` (Pearson r with human), `ai_tpp_names`, `countries`, `h_heatmap`, `ai_heatmap`.

### dg_data.mat
Processed DG data. Key variables: `h_dg_mean` (human mean allocation by country), `ai_dg_mean` (AI mean allocation by country and model), `ai_dg_mae`, `countries`, `ai_dg_names`.

### data_ai_all.mat / data_human_filtered.mat
Subject-level trial-by-trial data formatted for model fitting (M1–M8). Used directly by `FittingModelAI.m`.

### Results_<Model>.mat
Per-model fitting results containing best-fitting model index, AICc, BIC, and parameter estimates for each subject.

### results_ai.mat / results_human_aligned.mat
Aggregated fitting results across all AI models / human participants. Used by Figure 4 and S5 scripts.

---

## Model Description (M1–M8)

The motive-cocktail framework decomposes TPI decisions into up to 8 motivational components:

| Model | Added motive | Parameters |
|-------|-------------|------------|
| M1 | Baseline (fixed probability) | γ |
| M2 | Self-interest (SI) | + α |
| M3 | Shared cost inequality (SCI) | + β |
| M4 | Victim cost inequality (VCI) | + λ |
| M5 | Empathy / cost sensitivity (EC) | + ω |
| M6 | Relative payoff (RP) | + κ |
| M7 | Inequality aversion (II) | + η_no |
| M8 | Full model | + η_yes |

---

## Citation

If you use this data or code, please cite:

> Wu, X. (2026). Behavioral Fidelity Without Cultural Fidelity: How Large Language Models Simulate Cross-Cultural Prosocial Behavior. *Proceedings of the National Academy of Sciences*.

---

## License

Code is released under the MIT License. Data are released under CC BY 4.0.
