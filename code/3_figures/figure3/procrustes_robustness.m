% =============================================================================
% Author : Xiaoyan Wu
% Date   : June 2026
% Description: Figure 3 — Procrustes robustness analysis across all TPI models
% =============================================================================

% procrustes_robustness.m
% =========================================================================
% Robustness check for RSA Mantel test:
%   Procrustes correlation (permutation-based) between human and AI
%   country similarity structures.
%
% Output: console table + saved results struct
% =========================================================================

clear; clc;

DATA_DIR = fullfile(fileparts(mfilename('fullpath')), 'data');
load(fullfile(DATA_DIR, 'fig3_data.mat'));

N_PERM   = 5000;
N_DIM    = 2;
N_CTRY   = 8;
n_ai     = double(n_ai_delta);

clean_name = @(s) strrep(strrep(s,'_prob',''),'_logprobs',' [lp]');

% ── Human MDS configuration (reference) ───────────────────────────────────
X_human = rdm2mds(h_rdm, N_DIM);

% ── Procrustes per model ───────────────────────────────────────────────────
proc_d    = zeros(1, n_ai);
proc_p    = zeros(1, n_ai);
proc_null = zeros(n_ai, N_PERM);

rng(42);

fprintf('\n%s\n', repmat('=',1,65));
fprintf('Procrustes robustness check (N_perm = %d, MDS dim = %d)\n', N_PERM, N_DIM);
fprintf('%s\n', repmat('-',1,65));
fprintf('%-35s  %8s  %8s  %6s\n','Model','Proc. d','perm_p','Sig.');
fprintf('%s\n', repmat('-',1,65));

for m = 1:n_ai
    ai_rdm_m = squeeze(ai_rdm(m,:,:));
    X_ai     = rdm2mds(ai_rdm_m, N_DIM);

    [d_obs, ~, ~] = procrustes(X_human, X_ai, 'Scaling', true, 'Reflection', false);
    proc_d(m) = d_obs;

    null_d = zeros(1, N_PERM);
    for k = 1:N_PERM
        perm_idx   = randperm(N_CTRY);
        rdm_perm   = ai_rdm_m(perm_idx, perm_idx);
        X_perm     = rdm2mds(rdm_perm, N_DIM);
        [dp, ~, ~] = procrustes(X_human, X_perm, 'Scaling', true, 'Reflection', false);
        null_d(k)  = dp;
    end
    proc_null(m,:) = null_d;

    proc_p(m) = mean(null_d <= d_obs);

    if proc_p(m) < 0.001, sig = '***';
    elseif proc_p(m) < 0.01, sig = '**';
    elseif proc_p(m) < 0.05, sig = '*';
    else, sig = 'ns'; end

    fprintf('  %-33s  %8.4f  %8.4f  %6s\n', ...
        clean_name(delta_ai_labels{m}), d_obs, proc_p(m), sig);
end

fprintf('%s\n', repmat('=',1,65));
fprintf('Note: Procrustes d = 0 (perfect fit) to 1 (no fit).\n');
fprintf('p-value: proportion of permuted d <= observed d.\n\n');

% ── Save results ───────────────────────────────────────────────────────────
OUT_DIR = fullfile(fileparts(mfilename('fullpath')), 'output');
if ~exist(OUT_DIR,'dir'), mkdir(OUT_DIR); end

procrustes_results.labels    = delta_ai_labels;
procrustes_results.proc_d    = proc_d;
procrustes_results.proc_p    = proc_p;
procrustes_results.proc_null = proc_null;
procrustes_results.N_PERM    = N_PERM;
procrustes_results.N_DIM     = N_DIM;

save(fullfile(OUT_DIR,'procrustes_results.mat'), 'procrustes_results');
fprintf('Results saved to output/procrustes_results.mat\n');

% =========================================================================
% Local function — must be at end of file
% =========================================================================
function X = rdm2mds(rdm, ndim)
    D = sqrt(2 * max(1 - rdm, 0));
    D = (D + D') / 2;
    D(1:size(D,1)+1:end) = 0;
    X = cmdscale(D, ndim);
end
