% =============================================================================
% Author : Xiaoyan Wu
% Date   : June 2026
% Description: Figure 4 — distribution of kappa (inequality reversal preference) parameter
% =============================================================================

% panel_kappa_dist.m
% =========================================================================
% Distribution of kappa (κ, reciprocity/punishment parameter) from M8
% Human vs. GPT-4o-mini [lp] vs. GPT-5.4 vs. Gemini-3.5-Flash
%
% Human: kernel density estimate (N=875)
% AI models: individual points + mean±SD (N=16 each)
% =========================================================================

clear; clc;

% ── Load raw data ─────────────────────────────────────────────────────────
MFIT = '/Users/wuxiaoyan/Desktop/TPP_culture_AI/ModelFitting';

hr    = load(fullfile(MFIT,'results_human_aligned.mat'));
hr    = hr.results_aligned;
ar    = load(fullfile(MFIT,'results_ai.mat'));
ar    = ar.results_ai;
tmp   = load(fullfile(MFIT,'data_ai_all.mat'));
adata = tmp.data_ai;

% kappa = column 6 (index 5 in Python → col 6 in MATLAB)
h_kappa = hr(8).x(:, 6);  % (875,1)

% AI kappa from M8 (x is stored as matrix or cell — handle both)
n_ai     = length(adata);
ai_kappa = zeros(n_ai,1);
x_all    = ar(8).x;
for si = 1:n_ai
    if iscell(x_all)
        row = x_all{si};
        ai_kappa(si) = row(6);
    else
        ai_kappa(si) = x_all(si, 6);
    end
end

% Group indices
models  = {adata.model};
methods = {adata.Method};

idx_54   = find(strcmp(models,'GPT-5.4')          & strcmp(methods,'prob'));
idx_g35  = find(strcmp(models,'Gemini-3.5-Flash') & strcmp(methods,'prob'));

kappa_54  = ai_kappa(idx_54);
kappa_g35 = ai_kappa(idx_g35);

% ── FDR correction across all 16 comparisons (8 params × 2 models) ────────
x_all_params = ar(8).x;
h_all_params = hr(8).x(:, 1:8);
g54_all = x_all_params(idx_54,  1:8);
g35_all = x_all_params(idx_g35, 1:8);

p_raw_all = zeros(16,1);
for k = 1:8
    p_raw_all(2*k-1) = ranksum(g54_all(:,k), h_all_params(:,k));
    p_raw_all(2*k)   = ranksum(g35_all(:,k), h_all_params(:,k));
end
n_tests = 16;
[~, sort_idx] = sort(p_raw_all);
p_fdr_all = zeros(n_tests,1);
for i = 1:n_tests
    p_fdr_all(sort_idx(i)) = min(p_raw_all(sort_idx(i)) * n_tests / i, 1);
end
for i = n_tests-1:-1:1
    p_fdr_all(sort_idx(i)) = min(p_fdr_all(sort_idx(i)), p_fdr_all(sort_idx(i+1)));
end
% kappa is param 6 → indices 11 (GPT-5.4) and 12 (Gemini)
p_fdr_kappa = [p_fdr_all(11); p_fdr_all(12)];

% ── Colors ────────────────────────────────────────────────────────────────
FONT_NAME = 'Helvetica';
C_HUM   = [0.30 0.30 0.30];
C_GPT54 = [0.45 0.35 0.75];
C_GEM35 = [0.10 0.40 0.82];

figW = 13;
figH = 7;

% ── Figure ────────────────────────────────────────────────────────────────
fig = figure('Units','centimeters','Position',[2 3 figW figH], ...
             'Color','white','PaperPositionMode','auto');

ax = axes(fig,'Units','normalized','Position',[0.12 0.16 0.82 0.76], ...
          'FontName',FONT_NAME,'FontSize',9,'Box','off');
hold(ax,'on');

% ── Human KDE ─────────────────────────────────────────────────────────────
x_grid = linspace(-10.5, 10.5, 500);
bw = 1.2;   % bandwidth
kde_h = zeros(size(x_grid));
for xi = 1:length(x_grid)
    kde_h(xi) = mean(normpdf(x_grid(xi), h_kappa, bw));
end
% Scale KDE so peak ~ 1 (relative density)
kde_h = kde_h / max(kde_h);

fill(ax, [x_grid, fliplr(x_grid)], [kde_h, zeros(size(kde_h))], ...
     C_HUM, 'FaceAlpha',0.15, 'EdgeColor','none');
plot(ax, x_grid, kde_h, '-', 'Color',C_HUM, 'LineWidth',2.0);

% Human mean line
xline(ax, mean(h_kappa), '--', 'Color',C_HUM, 'LineWidth',1.2, 'Alpha',0.7);

% ── AI: jittered dots + mean±SD ───────────────────────────────────────────
AI_GROUPS = {kappa_54, kappa_g35};
AI_COLORS = {C_GPT54, C_GEM35};
AI_LABELS = {'GPT-5.4','Gemini-3.5-Flash'};
Y_LEVELS  = [0.72, 0.52];

rng(42);
for gi = 1:2
    kv  = AI_GROUPS{gi};
    col = AI_COLORS{gi};
    yl  = Y_LEVELS(gi);

    % FDR-corrected p-value for kappa
    pv = p_fdr_kappa(gi);
    if     pv < 0.001, sig_str = '***';
    elseif pv < 0.01,  sig_str = '**';
    elseif pv < 0.05,  sig_str = '*';
    else,               sig_str = 'ns';
    end

    % Mean ± SD bar
    m  = mean(kv);
    sd = std(kv);
    plot(ax, [m-sd, m+sd], [yl yl], '-', 'Color',col, 'LineWidth',1.2);
    % End caps
    plot(ax, [m-sd, m-sd], [yl-0.015, yl+0.015], '-', 'Color',col, 'LineWidth',1.0);
    plot(ax, [m+sd, m+sd], [yl-0.015, yl+0.015], '-', 'Color',col, 'LineWidth',1.0);
    plot(ax, m, yl, 'o', 'Color',col, 'MarkerFaceColor',col, ...
         'MarkerSize',5, 'MarkerEdgeColor','white','LineWidth',0.8);

    % Significance marker above the bar
    text(ax, m, yl+0.01, sig_str, ...
         'HorizontalAlignment','center','VerticalAlignment','bottom', ...
         'FontSize',15,'FontName',FONT_NAME,'Color',col,'FontWeight','bold');
end

% ── Reference line at κ=0 ─────────────────────────────────────────────────
xline(ax, 0, ':', 'Color',[0.5 0.5 0.5], 'LineWidth',1.0, 'Alpha',0.8);
text(ax, 0.2, 0.97, '\kappa = 0', 'FontSize',8,'FontName',FONT_NAME, ...
     'Color',[0.45 0.45 0.45],'Interpreter','tex');

% ── Human mean annotation ─────────────────────────────────────────────────
text(ax, mean(h_kappa)+0.2, 1.02, ...
     sprintf('Human mean = %.2f', mean(h_kappa)), ...
     'FontSize',8,'FontName',FONT_NAME,'Color',C_HUM);

% ── Axes ──────────────────────────────────────────────────────────────────
xlim(ax, [-10.5, 10.5]);
ylim(ax, [0, 1.12]);
ax.XTick = -10:2:10;
ax.YTick = [];
ax.XGrid = 'on';
ax.GridLineStyle = ':';
ax.GridAlpha = 0.30;
ax.LineWidth = 0.8;

xlabel(ax, '\kappa', ...
       'FontSize',13,'FontName',FONT_NAME,'Interpreter','tex');
ylabel(ax, 'Density', 'FontSize',10,'FontName',FONT_NAME);

title(ax, 'Inequality reversal preference', ...
      'FontSize',11,'FontName',FONT_NAME,'FontWeight','bold', ...
      'Interpreter','tex','Color','k');

% ── Legend (right side, all three) ────────────────────────────────────────
ax_leg = axes(fig,'Units','normalized','Position',[0.88 0.2 0.01 0.6],'Visible','off');
hold(ax_leg,'on');
h_hum = fill(ax_leg,[0 0 0 0],[0 0 0 0],C_HUM,'FaceAlpha',0.15,'EdgeColor',C_HUM,'LineWidth',1.5);
h_54  = plot(ax_leg, NaN, NaN, '-o','Color',C_GPT54,'LineWidth',1.2,'MarkerSize',5,'MarkerFaceColor',C_GPT54,'MarkerEdgeColor','white');
h_g35 = plot(ax_leg, NaN, NaN, '-o','Color',C_GEM35,'LineWidth',1.2,'MarkerSize',5,'MarkerFaceColor',C_GEM35,'MarkerEdgeColor','white');
legend(ax_leg, [h_hum, h_54, h_g35], ...
       {'Human', 'GPT-5.4', 'Gemini-3.5-Flash'}, ...
       'Orientation','vertical','Location','east', ...
       'FontSize',8.5,'FontName',FONT_NAME,'Box','off');

% ── Save ──────────────────────────────────────────────────────────────────
OUT_DIR = fullfile(fileparts(mfilename('fullpath')), 'output');
if ~exist(OUT_DIR,'dir'), mkdir(OUT_DIR); end
print(fig, fullfile(OUT_DIR,'panel_kappa_dist.pdf'), '-dpdf', '-r300', '-bestfit');
print(fig, fullfile(OUT_DIR,'panel_kappa_dist.png'), '-dpng', '-r300');
fprintf('Saved panel_kappa_dist.pdf/.png\n');
