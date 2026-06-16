% =============================================================================
% Author : Xiaoyan Wu
% Date   : June 2026
% Description: Supplementary Figure S1 — Dictator Game bar charts for Human
%              and all 14 AI models by country
% =============================================================================

% SI_A_DG_bar_allmodels.m
% =========================================================================
% Supplementary Fig S1: Dictator Game — Human + all 14 AI models
% Format: Arial, 10pt axis labels, 12pt panel letters (top-left)
% =========================================================================

clear; clc;

DATA_DIR = fullfile(fileparts(mfilename('fullpath')), 'data');
load(fullfile(DATA_DIR, 'dg_data.mat'));

% ── FORMAT PARAMETERS ─────────────────────────────────────────────────────
FONT_NAME       = 'Arial';
FONT_SIZE_AX    = 8;     % axis tick labels
FONT_SIZE_LB    = 8;     % axis labels & subplot titles
FONT_SIZE_PANEL = 9;     % panel letter (a, b, c ...)

BAR_W     = 0.65;
BAR_ALPHA = 0.88;
Y_MIN     = 0;
Y_MAX     = 30;
N_COLS    = 5;

% Country colors — ALPHABETICAL order
% Chile Greece Italy Mexico Poland Portugal South Africa Spain
COUNTRY_COLORS = [
    0.831  0.106  0.224;
    0.000  0.333  0.855;
    0.000  0.776  0.553;
    0.996  0.929  0.255;
    1.000  0.000  0.322;
    0.349  0.698  0.573;
    1.000  0.796  0.302;
    1.000  0.380  0.376;
];

clean_name = @(s) strtrim(strrep(strrep(s,' (logprobs)',''),'_logprobs',''));
panel_letter = @(i) char('a' + i - 1);

% ── Data prep ─────────────────────────────────────────────────────────────
n_ctry   = length(countries);
n_models = length(ai_dg_names);

[~, sort_ord]   = sort(h_dg_mean, 'descend');
h_sorted        = h_dg_mean(sort_ord);
ctry_sorted     = countries(sort_ord);
col_sorted      = COUNTRY_COLORS(sort_ord, :);

[mae_sorted, mae_idx] = sort(ai_dg_mae);
name_sorted = ai_dg_names(mae_idx);
mean_sorted = ai_dg_mean(mae_idx, :);

% Print MAE table
fprintf('\n=== SI_A: All DG models (sorted by MAE) ===\n');
for m = 1:n_models
    fprintf('  %2d. %-35s  MAE=%.3f\n', m, name_sorted{m}, mae_sorted(m));
end

% ── Layout ────────────────────────────────────────────────────────────────
N_TOTAL = n_models + 1;          % Human + 14 AI
N_ROWS  = ceil(N_TOTAL / N_COLS);
figW    = N_COLS * 7.2;          % cm per column
figH    = N_ROWS * 5.8;          % cm per row

fig = figure('Units','centimeters','Position',[1 1 figW figH], ...
             'Color','white','PaperPositionMode','auto');

for sp = 1:N_TOTAL
    ax = subplot(N_ROWS, N_COLS, sp);
    hold(ax, 'on');

    if sp == 1
        patch(ax, [0.5 n_ctry+0.5 n_ctry+0.5 0.5], ...
              [Y_MIN Y_MIN Y_MAX Y_MAX], [0.94 0.94 0.94], ...
              'FaceAlpha',0.5,'EdgeColor','none');
        vals = h_sorted;
        ttl  = 'Human (reference)';
    else
        m    = sp - 1;
        vals = mean_sorted(m, sort_ord);
        ttl  = sprintf('%s  MAE=%.2f', clean_name(name_sorted{m}), mae_sorted(m));
    end

    for c = 1:n_ctry
        bar(ax, c, vals(c), BAR_W, ...
            'FaceColor', col_sorted(c,:), 'EdgeColor','none', 'FaceAlpha', BAR_ALPHA);
    end

    ylim(ax, [Y_MIN Y_MAX]);
    xlim(ax, [0.3  n_ctry+0.7]);
    ax.XTick = 1:n_ctry;
    ax.XTickLabel = cellfun(@(s) s(1:min(3,end)), ctry_sorted, 'UniformOutput', false);
    ax.XTickLabelRotation = 0;
    ax.YTick = 0:5:Y_MAX;
    ax.YGrid = 'on';
    ax.GridColor     = [0.85 0.85 0.85];
    ax.GridLineStyle = '--';
    ax.GridAlpha     = 0.55;
    ax.Box           = 'off';
    ax.Layer         = 'top';
    ax.FontSize      = FONT_SIZE_AX;
    ax.FontName      = FONT_NAME;
    ax.XColor = 'k'; ax.YColor = 'k';
    ax.TickLabelInterpreter = 'none';

    title(ax, ttl, 'FontSize',FONT_SIZE_LB, 'FontName',FONT_NAME, ...
          'FontWeight','normal', 'Color','k', 'Interpreter','none');

    if mod(sp-1, N_COLS) == 0
        ylabel(ax, 'Allocation (tokens)', 'FontSize',FONT_SIZE_LB, 'FontName',FONT_NAME);
    end

    % Panel letter — top-left corner
    text(ax, 0.02, 0.97, panel_letter(sp), ...
         'Units','normalized', 'FontSize',FONT_SIZE_PANEL, ...
         'FontName',FONT_NAME, 'FontWeight','normal', ...
         'HorizontalAlignment','left', 'VerticalAlignment','top', 'Color','k');
end

% Legend in last empty slot (if any)
leg_sp = N_TOTAL + 1;
if leg_sp <= N_ROWS * N_COLS
    ax_lg = subplot(N_ROWS, N_COLS, leg_sp);
    axis(ax_lg, 'off');
    leg_h = gobjects(n_ctry, 1);
    for c = 1:n_ctry
        leg_h(c) = patch(ax_lg, NaN, NaN, col_sorted(c,:), ...
                         'EdgeColor','none','FaceAlpha',BAR_ALPHA);
    end
    legend(ax_lg, leg_h, ctry_sorted, ...
           'FontSize',FONT_SIZE_AX, 'FontName',FONT_NAME, ...
           'Box','off', 'NumColumns',1, 'Location','best');
end

% ── Save as PDF ────────────────────────────────────────────────────────────
OUT_DIR    = fullfile(fileparts(mfilename('fullpath')), 'output');
PAPER_DIR  = '/Users/wuxiaoyan/Desktop/TPP_culture_AI/manuscript/AI_Culture_altruism';
if ~exist(OUT_DIR,'dir'), mkdir(OUT_DIR); end

set(fig, 'PaperUnits','centimeters', 'PaperSize',[figW figH], ...
         'PaperPosition',[0 0 figW figH]);
print(fig, fullfile(OUT_DIR,'SI_A_DG_bar_allmodels.pdf'), '-dpdf', '-r300');
print(fig, fullfile(OUT_DIR,'SI_A_DG_bar_allmodels.png'), '-dpng', '-r300');
print(fig, fullfile(PAPER_DIR,'FigureS1.pdf'), '-dpdf', '-r300');
fprintf('Saved SI_A_DG_bar_allmodels.pdf/.png\nSaved FigureS1.pdf to manuscript folder\n');
