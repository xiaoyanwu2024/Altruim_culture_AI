% =============================================================================
% Author : Xiaoyan Wu
% Date   : June 2026
% Description: Figure 4 panel A — best-fitting model complexity (probability-elicitation models only)
% =============================================================================

% panel_A_prob_only.m
% =========================================================================
% Panel A — Best-fitting model complexity: Human + prob-only AI models
% Excludes all logprobs variants ([lp])
% =========================================================================

clear; clc;

DATA_DIR = fullfile(fileparts(mfilename('fullpath')), 'data');
load(fullfile(DATA_DIR, 'fig4_data.mat'));

% ── USER PARAMETERS ───────────────────────────────────────────────────────
FONT_NAME        = 'Helvetica';
FONT_SIZE_AX     = 9;
FONT_SIZE_LB     = 10.5;
FONT_SIZE_TT     = 12;
FONT_SIZE_PCT    = 7.5;
FONT_SIZE_LEG    = 8;

BAR_HEIGHT       = 0.62;
HUMAN_ROW_ALPHA  = 0.92;
AI_ROW_ALPHA     = 0.80;
SHOW_PCT_LABELS  = true;
PCT_MIN_WIDTH    = 0.05;
SEPARATOR_COLOR  = [0.80 0.30 0.20];

figW = 17;
figH = 10;

% M1→M8 colors: green family (light → dark) then orange
CMPLX_COLS = [
    0.84  0.91  0.84;   % M1
    0.70  0.85  0.70;   % M2
    0.51  0.78  0.51;   % M3
    0.30  0.69  0.31;   % M4
    1.00  0.88  0.70;   % M5
    1.00  0.72  0.30;   % M6
    0.99  0.55  0.13;   % M7
    0.81  0.26  0.00;   % M8
];

% ── Override model labels with full cumulative names ──────────────────────
model_labels = {
    'M1  Baseline';
    'M2  SI';
    'M3  SI + SCI';
    'M4  SI + SCI + VCI';
    'M5  SI + SCI + VCI + EC';
    'M6  SI + SCI + VCI + EC + RP';
    'M7  SI + SCI + VCI + EC + RP + II';
    'M8  SI + SCI + VCI + EC + RP + II';
};

% ── Smart filter: remove [lp] only when a prob version also exists ─────────
% (keeps DeepSeek-V4-Flash which is logprobs-only)
base_fn = @(s) strtrim(strrep(s, ' [lp]', ''));
keep = true(length(row_names), 1);
for i = 1:length(row_names)
    if contains(row_names{i}, '[lp]')
        base = base_fn(row_names{i});
        prob_exists = any(cellfun(@(s) strcmp(s, base), row_names));
        if prob_exists
            keep(i) = false;
        end
    end
end

row_names_f   = cellfun(base_fn, row_names(keep), 'UniformOutput', false);  % strip [lp] from display
best_counts_f = best_counts_mat(keep, :);
is_human_f    = is_human(keep);

% ── Data preparation ──────────────────────────────────────────────────────
n_rows   = size(best_counts_f, 1);   % Human + 8 prob AI = 9
n_models = size(best_counts_f, 2);   % 8 cognitive models

total_per_row = sum(best_counts_f, 2);
pct_mat       = best_counts_f ./ max(total_per_row, 1);

y_pos = (n_rows:-1:1)';   % Human at top (y=9), Mistral at bottom (y=1)

% ── Figure ────────────────────────────────────────────────────────────────
fig = figure('Units','centimeters','Position',[2 2 figW figH], ...
             'Color','white','PaperPositionMode','auto');

ax = axes(fig,'Units','normalized','Position',[0.22 0.10 0.58 0.82], ...
          'FontName',FONT_NAME,'FontSize',FONT_SIZE_AX,'Box','off');
hold(ax,'on');

for row = 1:n_rows
    left = 0;
    alpha_val = HUMAN_ROW_ALPHA * is_human_f(row) + AI_ROW_ALPHA * (1 - is_human_f(row));

    for mi = 1:n_models
        w = pct_mat(row, mi);
        if w < 1e-6, continue; end

        col = CMPLX_COLS(mi, :);
        yb  = y_pos(row) - BAR_HEIGHT/2;
        yt  = y_pos(row) + BAR_HEIGHT/2;
        patch(ax, [left, left+w, left+w, left], [yb, yb, yt, yt], col, ...
              'EdgeColor','white', 'LineWidth', 0.4, 'FaceAlpha', alpha_val);

        if SHOW_PCT_LABELS && w >= PCT_MIN_WIDTH
            x_center = left + w/2;
            pct_str  = sprintf('%d%%', round(w * 100));
            text(ax, x_center, y_pos(row), pct_str, ...
                 'HorizontalAlignment','center','VerticalAlignment','middle', ...
                 'FontSize',FONT_SIZE_PCT,'FontName',FONT_NAME, ...
                 'Color','white','FontWeight','bold');
        end
        left = left + w;
    end
end

% Human / AI separator
yline(ax, y_pos(1) - 0.55, '-', 'Color', SEPARATOR_COLOR, ...
      'LineWidth', 1.8, 'Alpha', 0.85);

% Company group dividers
% Rows (top→bottom): Human(1), GPT-4o(2), GPT-4o-mini(3), GPT-5.4(4),
%   GPT-5.4-mini(5), Gemini-2.5(6), Gemini-3.5(7),
%   DeepSeek-V4-Flash(8), DeepSeek-V4-Pro(9), Mistral(10)
% Dividers: after OpenAI (row5→6), after Gemini (row7→8), after DeepSeek (row9→10)
DIV_Y = [y_pos(5) - 0.55, y_pos(7) - 0.55, y_pos(9) - 0.55];
for d = 1:length(DIV_Y)
    yline(ax, DIV_Y(d), ':', 'Color', [0.65 0.65 0.65], 'LineWidth', 0.8, 'Alpha', 0.7);
end

% ── Y-axis labels ─────────────────────────────────────────────────────────
ax.YTick = [];
for row = 1:n_rows
    lbl = row_names_f{row};
    if strcmp(lbl, 'Human')
        fc = SEPARATOR_COLOR; fw = 'bold';
    else
        fc = [0.20 0.20 0.20]; fw = 'normal';
    end
    text(ax, -0.01, y_pos(row), lbl, ...
         'HorizontalAlignment','right','VerticalAlignment','middle', ...
         'FontSize',FONT_SIZE_AX,'FontName',FONT_NAME, ...
         'Color',fc,'FontWeight',fw,'Interpreter','none');
end

% ── Axes formatting ───────────────────────────────────────────────────────
xlim(ax, [0, 1.0]);
ylim(ax, [0.3, n_rows + 0.7]);
ax.XTick      = 0:0.2:1.0;
ax.XTickLabel = {'0%','20%','40%','60%','80%','100%'};
ax.XGrid      = 'on';
ax.GridLineStyle = '--';
ax.GridAlpha  = 0.35;
ax.LineWidth  = 0.8;
ax.Layer      = 'top';

xlabel(ax,'Proportion of subjects selecting each model', ...
       'FontSize',FONT_SIZE_LB,'FontName',FONT_NAME);
title(ax,'Best-fitting model complexity (Human vs. AI models)', ...
      'FontSize',FONT_SIZE_TT,'FontName',FONT_NAME,'FontWeight','bold', ...
      'HorizontalAlignment','left','Units','normalized','Position',[0, 1.04, 0]);

% ── Legend: vertical, right side ──────────────────────────────────────────
leg_handles = gobjects(n_models, 1);
for mi = 1:n_models
    leg_handles(mi) = patch(ax, [0 0 0 0], [-1 -1 -1 -1], CMPLX_COLS(mi,:), ...
                            'EdgeColor','white','FaceAlpha',0.88, ...
                            'DisplayName', model_labels{mi});
end
lg = legend(ax, leg_handles, ...
            'Location','eastoutside', ...
            'Orientation','vertical', ...
            'FontSize',FONT_SIZE_LEG, ...
            'FontName',FONT_NAME,'Box','off','NumColumns',1);
lg.Title.String   = 'Winning model';
lg.Title.FontSize = FONT_SIZE_LEG;

% ── Save ──────────────────────────────────────────────────────────────────
OUT_DIR = fullfile(fileparts(mfilename('fullpath')), 'output');
if ~exist(OUT_DIR,'dir'), mkdir(OUT_DIR); end
print(fig, fullfile(OUT_DIR,'panel_A_prob_only.pdf'), '-dpdf', '-r300', '-bestfit');
print(fig, fullfile(OUT_DIR,'panel_A_prob_only.png'), '-dpng', '-r300');
fprintf('Saved panel_A_prob_only.pdf/.png\n');
