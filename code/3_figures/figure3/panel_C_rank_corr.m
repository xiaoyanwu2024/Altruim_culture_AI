% =============================================================================
% Author : Xiaoyan Wu
% Date   : June 2026
% Description: Figure 3 panel C — rank correlation between AI and human cultural distances
% =============================================================================

% panel_C_rank_corr.m
% =========================================================================
% Panel C — Spearman rank correlation: do AI models replicate the
%           country rank ordering observed in human data?
%
% Design: Horizontal bar chart
%   - Each bar = Spearman ρ between model's 8-country means and human's
%   - Bars colored by significance (significant = colored; ns = gray)
%   - Vertical line at ρ=0 (random) and optional ρ=1 (perfect)
%   - Stars indicate significance
%   - Sorted by ρ descending
%
% HOW TO MODIFY:
%   FONT_NAME           : font family
%   FONT_SIZE_*         : sizes
%   COLOR_SIG           : RGB for significant bars
%   COLOR_NS            : RGB for non-significant bars
%   SHOW_ZERO_LINE      : true/false — draw ρ=0 reference
%   X_MIN / X_MAX       : x-axis range for ρ (default -1 to 1)
%   figW / figH         : figure size in cm
% =========================================================================

clear; clc;

DATA_DIR = fullfile(fileparts(mfilename('fullpath')), 'data');
load(fullfile(DATA_DIR, 'fig3_data.mat'));

% ── USER PARAMETERS ───────────────────────────────────────────────────────
FONT_NAME      = 'Helvetica';
FONT_SIZE_AX   = 9;
FONT_SIZE_LB   = 10.5;
FONT_SIZE_TT   = 12;
FONT_SIZE_STAR = 8.5;
FONT_SIZE_VAL  = 8;

% Significance threshold
ALPHA = 0.05;

% Colors
COLOR_NS       = [0.72 0.72 0.72];   % non-significant
COLOR_ZERO     = [0.30 0.30 0.30];   % ρ=0 reference line color
COLOR_PERFLINE = [0.60 0.60 0.60];   % ρ=1 dashed line

% Model color map (consistent across all Figure3 panels)
MODEL_COLOR_KEYS = {
    'GPT-4o_logprobs',            [0.65  0.20  0.10];
    'GPT-4o_prob',                [0.92  0.35  0.22];
    'GPT-4o-mini_logprobs',       [0.80  0.45  0.05];
    'GPT-4o-mini_prob',           [0.96  0.63  0.13];
    'GPT-5.4_prob',               [0.60  0.55  0.80];
    'GPT-5.4-mini_prob',          [0.96  0.63  0.72];
    'Gemini-2.5-Flash_prob',      [0.20  0.62  0.55];
    'Gemini-3.5-Flash_prob',      [0.94  0.78  0.19];
    'DeepSeek-V4-Flash_logprobs', [0.45  0.25  0.60];
    'DeepSeek-V4-Pro_logprobs',   [0.72  0.52  0.88];
    'DeepSeek-V4-Pro_prob',       [0.60  0.40  0.72];
    'Mistral-Small-4_prob',       [0.72  0.79  0.41];
};

SHOW_ZERO_LINE = true;
BAR_HEIGHT     = 0.60;   % bar thickness (0–1)

X_MIN = -1.0;
X_MAX =  1.0;

figW = 14;
figH = 9;

% ── Data — sort by ρ descending ───────────────────────────────────────────
n_models = length(rho_vals);

[~, sort_idx] = sort(rho_vals, 'descend');
rho_sorted   = rho_vals(sort_idx);
pval_sorted  = rho_pvals(sort_idx);
sig_sorted   = rho_sig(sort_idx);
label_sorted = rho_labels(sort_idx);

% Clean label strings
disp_labels = label_sorted;
for k = 1:n_models
    lbl = disp_labels{k};
    lbl = strrep(lbl, '_logprobs', ' [lp]');
    lbl = strrep(lbl, '_prob',     '');
    disp_labels{k} = lbl;
end

% Assign bar colors — model-specific palette; gray if ns
bar_cols = zeros(n_models, 3);
for k = 1:n_models
    key = label_sorted{k};
    ci  = find(strcmp(MODEL_COLOR_KEYS(:,1), key));
    if ~isempty(ci)
        model_col = MODEL_COLOR_KEYS{ci,2};
    else
        model_col = [0.5 0.5 0.5];
    end
    if pval_sorted(k) < ALPHA
        bar_cols(k,:) = model_col;
    else
        bar_cols(k,:) = COLOR_NS;
    end
end

y_pos = (n_models:-1:1)';   % k=1 (highest ρ) → top

% ── Figure ────────────────────────────────────────────────────────────────
fig = figure('Units','centimeters','Position',[2 2 figW figH], ...
             'Color','white','PaperPositionMode','auto');

ax = axes(fig,'Units','normalized','Position',[0.30 0.10 0.63 0.83], ...
          'FontName',FONT_NAME,'FontSize',FONT_SIZE_AX,'Box','off');
hold(ax,'on');

% Bars
for k = 1:n_models
    barh(ax, y_pos(k), rho_sorted(k), BAR_HEIGHT, ...
         'FaceColor', bar_cols(k,:), 'EdgeColor','none', 'FaceAlpha',0.88);
end

% ρ=0 reference line
if SHOW_ZERO_LINE
    xline(ax, 0, '-', 'Color', COLOR_ZERO, 'LineWidth', 1.4, ...
          'Alpha', 0.7, 'DisplayName','\rho = 0 (chance)');
end

% ρ=1 reference
xline(ax, 1, '--', 'Color', COLOR_PERFLINE, 'LineWidth', 0.8, ...
      'Alpha', 0.5, 'DisplayName','\rho = 1 (perfect)');

% Significance stars / "ns" at end of bars
for k = 1:n_models
    rho = rho_sorted(k);
    sig = sig_sorted{k};
    x_label = rho + sign(rho) * 0.03;
    if rho == 0, x_label = 0.03; end
    ha = 'left';
    if rho < 0, ha = 'right'; x_label = rho - 0.03; end

    text(ax, x_label, y_pos(k), sig, ...
         'HorizontalAlignment', ha, 'VerticalAlignment','middle', ...
         'FontSize', FONT_SIZE_STAR, 'FontName', FONT_NAME, ...
         'Color', bar_cols(k,:) * 0.75);

    % ρ value text inside or outside bar
    if abs(rho) > 0.12
        text(ax, rho/2, y_pos(k), sprintf('%.3f', rho_sorted(k)), ...
             'HorizontalAlignment','center','VerticalAlignment','middle', ...
             'FontSize', FONT_SIZE_VAL, 'FontName', FONT_NAME, ...
             'Color','white','FontWeight','bold');
    else
        x_val = rho + sign(rho) * 0.22;
        if rho == 0, x_val = 0.22; end
        text(ax, x_val, y_pos(k), sprintf('%.3f', rho_sorted(k)), ...
             'HorizontalAlignment','center','VerticalAlignment','middle', ...
             'FontSize', FONT_SIZE_VAL, 'FontName', FONT_NAME, ...
             'Color', [0.35 0.35 0.35]);
    end
end

% ── Axes ──────────────────────────────────────────────────────────────────
xlim(ax, [X_MIN - 0.05, X_MAX + 0.15]);
ylim(ax, [0.5, n_models + 0.5]);
ax.YTick     = (1:n_models)';
ax.YTickLabel = flipud(disp_labels);   % disp_labels already sorted desc, flip → bottom=lowest ρ
ax.TickLabelInterpreter = 'none';
ax.XGrid = 'on';
ax.GridLineStyle = '--';
ax.GridAlpha = 0.40;
ax.LineWidth = 0.8;

xlabel(ax, 'Spearman ρ  (AI country ranking vs. Human country ranking)', ...
       'FontSize',FONT_SIZE_LB,'FontName',FONT_NAME);
title(ax, 'C   Country rank order alignment: AI vs. Human', ...
      'FontSize',FONT_SIZE_TT,'FontName',FONT_NAME,'FontWeight','bold', ...
      'HorizontalAlignment','left','Units','normalized','Position',[0,1.04,0]);

% Explanatory text box
txt = sprintf(['Each bar: Spearman \\rho between model''s 8-country\n' ...
               'intervention rate ranking and human ranking.\n' ...
               '\\rho = 1: perfect cultural ordering; \\rho \\approx 0: random.\n' ...
               'Colored = p < .05; gray = not significant.']);
text(ax, X_MIN + 0.02, 1.6, txt, ...
     'FontSize', FONT_SIZE_AX - 0.5, 'FontName', FONT_NAME, ...
     'VerticalAlignment','bottom', 'Color',[0.40 0.40 0.40], ...
     'BackgroundColor','white','EdgeColor',[0.80 0.80 0.80],'Margin',4, ...
     'Interpreter','tex');

% Legend (only reference lines)
legend(ax, 'Location','southeast','FontSize',FONT_SIZE_AX, ...
       'FontName',FONT_NAME,'Box','off');

% ── Save ──────────────────────────────────────────────────────────────────
OUT_DIR = fullfile(fileparts(mfilename('fullpath')), 'output');
if ~exist(OUT_DIR,'dir'), mkdir(OUT_DIR); end
print(fig, fullfile(OUT_DIR,'panel_C_rank_corr.pdf'), '-dpdf', '-r300', '-bestfit');
print(fig, fullfile(OUT_DIR,'panel_C_rank_corr.png'), '-dpng', '-r300');
fprintf('Saved panel_C_rank_corr.pdf/.png\n');
