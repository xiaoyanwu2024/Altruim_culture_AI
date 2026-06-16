% =============================================================================
% Author : Xiaoyan Wu
% Date   : June 2026
% Description: Figure 3 panel B — F-statistics for country-level variation
% =============================================================================

% panel_B_Fstat.m
% =========================================================================
% Panel B — F-statistic (one-way ANOVA, country as factor)
%           Human vs. all AI models
%
% Design: Vertical bar chart
%   - Human bar: red, tall (F≈335), clearly separated
%   - AI bars:   colored by model family, all near floor
%   - Red dashed line: Human F value
%   - Orange dashed line: F-critical (α=.05)
%   - Significance stars above each bar
%   - Y-axis break to handle the large human F vs. tiny AI F
%
% HOW TO MODIFY:
%   FONT_NAME        : font family
%   FONT_SIZE_*      : font sizes
%   COLOR_HUMAN      : RGB for human bar
%   COLOR_FCRIT      : RGB for F-critical reference line
%   BAR_WIDTH        : bar width (0–1)
%   BREAK_THRESHOLD  : where to break the y-axis (set to 0 to disable break)
%   Y_MAX_TOP        : upper panel top limit
%   Y_MAX_BOT        : lower panel top limit (below the break)
%   figW / figH      : figure size in cm
%
% Note on axis break: MATLAB does not natively support broken axes.
%   This script simulates a break by drawing two subplot panels with a
%   diagonal break marker. Alternatively, set BREAK_THRESHOLD = 0 to
%   use a single panel with log-scale (set USE_LOG = true).
% =========================================================================

clear; clc;

DATA_DIR = fullfile(fileparts(mfilename('fullpath')), 'data');
load(fullfile(DATA_DIR, 'fig3_data.mat'));

% ── USER PARAMETERS ───────────────────────────────────────────────────────
FONT_NAME       = 'Helvetica';
FONT_SIZE_AX    = 8.5;
FONT_SIZE_LB    = 10;
FONT_SIZE_TT    = 12;
FONT_SIZE_STAR  = 8;
FONT_SIZE_VAL   = 7.5;

COLOR_HUMAN     = [0.75 0.12 0.12];   % red for human
COLOR_FCRIT     = [0.90 0.55 0.10];   % orange for F-critical line
BAR_WIDTH       = 0.65;
USE_LOG         = true;    % true = log scale (recommended); false = linear

figW = 20;
figH = 9;

% ── Model color map (consistent across all Figure3 panels) ────────────────
% Key = exact label string, Value = RGB
MODEL_COLOR_KEYS = {
    'GPT-4o_logprobs',           [0.65  0.20  0.10];   % dark coral
    'GPT-4o_prob',               [0.92  0.35  0.22];   % coral-red
    'GPT-4o-mini_logprobs',      [0.80  0.45  0.05];   % dark orange
    'GPT-4o-mini_prob',          [0.96  0.63  0.13];   % orange
    'GPT-5.4_prob',              [0.60  0.55  0.80];   % lavender
    'GPT-5.4-mini_prob',         [0.96  0.63  0.72];   % pink
    'Gemini-2.5-Flash_prob',     [0.20  0.62  0.55];   % dark teal
    'Gemini-3.5-Flash_prob',     [0.94  0.78  0.19];   % yellow
    'DeepSeek-V4-Flash_logprobs',[0.45  0.25  0.60];   % dark purple
    'DeepSeek-V4-Pro_logprobs',  [0.72  0.52  0.88];   % light purple
    'DeepSeek-V4-Pro_prob',      [0.60  0.40  0.72];   % medium purple
    'Mistral-Small-4_prob',      [0.72  0.79  0.41];   % yellow-green
};

% ── Desired order: Human → OpenAI → Gemini → DeepSeek → Mistral ──────────
DESIRED_ORDER = {
    'Human',
    'GPT-4o_logprobs',
    'GPT-4o_prob',
    'GPT-4o-mini_logprobs',
    'GPT-4o-mini_prob',
    'GPT-5.4_prob',
    'GPT-5.4-mini_prob',
    'Gemini-2.5-Flash_prob',
    'Gemini-3.5-Flash_prob',
    'DeepSeek-V4-Flash_logprobs',
    'DeepSeek-V4-Pro_logprobs',
    'DeepSeek-V4-Pro_prob',
    'Mistral-Small-4_prob',
};

% ── Reindex data to desired order ─────────────────────────────────────────
n_total    = length(DESIRED_ORDER);
x_pos      = 1:n_total;
F_ordered  = zeros(1, n_total);
sig_ordered= cell(1, n_total);
bar_cols   = zeros(n_total, 3);
disp_labels= cell(1, n_total);

for k = 1:n_total
    key = DESIRED_ORDER{k};
    % Find in all_labels_B
    idx = find(strcmp(all_labels_B, key));
    if isempty(idx)
        warning('Label not found: %s', key);
        F_ordered(k)   = NaN;
        sig_ordered{k} = '';
    else
        F_ordered(k)   = all_Fstat_B(idx);
        sig_ordered{k} = all_sig_B{idx};
    end
    % Color
    if strcmp(key, 'Human')
        bar_cols(k,:) = COLOR_HUMAN;
    else
        ci = find(strcmp(MODEL_COLOR_KEYS(:,1), key));
        if ~isempty(ci)
            bar_cols(k,:) = MODEL_COLOR_KEYS{ci,2};
        else
            bar_cols(k,:) = [0.5 0.5 0.5];
        end
    end
    % Display label
    lbl = strrep(key, '_logprobs', sprintf('\n[logprobs]'));
    lbl = strrep(lbl, '_prob',     '');
    disp_labels{k} = lbl;
end

% Company divider x-positions (after last model of each group)
% OpenAI: pos 2–7, Gemini: 8–9, DeepSeek: 10–12, Mistral: 13
divider_x = [7.5, 9.5, 12.5];   % vertical lines between groups

% Company label positions and names
company_x    = [4.5,  8.5, 11.0, 13.0];
company_names= {'OpenAI', 'Google Gemini', 'DeepSeek', 'Mistral'};

F_crit_val = F_critical(1);
h_F_val    = h_Fstat(1);

% ── Figure ────────────────────────────────────────────────────────────────
fig = figure('Units','centimeters','Position',[2 2 figW figH], ...
             'Color','white','PaperPositionMode','auto');

ax = axes(fig, 'Units','normalized','Position',[0.10 0.25 0.87 0.62], ...
          'FontName',FONT_NAME,'FontSize',FONT_SIZE_AX,'Box','off');
hold(ax,'on');

% Bars (using reordered data)
for k = 1:n_total
    if ~isnan(F_ordered(k))
        bar(ax, x_pos(k), F_ordered(k), BAR_WIDTH, ...
            'FaceColor', bar_cols(k,:), 'EdgeColor','none', 'FaceAlpha', 0.88);
    end
end

% Company divider lines
yl_tmp = [0.5, h_F_val * 4];
for d = 1:length(divider_x)
    plot(ax, [divider_x(d) divider_x(d)], yl_tmp, '-', ...
         'Color',[0.80 0.80 0.80], 'LineWidth', 0.8);
end

% F-critical dashed line
yline(ax, F_crit_val, '--', 'Color', COLOR_FCRIT, 'LineWidth', 1.8, ...
      'DisplayName', sprintf('F_{critical} (\\alpha=.05) = %.2f', F_crit_val));

% Human F dashed line
yline(ax, h_F_val, '-', 'Color', COLOR_HUMAN, 'LineWidth', 1.5, ...
      'Alpha', 0.6, 'DisplayName', sprintf('Human F = %.1f***', h_F_val));

% Significance stars and value labels
for k = 1:n_total
    fval  = F_ordered(k);
    sstr  = sig_ordered{k};
    if isnan(fval), continue; end

    if ~strcmp(sstr,'ns') && ~isempty(sstr)
        y_off = fval * 1.15;
        text(ax, x_pos(k), y_off, sstr, ...
             'HorizontalAlignment','center','VerticalAlignment','bottom', ...
             'FontSize', FONT_SIZE_STAR, 'FontName', FONT_NAME, ...
             'Color', bar_cols(k,:) * 0.75);
    else
        y_off = fval * 1.15;
        text(ax, x_pos(k), y_off, 'ns', ...
             'HorizontalAlignment','center','VerticalAlignment','bottom', ...
             'FontSize', FONT_SIZE_STAR-1, 'FontName', FONT_NAME, ...
             'Color', [0.65 0.65 0.65]);
    end

    % F value label
    if k > 1
        text(ax, x_pos(k), fval * 1.02, sprintf('%.1f', fval), ...
             'HorizontalAlignment','center','VerticalAlignment','bottom', ...
             'FontSize', FONT_SIZE_VAL-0.5, 'FontName', FONT_NAME, ...
             'Color', [0.35 0.35 0.35]);
    else
        text(ax, x_pos(k), 2.5, sprintf('F = %.1f', h_F_val), ...
             'HorizontalAlignment','center','VerticalAlignment','bottom', ...
             'FontSize', FONT_SIZE_VAL, 'FontName', FONT_NAME, ...
             'Color', 'white', 'FontWeight','bold');
    end
end

% ── Scale & formatting ────────────────────────────────────────────────────
if USE_LOG
    ax.YScale = 'log';
    ylim(ax, [0.5, h_F_val * 4]);
    ylabel(ax, 'F-statistic (log scale)', 'FontSize',FONT_SIZE_LB,'FontName',FONT_NAME);
else
    ylim(ax, [0, h_F_val * 1.15]);
    ylabel(ax, 'F-statistic', 'FontSize',FONT_SIZE_LB,'FontName',FONT_NAME);
end

ax.XTick = x_pos;
ax.XTickLabel = disp_labels;
ax.XTickLabelRotation = 35;
ax.TickLabelInterpreter = 'none';
ax.YGrid = 'on';
ax.GridLineStyle = '--';
ax.GridAlpha = 0.40;
ax.LineWidth = 0.8;
xlim(ax, [0.4, n_total + 0.6]);

xlabel(ax, '', 'FontSize',FONT_SIZE_LB,'FontName',FONT_NAME);
title(ax, 'B   F-statistic: Country as factor (one-way ANOVA)', ...
      'FontSize',FONT_SIZE_TT,'FontName',FONT_NAME,'FontWeight','bold', ...
      'HorizontalAlignment','left','Units','normalized','Position',[0,1.04,0]);

drawnow;
ax.XAxis.TickLabelColor = [0 0 0];

% ── Company group labels below x-axis ─────────────────────────────────────
company_colors = [
    0.80 0.20 0.10;   % OpenAI — red-ish
    0.10 0.55 0.85;   % Google Gemini — blue
    0.45 0.25 0.60;   % DeepSeek — purple
    0.72 0.79 0.41;   % Mistral — yellow-green
];
ax_pos = get(ax, 'Position');   % [left bottom width height] normalized
for ci = 1:length(company_names)
    % Convert data x to normalized figure coords
    x_norm = ax_pos(1) + (company_x(ci)-0.4)/(n_total+0.2) * ax_pos(3);
    annotation(fig, 'textbox', [x_norm-0.06, ax_pos(2)-0.14, 0.12, 0.10], ...
               'String', company_names{ci}, ...
               'Color', company_colors(ci,:), ...
               'FontSize', 8, 'FontName', FONT_NAME, 'FontWeight','bold', ...
               'HorizontalAlignment','center','VerticalAlignment','middle', ...
               'EdgeColor','none','FitBoxToText',false);
end

% Legend
legend(ax, 'Location','northeast','FontSize',FONT_SIZE_AX, ...
       'FontName',FONT_NAME,'Box','off');

% ── Save ──────────────────────────────────────────────────────────────────
OUT_DIR = fullfile(fileparts(mfilename('fullpath')), 'output');
if ~exist(OUT_DIR,'dir'), mkdir(OUT_DIR); end
print(fig, fullfile(OUT_DIR,'panel_B_Fstat.pdf'), '-dpdf', '-r300', '-bestfit');
print(fig, fullfile(OUT_DIR,'panel_B_Fstat.png'), '-dpng', '-r300');
fprintf('Saved panel_B_Fstat.pdf/.png\n');
