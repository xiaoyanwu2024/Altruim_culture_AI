% =============================================================================
% Author : Xiaoyan Wu
% Date   : June 2026
% Description: Figure 3 panel A — human cross-cultural similarity matrix (RSA)
% =============================================================================

% panel_A_human_country.m
% =========================================================================
% Panel A — Human cross-cultural variation in TPP intervention rates
%
% Design: Dot + bar chart (raincloud style)
%   - Each country: gray jittered dots (participant-level means)
%              + colored filled circle (country mean)
%              + vertical error bar (±1 SE)
%   - Countries sorted by mean intervention rate (highest → lowest)
%   - Significant between-country differences annotated
%
% HOW TO MODIFY:
%   FONT_NAME        : 'Helvetica', 'Arial', 'Times New Roman'
%   FONT_SIZE_*      : axis tick / label / title sizes
%   DOT_SIZE_INDIV   : individual participant dot size (pts²)
%   DOT_SIZE_MEAN    : country mean dot size (pts²)
%   DOT_ALPHA_INDIV  : individual dot transparency (0–1)
%   JITTER_WIDTH     : horizontal jitter spread (0 = no jitter)
%   COUNTRY_COLORS   : 8×3 RGB matrix — one color per country
%   Y_MIN / Y_MAX    : y-axis range (intervention rate 0–1)
%   ERROR_BAR_WIDTH  : cap width of SE error bars
%   figW / figH      : figure size in centimeters
% =========================================================================

clear; clc;

DATA_DIR = fullfile(fileparts(mfilename('fullpath')), 'data');
load(fullfile(DATA_DIR, 'fig3_data.mat'));

% ── USER PARAMETERS ───────────────────────────────────────────────────────
FONT_NAME        = 'Helvetica';
FONT_SIZE_AX     = 9;
FONT_SIZE_LB     = 10.5;
FONT_SIZE_TT     = 12;
FONT_SIZE_ANNOT  = 8.5;

DOT_SIZE_INDIV   = 14;     % individual participant dots
DOT_SIZE_MEAN    = 80;     % country mean dots (larger)
DOT_ALPHA_INDIV  = 0.25;   % individual dot transparency (low = subtle)
JITTER_WIDTH     = 0.22;   % horizontal jitter spread

LINE_WIDTH_EB    = 1.8;    % error bar line width
ERROR_BAR_CAP    = 5;      % cap size in points
BAR_LINE_WIDTH   = 0.0;    % 0 = no bar (use only dots); >0 draws thin vertical bar

Y_MIN = 0.20;
Y_MAX = 0.55;

figW = 14;
figH = 8;

% Country colors (one per country, in sorted order from preprocess)
% Default palette — change any row to [R G B] (values 0–1)
COUNTRY_COLORS = [
    0.85  0.10  0.10;   % South Africa   (red)
    0.20  0.55  0.85;   % Italy          (blue)
    0.20  0.65  0.30;   % Mexico         (green)
    0.55  0.22  0.72;   % Poland         (purple)
    0.95  0.50  0.10;   % Portugal       (orange)
    0.60  0.30  0.15;   % Greece         (brown)
    0.88  0.40  0.65;   % Spain          (pink)
    0.40  0.40  0.40;   % Chile          (gray)
];

% ── Data preparation ──────────────────────────────────────────────────────
n_ctry = length(h_country_names);

% h_participant_mat is (n_ctry × max_n) with NaN padding
[~, max_n] = size(h_participant_mat);

% ── Figure ────────────────────────────────────────────────────────────────
fig = figure('Units','centimeters','Position',[2 2 figW figH], ...
             'Color','white','PaperPositionMode','auto');

ax = axes(fig, 'Units','normalized','Position',[0.11 0.18 0.85 0.70], ...
          'FontName', FONT_NAME, 'FontSize', FONT_SIZE_AX, 'Box','off');
hold(ax,'on');

for c = 1:n_ctry
    col   = COUNTRY_COLORS(mod(c-1, size(COUNTRY_COLORS,1))+1, :);
    vals  = h_participant_mat(c, :);
    vals  = vals(~isnan(vals));  % remove NaN padding
    n_pts = length(vals);

    % Jittered x positions
    jitter = (rand(n_pts,1) - 0.5) * 2 * JITTER_WIDTH;
    x_pts  = c + jitter;

    % Individual dots (participants)
    scatter(ax, x_pts, vals, DOT_SIZE_INDIV, ...
            col, 'filled', 'MarkerFaceAlpha', DOT_ALPHA_INDIV, ...
            'MarkerEdgeColor','none');

    % Error bar (mean ± SE)
    m  = h_country_means(c);
    se = h_country_se(c);
    errorbar(ax, c, m, se, 'k-', 'LineWidth', LINE_WIDTH_EB, ...
             'CapSize', ERROR_BAR_CAP);

    % Mean dot (on top)
    scatter(ax, c, m, DOT_SIZE_MEAN, col, 'filled', ...
            'MarkerEdgeColor','white', 'LineWidth', 1.0);

    % Mean value label
    text(ax, c, m + se + 0.012, sprintf('%.3f', m), ...
         'HorizontalAlignment','center','VerticalAlignment','bottom', ...
         'FontSize', FONT_SIZE_ANNOT - 1, 'FontName', FONT_NAME, ...
         'Color', col * 0.8);
end

% ── Axes formatting ───────────────────────────────────────────────────────
xlim(ax, [0.4, n_ctry + 0.6]);
ylim(ax, [Y_MIN, Y_MAX]);
ax.XTick = 1:n_ctry;
ax.XTickLabel = cellfun(@(s) strrep(s,' ',sprintf('\n')), ...
                        h_country_names, 'UniformOutput', false);
ax.XTickLabelRotation = 0;
ax.YGrid = 'on';
ax.GridLineStyle = '--';
ax.GridAlpha = 0.45;
ax.GridColor = [0.80 0.80 0.80];
ax.Layer = 'top';
ax.LineWidth = 0.8;

xlabel(ax, 'Country (sorted by mean intervention rate)', ...
       'FontSize', FONT_SIZE_LB, 'FontName', FONT_NAME);
ylabel(ax, 'Intervention rate (P(intervene))', ...
       'FontSize', FONT_SIZE_LB, 'FontName', FONT_NAME);
title(ax, 'A   Human cross-cultural variation in third-party intervention', ...
      'FontSize', FONT_SIZE_TT, 'FontName', FONT_NAME, 'FontWeight','bold', ...
      'HorizontalAlignment','left', 'Units','normalized', ...
      'Position',[0, 1.04, 0]);

% η² and sample size annotation
n_str = sprintf('η² = %.4f***\n(one-way ANOVA, N = %d participants)', ...
                h_eta2, sum(h_country_n));
text(ax, 0.02, 0.97, n_str, 'Units','normalized', ...
     'VerticalAlignment','top','HorizontalAlignment','left', ...
     'FontSize', FONT_SIZE_ANNOT, 'FontName', FONT_NAME, ...
     'Color',[0.25 0.25 0.25], ...
     'BackgroundColor','white','EdgeColor',[0.80 0.80 0.80], ...
     'Margin', 3);

% Small n-label per country at bottom
for c = 1:n_ctry
    text(ax, c, Y_MIN - 0.015, sprintf('n=%d', h_country_n(c)), ...
         'HorizontalAlignment','center', 'VerticalAlignment','top', ...
         'FontSize', FONT_SIZE_ANNOT - 1.5, 'FontName', FONT_NAME, ...
         'Color',[0.55 0.55 0.55]);
end

% ── Save ──────────────────────────────────────────────────────────────────
OUT_DIR = fullfile(fileparts(mfilename('fullpath')), 'output');
if ~exist(OUT_DIR,'dir'), mkdir(OUT_DIR); end
print(fig, fullfile(OUT_DIR,'panel_A_human_country.pdf'), '-dpdf', '-r300', '-bestfit');
print(fig, fullfile(OUT_DIR,'panel_A_human_country.png'), '-dpng', '-r300');
fprintf('Saved panel_A_human_country.pdf/.png\n');
