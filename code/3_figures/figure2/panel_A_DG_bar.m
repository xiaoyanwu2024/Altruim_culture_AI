% =============================================================================
% Author : Xiaoyan Wu
% Date   : June 2026
% Description: Figure 2 panel A — bar chart of DG donation rates by country (Human vs AI)
% =============================================================================

% panel_A_DG_bar.m
% =========================================================================
% Panel A: Dictator Game — Human + Top-3 AI, 4 sections × 8 countries
%
% HOW TO MODIFY:
%   COUNTRY_COLORS  : 8×3 RGB in ALPHABETICAL order
%                     (Chile, Greece, Italy, Mexico, Poland, Portugal, South Africa, Spain)
%   BAR_W           : bar width
%   GAP_INNER       : gap between bars within section
%   GAP_SECTION     : gap between sections
%   Y_MIN / Y_MAX   : y-axis range
%   figW / figH     : figure size in cm
% =========================================================================

clear; clc;

DATA_DIR = fullfile(fileparts(mfilename('fullpath')), 'data');
load(fullfile(DATA_DIR, 'dg_data.mat'));

% ── USER PARAMETERS ───────────────────────────────────────────────────────
FONT_NAME    = 'Helvetica';
FONT_SIZE_AX = 9;
FONT_SIZE_LB = 10;
FONT_SIZE_TT = 11;
FONT_SIZE_LG = 8.5;
FONT_SIZE_SEC= 9.5;

BAR_W       = 0.70;
GAP_INNER   = 0.15;
GAP_SECTION = 1.8;

Y_MIN = 0;
Y_MAX = 30;

figW = 18;
figH = 7;

% Country colors — ALPHABETICAL order:
%   Chile Greece Italy Mexico Poland Portugal South Africa Spain
COUNTRY_COLORS = [
    0.831  0.106  0.224;   % Chile        #D51C39
    0.000  0.333  0.855;   % Greece       #0055DA
    0.000  0.776  0.553;   % Italy        #00C68D
    0.996  0.929  0.255;   % Mexico       #FEEC41
    1.000  0.000  0.322;   % Poland       #FF0052
    0.349  0.698  0.573;   % Portugal     #59B292
    1.000  0.796  0.302;   % South Africa #FFC94D
    1.000  0.380  0.376;   % Spain        #FF6060
];

% ── Data prep ────────────────────────────────────────────────────────────
n_ctry = length(countries);
top3   = top3_dg_idx;

name1 = ai_dg_names{top3(1)};
name2 = ai_dg_names{top3(2)};
name3 = ai_dg_names{top3(3)};

[h_sorted, sort_ord] = sort(h_dg_mean, 'descend');
h_se_sorted = h_dg_se(sort_ord);
ai1_sorted  = ai_dg_mean(top3(1), sort_ord);
ai2_sorted  = ai_dg_mean(top3(2), sort_ord);
ai3_sorted  = ai_dg_mean(top3(3), sort_ord);
ctry_sorted = countries(sort_ord);
col_sorted  = COUNTRY_COLORS(sort_ord, :);

% ── Print stats to console ────────────────────────────────────────────────
fprintf('\n=== Panel A: Dictator Game stats ===\n');
fprintf('Human country means (sorted high→low):\n');
for c = 1:n_ctry
    fprintf('  %-15s  mean=%.2f  SE=%.2f\n', ctry_sorted{c}, h_sorted(c), h_se_sorted(c));
end
fprintf('\nTop-3 AI models (MAE vs human):\n');
fprintf('  1. %-30s  MAE=%.3f\n', name1, ai_dg_mae(top3(1)));
fprintf('  2. %-30s  MAE=%.3f\n', name2, ai_dg_mae(top3(2)));
fprintf('  3. %-30s  MAE=%.3f\n', name3, ai_dg_mae(top3(3)));
fprintf('\nAI country means:\n');
for c = 1:n_ctry
    fprintf('  %-15s  %s=%.2f  %s=%.2f  %s=%.2f\n', ...
        ctry_sorted{c}, name1, ai1_sorted(c), name2, ai2_sorted(c), name3, ai3_sorted(c));
end

% ── Build x-positions ─────────────────────────────────────────────────────
step   = BAR_W + GAP_INNER;
x_sec1 = (1:n_ctry) * step;
x_sec2 = x_sec1(end) + GAP_SECTION + (1:n_ctry)*step;
x_sec3 = x_sec2(end) + GAP_SECTION + (1:n_ctry)*step;
x_sec4 = x_sec3(end) + GAP_SECTION + (1:n_ctry)*step;

mid1 = mean(x_sec1); mid2 = mean(x_sec2);
mid3 = mean(x_sec3); mid4 = mean(x_sec4);

% ── Figure ────────────────────────────────────────────────────────────────
fig = figure('Units','centimeters','Position',[2 2 figW figH], ...
             'Color','white','PaperPositionMode','auto');

ax = axes('Parent',fig,'FontName',FONT_NAME,'FontSize',FONT_SIZE_AX, ...
          'Box','off','LineWidth',0.8,'Units','normalized', ...
          'Position',[0.07 0.14 0.91 0.72]);
hold(ax,'on');

% Human section light background
patch(ax,[x_sec1(1)-step/2, x_sec1(end)+step/2, x_sec1(end)+step/2, x_sec1(1)-step/2], ...
         [Y_MIN Y_MIN Y_MAX Y_MAX],[0.94 0.94 0.94],'FaceAlpha',0.5,'EdgeColor','none');

% ── Bars (all sections use country colors) ────────────────────────────────
all_vals = {h_sorted, ai1_sorted, ai2_sorted, ai3_sorted};
all_xpos = {x_sec1,  x_sec2,     x_sec3,     x_sec4};
alpha    = [0.88, 0.88, 0.88, 0.88];

for k = 1:4
    for c = 1:n_ctry
        bar(ax, all_xpos{k}(c), all_vals{k}(c), BAR_W, ...
            'FaceColor', col_sorted(c,:), 'EdgeColor','none', 'FaceAlpha', alpha(k));
    end
end

% Section separator lines
for xs = [x_sec1(end)+GAP_SECTION/2, x_sec2(end)+GAP_SECTION/2, x_sec3(end)+GAP_SECTION/2]
    xline(ax, xs, '--', 'Color',[0.72 0.72 0.72],'LineWidth',0.8,'Alpha',0.7);
end

% ── X-axis: section labels as ticks ───────────────────────────────────────
ax.XTick      = [mid1, mid2, mid3, mid4];
ax.XTickLabel = {'Human', name1, name2, name3};
ax.XTickLabelRotation = 0;
ax.TickLabelInterpreter = 'none';

% ── Axes ──────────────────────────────────────────────────────────────────
xlim(ax, [x_sec1(1)-step, x_sec4(end)+step/2]);
ylim(ax, [Y_MIN Y_MAX]);
ax.YTick     = 0:5:Y_MAX;
ax.YGrid     = 'on';
ax.GridColor = [0.85 0.85 0.85];
ax.GridLineStyle = '--';
ax.GridAlpha = 0.55;
ax.Layer     = 'top';
ax.XColor    = 'k';
ax.YColor    = 'k';

ylabel(ax,'Allocation (tokens out of 50)','FontSize',FONT_SIZE_LB,'FontName',FONT_NAME);
title(ax,'Dictator Game: Human vs. Top-3 AI models (by country)', ...
      'FontSize',FONT_SIZE_TT,'FontName',FONT_NAME,'FontWeight','bold', ...
      'HorizontalAlignment','center');

% ── Country legend — placed bottom-left (below lowest bars) ───────────────
leg_h = gobjects(n_ctry,1);
for c = 1:n_ctry
    leg_h(c) = patch(ax,NaN,NaN,col_sorted(c,:),'EdgeColor','none','FaceAlpha',0.92);
end
lg = legend(ax, leg_h, ctry_sorted, 'FontSize',FONT_SIZE_LG,'FontName',FONT_NAME, ...
            'Box','off','NumColumns',2,'Location','northwest');

% ── Save ─────────────────────────────────────────────────────────────────
OUT_DIR = fullfile(fileparts(mfilename('fullpath')), 'output');
if ~exist(OUT_DIR,'dir'), mkdir(OUT_DIR); end
print(fig, fullfile(OUT_DIR,'panel_A_DG_bar.pdf'), '-dpdf', '-r300', '-bestfit');
print(fig, fullfile(OUT_DIR,'panel_A_DG_bar.png'), '-dpng', '-r300');
fprintf('\nSaved panel_A_DG_bar.pdf/.png\n');
