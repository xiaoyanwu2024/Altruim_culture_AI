% =============================================================================
% Author : Xiaoyan Wu
% Date   : June 2026
% Description: Supplementary Figure S5 — full M8 parameter distributions (Human vs GPT-5.4 vs Gemini-3.5-Flash)
% =============================================================================

% FigureS5_M8_params_dist.m
% =========================================================================
% Supplementary Fig S5: Full M8 parameter distributions
% Human KDE (gray) vs. GPT-5.4 vs. Gemini-3.5-Flash
% Layout: 2 rows × 4 columns (a–h panel letters, top-left of each panel)
% =========================================================================

clear; clc;

MFIT = '/Users/wuxiaoyan/Desktop/TPP_culture_AI/ModelFitting';
hr    = load(fullfile(MFIT,'results_human_aligned.mat')); hr = hr.results_aligned;
ar    = load(fullfile(MFIT,'results_ai.mat'));             ar = ar.results_ai;
tmp   = load(fullfile(MFIT,'data_ai_all.mat'));            adata = tmp.data_ai;

% ── Extract parameters ────────────────────────────────────────────────────
h_params = hr(8).x(:, 1:8);   % (875, 8)

x_all   = ar(8).x;            % (192, 10)
models  = {adata.model};
methods = {adata.Method};
idx_54  = find(strcmp(models,'GPT-5.4')          & strcmp(methods,'prob'));
idx_g35 = find(strcmp(models,'Gemini-3.5-Flash') & strcmp(methods,'prob'));

g54_params = x_all(idx_54,  1:8);   % (16, 8)
g35_params = x_all(idx_g35, 1:8);   % (16, 8)

% ── Pre-compute all 16 p-values and apply FDR correction ─────────────────
p_raw = zeros(16, 1);
for k = 1:8
    p_raw(2*k-1) = ranksum(g54_params(:,k), h_params(:,k));
    p_raw(2*k)   = ranksum(g35_params(:,k), h_params(:,k));
end

n_tests = 16;
[~, sort_idx] = sort(p_raw);
p_fdr = zeros(n_tests, 1);
for i = 1:n_tests
    p_fdr(sort_idx(i)) = min(p_raw(sort_idx(i)) * n_tests / i, 1);
end
for i = n_tests-1:-1:1
    p_fdr(sort_idx(i)) = min(p_fdr(sort_idx(i)), p_fdr(sort_idx(i+1)));
end

% ── Labels & settings ─────────────────────────────────────────────────────
PARAM_NAMES = {'\gamma', '\alpha', '\beta', '\lambda', '\omega', '\kappa', '\eta_{no}', '\eta_{yes}'};

FONT_NAME     = 'Arial';
FONT_SIZE_AX  = 7.5;
FONT_SIZE_LB  = 9;
FONT_SIZE_PANEL = 9;
C_HUM   = [0.30 0.30 0.30];
C_GPT54 = [0.45 0.35 0.75];
C_GEM35 = [0.10 0.40 0.82];

Y_LEVELS = [0.70, 0.50];   % vertical positions for GPT-5.4, Gemini strips

figW = 26;
figH = 12;

fig = figure('Units','centimeters','Position',[1 1 figW figH], ...
             'Color','white','PaperPositionMode','auto');

panel_letter = @(i) char('a' + i - 1);

for p = 1:8
    row = ceil(p/4);
    col = mod(p-1, 4) + 1;

    left = 0.06 + (col-1) * 0.235;
    bot  = 0.57 - (row-1) * 0.50;

    ax = axes(fig,'Units','normalized','Position',[left, bot, 0.19, 0.38], ...
              'FontName',FONT_NAME,'FontSize',FONT_SIZE_AX,'Box','off'); %#ok<LAXES>
    hold(ax,'on');

    hv = h_params(:, p);
    av = g54_params(:, p);
    gv = g35_params(:, p);

    % Dynamic x range
    all_vals = [hv; av; gv];
    xlo = prctile(all_vals, 0.5);
    xhi = prctile(all_vals, 99.5);
    pad = (xhi - xlo) * 0.15;
    xlo = xlo - pad;   xhi = xhi + pad;
    x_grid = linspace(xlo, xhi, 400);

    % ── Human KDE ─────────────────────────────────────────────────────────
    bw = 1.06 * std(hv) * length(hv)^(-1/5);
    bw = max(bw, (xhi-xlo)/50);
    kde_h = zeros(size(x_grid));
    for xi = 1:length(x_grid)
        kde_h(xi) = mean(normpdf(x_grid(xi), hv, bw));
    end
    kde_h = kde_h / max(kde_h);

    fill(ax, [x_grid, fliplr(x_grid)], [kde_h, zeros(size(kde_h))], ...
         C_HUM, 'FaceAlpha',0.13, 'EdgeColor','none');
    plot(ax, x_grid, kde_h, '-', 'Color',C_HUM, 'LineWidth',1.6);
    xline(ax, mean(hv), '--', 'Color',C_HUM, 'LineWidth',1.0, 'Alpha',0.6);

    % ── AI: mean ± SD + significance ──────────────────────────────────────
    AI_DATA   = {av, gv};
    AI_COLORS = {C_GPT54, C_GEM35};

    for gi = 1:2
        kv   = AI_DATA{gi};
        col_ = AI_COLORS{gi};
        yl   = Y_LEVELS(gi);

        fdr_idx = 2*p - 2 + gi;
        pv = p_fdr(fdr_idx);
        if     pv < 0.001, sig_str = '***';
        elseif pv < 0.01,  sig_str = '**';
        elseif pv < 0.05,  sig_str = '*';
        else,               sig_str = 'ns';
        end

        m  = mean(kv);
        sd = std(kv);

        plot(ax, [m-sd, m+sd], [yl yl],           '-', 'Color',col_, 'LineWidth',1.2);
        plot(ax, [m-sd, m-sd], [yl-0.02, yl+0.02],'-', 'Color',col_, 'LineWidth',0.9);
        plot(ax, [m+sd, m+sd], [yl-0.02, yl+0.02],'-', 'Color',col_, 'LineWidth',0.9);
        plot(ax, m, yl, 'o', 'Color',col_, 'MarkerFaceColor',col_, ...
             'MarkerSize',4.5, 'MarkerEdgeColor','white','LineWidth',0.7);
        text(ax, m, yl+0.01, sig_str, ...
             'HorizontalAlignment','center','VerticalAlignment','bottom', ...
             'FontSize',13,'FontName',FONT_NAME,'Color',col_,'FontWeight','bold');
    end

    % ── Axes ──────────────────────────────────────────────────────────────
    xlim(ax, [xlo, xhi]);
    ylim(ax, [0, 1.15]);
    ax.YTick = [];
    ax.XGrid = 'on';
    ax.GridLineStyle = ':';
    ax.GridAlpha = 0.25;
    ax.LineWidth = 0.7;

    if p >= 7, fw = 'normal'; else, fw = 'bold'; end
    xlabel(ax, PARAM_NAMES{p}, 'FontSize',FONT_SIZE_LB,'FontName',FONT_NAME,'Interpreter','tex');
    title(ax,  PARAM_NAMES{p}, 'FontSize',FONT_SIZE_LB,'FontName',FONT_NAME, ...
          'FontWeight',fw,'Interpreter','tex','Color','k');
    if col == 1
        ylabel(ax, 'Density', 'FontSize',FONT_SIZE_AX,'FontName',FONT_NAME);
    end

    % ── Panel letter outside axes (figure coordinates, top-left of panel) ──
    annotation(fig, 'textbox', [left - 0.032, bot + 0.38, 0.04, 0.04], ...
               'String', panel_letter(p), 'Color','k', 'EdgeColor','none', ...
               'FontSize',13, 'FontName',FONT_NAME, ...
               'FontWeight','normal', 'HorizontalAlignment','left', ...
               'VerticalAlignment','top', 'FitBoxToText',false);
end

% ── Shared legend ─────────────────────────────────────────────────────────
ax_leg = axes(fig,'Units','normalized','Position',[0.88 0.75 0.01 0.15],'Visible','off');
hold(ax_leg,'on');
h_hum = fill(ax_leg,[0 0 0 0],[0 0 0 0],C_HUM,'FaceAlpha',0.13,'EdgeColor',C_HUM,'LineWidth',1.2);
h_54  = plot(ax_leg, NaN, NaN, '-o','Color',C_GPT54,'LineWidth',1.2,'MarkerSize',4,'MarkerFaceColor',C_GPT54,'MarkerEdgeColor','white');
h_g35 = plot(ax_leg, NaN, NaN, '-o','Color',C_GEM35,'LineWidth',1.2,'MarkerSize',4,'MarkerFaceColor',C_GEM35,'MarkerEdgeColor','white');
legend(ax_leg, [h_hum, h_54, h_g35], ...
       {'Human', 'GPT-5.4', 'Gemini-3.5-Flash'}, ...
       'Orientation','vertical','Location','east', ...
       'FontSize',8,'FontName',FONT_NAME,'Box','off');

% ── Save ──────────────────────────────────────────────────────────────────
OUT_DIR   = fileparts(mfilename('fullpath'));
PAPER_DIR = '/Users/wuxiaoyan/Desktop/TPP_culture_AI/manuscript/AI_Culture_altruism';

set(fig, 'PaperUnits','centimeters', 'PaperSize',[figW figH], ...
         'PaperPosition',[0 0 figW figH]);
print(fig, fullfile(OUT_DIR,'FigureS5_M8_params_dist.pdf'), '-dpdf', '-r300');
print(fig, fullfile(OUT_DIR,'FigureS5_M8_params_dist.png'), '-dpng', '-r300');
print(fig, fullfile(PAPER_DIR,'FigureS5.pdf'), '-dpdf', '-r300');
fprintf('Saved FigureS5.pdf to manuscript folder\n');
