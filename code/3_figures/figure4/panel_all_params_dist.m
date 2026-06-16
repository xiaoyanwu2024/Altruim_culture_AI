% =============================================================================
% Author : Xiaoyan Wu
% Date   : June 2026
% Description: Figure 4 — distributions of all M8 parameters
% =============================================================================

% panel_all_params_dist.m
% =========================================================================
% All 8 M8 parameters: Human KDE vs. GPT-5.4 vs. Gemini-3.5-Flash
% Layout: 2 rows × 4 columns
% Each panel: Human KDE (gray) + AI mean±SD + significance vs Human
% =========================================================================

clear; clc;

MFIT = '/Users/wuxiaoyan/Desktop/TPP_culture_AI/ModelFitting';
hr    = load(fullfile(MFIT,'results_human_aligned.mat')); hr = hr.results_aligned;
ar    = load(fullfile(MFIT,'results_ai.mat'));             ar = ar.results_ai;
tmp   = load(fullfile(MFIT,'data_ai_all.mat'));            adata = tmp.data_ai;

% ── Extract parameters ────────────────────────────────────────────────────
h_params = hr(8).x(:, 1:8);   % (875, 8)

x_all = ar(8).x;              % (192, 10)
models  = {adata.model};
methods = {adata.Method};
idx_54  = find(strcmp(models,'GPT-5.4')          & strcmp(methods,'prob'));
idx_g35 = find(strcmp(models,'Gemini-3.5-Flash') & strcmp(methods,'prob'));

g54_params = x_all(idx_54,  1:8);   % (16, 8)
g35_params = x_all(idx_g35, 1:8);   % (16, 8)

% ── Pre-compute all 16 p-values and apply FDR correction ─────────────────
% Order: [GPT-5.4 param1, Gemini param1, GPT-5.4 param2, Gemini param2, ...]
p_raw = zeros(16, 1);
for k = 1:8
    p_raw(2*k-1) = ranksum(g54_params(:,k), h_params(:,k));
    p_raw(2*k)   = ranksum(g35_params(:,k), h_params(:,k));
end

% Benjamini-Hochberg FDR
n_tests = 16;
[p_sorted, sort_idx] = sort(p_raw);
p_fdr = zeros(n_tests, 1);
for i = 1:n_tests
    p_fdr(sort_idx(i)) = min(p_raw(sort_idx(i)) * n_tests / i, 1);
end
for i = n_tests-1:-1:1
    p_fdr(sort_idx(i)) = min(p_fdr(sort_idx(i)), p_fdr(sort_idx(i+1)));
end

% ── Labels & settings ─────────────────────────────────────────────────────
PARAM_NAMES = {'\gamma', '\alpha', '\beta', '\lambda', '\omega', '\kappa', '\eta_{no}', '\eta_{yes}'};
PARAM_FULL  = {'γ (gama)', 'α (envy)', 'β (guilt)', 'λ (lamda)', ...
               'ω (omega)', 'κ (kapa)', 'η_{no}', 'η_{yes}'};

FONT_NAME = 'Helvetica';
C_HUM   = [0.30 0.30 0.30];
C_GPT54 = [0.45 0.35 0.75];
C_GEM35 = [0.10 0.40 0.82];

Y_LEVELS = [0.70, 0.50];   % vertical positions for GPT-5.4, Gemini strips

figW = 26;
figH = 12;

fig = figure('Units','centimeters','Position',[1 1 figW figH], ...
             'Color','white','PaperPositionMode','auto');

for p = 1:8
    row = ceil(p/4);
    col = mod(p-1, 4) + 1;

    left = 0.06 + (col-1) * 0.235;
    bot  = 0.57 - (row-1) * 0.50;

    ax = axes(fig,'Units','normalized','Position',[left, bot, 0.19, 0.38], ...
              'FontName',FONT_NAME,'FontSize',7.5,'Box','off'); %#ok<LAXES>
    hold(ax,'on');

    hv = h_params(:, p);
    av = g54_params(:, p);
    gv = g35_params(:, p);

    % Dynamic x range: cover human data + AI means
    all_vals = [hv; av; gv];
    xlo = prctile(all_vals, 0.5);
    xhi = prctile(all_vals, 99.5);
    pad = (xhi - xlo) * 0.15;
    xlo = xlo - pad;   xhi = xhi + pad;
    x_grid = linspace(xlo, xhi, 400);

    % ── Human KDE ─────────────────────────────────────────────────────────
    bw = 1.06 * std(hv) * length(hv)^(-1/5);   % Silverman's rule
    bw = max(bw, (xhi-xlo)/50);
    kde_h = zeros(size(x_grid));
    for xi = 1:length(x_grid)
        kde_h(xi) = mean(normpdf(x_grid(xi), hv, bw));
    end
    kde_h = kde_h / max(kde_h);

    fill(ax, [x_grid, fliplr(x_grid)], [kde_h, zeros(size(kde_h))], ...
         C_HUM, 'FaceAlpha',0.13, 'EdgeColor','none');
    plot(ax, x_grid, kde_h, '-', 'Color',C_HUM, 'LineWidth',1.6);

    % Human mean line
    xline(ax, mean(hv), '--', 'Color',C_HUM, 'LineWidth',1.0, 'Alpha',0.6);

    % ── AI: mean ± SD + significance ──────────────────────────────────────
    AI_DATA   = {av, gv};
    AI_COLORS = {C_GPT54, C_GEM35};

    for gi = 1:2
        kv  = AI_DATA{gi};
        col_ = AI_COLORS{gi};
        yl  = Y_LEVELS(gi);

        fdr_idx = 2*p - 2 + gi;   % index into p_fdr vector
        pv = p_fdr(fdr_idx);
        if     pv < 0.001, sig_str = '***';
        elseif pv < 0.01,  sig_str = '**';
        elseif pv < 0.05,  sig_str = '*';
        else,               sig_str = 'ns';
        end

        m  = mean(kv);
        sd = std(kv);

        % SD bar + end caps
        plot(ax, [m-sd, m+sd], [yl yl], '-', 'Color',col_, 'LineWidth',1.2);
        plot(ax, [m-sd, m-sd], [yl-0.02, yl+0.02], '-', 'Color',col_, 'LineWidth',0.9);
        plot(ax, [m+sd, m+sd], [yl-0.02, yl+0.02], '-', 'Color',col_, 'LineWidth',0.9);
        % Mean dot
        plot(ax, m, yl, 'o', 'Color',col_, 'MarkerFaceColor',col_, ...
             'MarkerSize',4.5, 'MarkerEdgeColor','white','LineWidth',0.7);

        % Significance marker just above the mean dot
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

    xlabel(ax, PARAM_NAMES{p}, 'FontSize',9,'FontName',FONT_NAME,'Interpreter','tex');
    title(ax, PARAM_NAMES{p}, 'FontSize',9,'FontName',FONT_NAME, ...
          'FontWeight',fw,'Interpreter','tex','Color','k');

    if col == 1
        ylabel(ax, 'Density', 'FontSize',7.5,'FontName',FONT_NAME);
    end
end

% ── Shared legend (top right) ─────────────────────────────────────────────
ax_leg = axes(fig,'Units','normalized','Position',[0.88 0.75 0.01 0.15],'Visible','off');
hold(ax_leg,'on');
h_hum = fill(ax_leg,[0 0 0 0],[0 0 0 0],C_HUM,'FaceAlpha',0.13,'EdgeColor',C_HUM,'LineWidth',1.2);
h_54  = plot(ax_leg, NaN, NaN, '-o','Color',C_GPT54,'LineWidth',1.2,'MarkerSize',4,'MarkerFaceColor',C_GPT54,'MarkerEdgeColor','white');
h_g35 = plot(ax_leg, NaN, NaN, '-o','Color',C_GEM35,'LineWidth',1.2,'MarkerSize',4,'MarkerFaceColor',C_GEM35,'MarkerEdgeColor','white');
legend(ax_leg, [h_hum, h_54, h_g35], ...
       {'Human', 'GPT-5.4', 'Gemini-3.5-Flash'}, ...
       'Orientation','vertical','Location','east', ...
       'FontSize',8,'FontName',FONT_NAME,'Box','off');

% ── Overall title ─────────────────────────────────────────────────────────

% ── Save ──────────────────────────────────────────────────────────────────
OUT_DIR = fullfile(fileparts(mfilename('fullpath')), 'output');
if ~exist(OUT_DIR,'dir'), mkdir(OUT_DIR); end
print(fig, fullfile(OUT_DIR,'panel_all_params_dist.pdf'), '-dpdf', '-r300', '-bestfit');
print(fig, fullfile(OUT_DIR,'panel_all_params_dist.png'), '-dpng', '-r300');
fprintf('Saved panel_all_params_dist.pdf/.png\n');
