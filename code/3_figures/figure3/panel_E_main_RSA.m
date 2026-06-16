% =============================================================================
% Author : Xiaoyan Wu
% Date   : June 2026
% Description: Figure 3 panel E (main) — RSA cultural similarity heatmaps
% =============================================================================

% panel_E_main_RSA.m
% =========================================================================
% Main-figure RSA: Human + 4 selected models
%
% Models shown:
%   1. Human (reference)
%   2. Mistral-Small-4       (Fig2 top-3 #1)
%   3. GPT-4o-mini           (Fig2 top-3 #2)
%   4. GPT-4o [logprobs]     (Fig2 top-3 #3)
%   5. DeepSeek-V4-Pro [lp]  (best Mantel ρ in Fig3 RSA)
%
% Figure 1 — 5 RDM heatmaps in one row (Human + 4 AI), shared colorbar
% Figure 2 — 4 Mantel null distributions in one row
%
% Colormap: Spring (magenta → yellow), consistent with supplementary
% =========================================================================

clear; clc;

DATA_DIR = fullfile(fileparts(mfilename('fullpath')), 'data');
load(fullfile(DATA_DIR, 'fig3_data.mat'));

% ── Parameters ────────────────────────────────────────────────────────────
FONT_NAME    = 'Helvetica';
FONT_SIZE_AX = 7;
FONT_SIZE_TT = 9;
FONT_SIZE_ST = 7;

n_c      = 256;
CMAP_RDM = spring(n_c);

full_labs = {'Chile','Greece','Italy','Mexico','Poland','Portugal','South Africa','Spain'};
N_CTRY   = 8;

% ── Select the 4 AI models ────────────────────────────────────────────────
% delta_ai_labels is alphabetical: DeepSeek-V4-Flash_logprobs, DeepSeek-V4-Pro_logprobs,
%   DeepSeek-V4-Pro_prob, GPT-4o-mini_logprobs, GPT-4o-mini_prob, GPT-4o_logprobs,
%   GPT-4o_prob, GPT-5.4-mini_prob, GPT-5.4_prob, Gemini-2.5-Flash_prob,
%   Gemini-3.5-Flash_prob, Mistral-Small-4_prob
SELECT_MODELS = { ...
    'Mistral-Small-4_prob', ...
    'GPT-4o-mini_prob', ...
    'GPT-4o_logprobs', ...
    'DeepSeek-V4-Pro_logprobs' };

SELECT_COLORS = [
    0.72  0.79  0.41;   % yellow-green  Mistral-Small-4
    0.96  0.63  0.13;   % orange        GPT-4o-mini
    0.65  0.20  0.10;   % dark coral    GPT-4o [lp]
    0.45  0.25  0.60;   % dark purple   DeepSeek-V4-Pro [lp]
];

SELECT_LABELS = { ...
    'Mistral-Small-4', ...
    'GPT-4o-mini', ...
    'GPT-4o  [logprobs]', ...
    'DeepSeek-V4-Pro  [logprobs]' };

% Find indices in delta_ai_labels
n_sel = numel(SELECT_MODELS);
sel_idx = zeros(1, n_sel);
for k = 1:n_sel
    idx = find(strcmp(delta_ai_labels, SELECT_MODELS{k}));
    if isempty(idx)
        error('Model not found: %s', SELECT_MODELS{k});
    end
    sel_idx(k) = idx;
end
fprintf('Selected model indices: '); fprintf('%d ', sel_idx); fprintf('\n');

% ── Console summary ───────────────────────────────────────────────────────
fprintf('\n%s\n', repmat('=',1,65));
fprintf('RSA Mantel — 4 main-figure models\n');
fprintf('%s\n', repmat('-',1,65));
fprintf('%-35s  %8s  %8s\n','Model','Mantel ρ','perm_p');
for k = 1:n_sel
    m = sel_idx(k);
    p = mantel_p(m);
    if p == 0, ps = sprintf('<%.4f',1/size(mantel_null,2));
    else,      ps = sprintf('%.4f',p); end
    fprintf('  %-33s  %8.3f  %8s\n', SELECT_LABELS{k}, mantel_r(m), ps);
end
fprintf('%s\n\n', repmat('=',1,65));

% ── Global CLim from lower-triangle values (Human + 4 AI) ─────────────────
tril_mask = logical(tril(ones(N_CTRY), -1));
all_vals  = h_rdm(tril_mask);
for k = 1:n_sel
    tmp      = squeeze(ai_rdm(sel_idx(k),:,:));
    all_vals = [all_vals; tmp(tril_mask)]; %#ok<AGROW>
end
all_vals = all_vals(~isnan(all_vals));
CLIM     = [min(all_vals), max(all_vals)];
fprintf('Global CLim: [%.3f, %.3f]\n\n', CLIM(1), CLIM(2));

% ═══════════════════════════════════════════════════════════════════════════
% FIGURE 1 — RDM heatmaps (1 row: Human + 4 AI + colorbar)
% ═══════════════════════════════════════════════════════════════════════════
N_PANELS = 1 + n_sel;   % 5 panels
figW1 = N_PANELS * 7.2 + 2.5;   % +2.5 for colorbar
figH1 = 8.5;

fig1 = figure('Units','centimeters','Position',[1 1 figW1 figH1], ...
              'Color','white','PaperPositionMode','auto');

for p = 1:N_PANELS
    ax = subplot(1, N_PANELS, p);

    if p == 1
        rdm   = h_rdm;
        ttl   = 'Human (reference)';
        fw    = 'bold';
        r_str = '';
    else
        k     = p - 1;
        m     = sel_idx(k);
        rdm   = squeeze(ai_rdm(m,:,:));
        ttl   = SELECT_LABELS{k};
        fw    = 'normal';
        rho   = mantel_r(m);
        pp    = mantel_p(m);
        if pp == 0, ps = sprintf('<%.4f',1/size(mantel_null,2));
        else,       ps = sprintf('%.4f',pp); end
        r_str = sprintf('Mantel \\rho = %.3f,  p = %s', rho, ps);
    end

    % Lower triangle only
    rdm_disp = rdm;
    rdm_disp(~tril_mask) = 0;
    alpha_data = double(tril_mask);

    h_img = imagesc(ax, rdm_disp, CLIM);
    h_img.AlphaData = alpha_data;
    colormap(ax, CMAP_RDM);
    ax.Color      = 'white';
    ax.TickLength = [0 0];
    ax.FontSize   = FONT_SIZE_AX;
    ax.FontName   = FONT_NAME;
    ax.XColor     = 'k';
    ax.YColor     = 'k';

    ax.XTick      = 1:7;
    ax.XTickLabel = full_labs(1:7);
    ax.XTickLabelRotation = 40;
    ax.YTick      = 2:8;
    ax.YTickLabel = full_labs(2:8);
    ax.Box        = 'off';
    ax.XAxis.TickLabelInterpreter = 'none';
    ax.YAxis.TickLabelInterpreter = 'none';

    title(ax, ttl, 'FontSize',FONT_SIZE_TT,'FontName',FONT_NAME, ...
          'FontWeight',fw,'Color','k','Interpreter','none');

    if ~isempty(r_str)
        xlabel(ax, r_str, 'FontSize',6.5,'FontName',FONT_NAME,'Color','k');
    end
end

% ── Standalone colorbar ────────────────────────────────────────────────────
sp_ref = subplot(1, N_PANELS, N_PANELS);
pos    = get(sp_ref, 'Position');
cb_w   = 0.012;
cb_h   = 0.55;
cb_x   = pos(1) + pos(3) + 0.040;
cb_y   = pos(2) + (pos(4) - cb_h)/2;

cb_ax = axes(fig1, 'Position',[cb_x, cb_y, cb_w, cb_h]);
imagesc(cb_ax, (1:n_c)');
colormap(cb_ax, CMAP_RDM);
set(cb_ax, 'YDir','normal','XTick',[], ...
    'YTick',linspace(1,n_c,5), ...
    'YTickLabel',arrayfun(@(v) sprintf('%.2f',v), ...
        linspace(CLIM(1),CLIM(2),5),'UniformOutput',false), ...
    'FontSize',FONT_SIZE_AX,'FontName',FONT_NAME,'YColor','k','XColor','k');
ylabel(cb_ax,'Pearson r  (country-pair similarity)', ...
       'FontSize',7,'FontName',FONT_NAME,'Color','k');

sgtitle({'RSA: Country similarity matrices  |  Human vs. selected AI models', ...
         'Lower triangle  ·  Shared color scale  ·  Mantel permutation test (N=5000)'}, ...
        'FontSize',10,'FontName',FONT_NAME,'FontWeight','bold','Color','k');

OUT_DIR = fullfile(fileparts(mfilename('fullpath')), 'output');
if ~exist(OUT_DIR,'dir'), mkdir(OUT_DIR); end
print(fig1, fullfile(OUT_DIR,'panel_E_main_RSA_rdm.pdf'),'-dpdf','-r200','-bestfit');
print(fig1, fullfile(OUT_DIR,'panel_E_main_RSA_rdm.png'),'-dpng','-r200');
fprintf('Saved panel_E_main_RSA_rdm.pdf/.png\n');

% ═══════════════════════════════════════════════════════════════════════════
% FIGURE 2 — Mantel null distributions (1 row × 4)
% ═══════════════════════════════════════════════════════════════════════════
figW2 = n_sel * 6.0;
figH2 = 5.5;

fig2 = figure('Units','centimeters','Position',[2 2 figW2 figH2], ...
              'Color','white','PaperPositionMode','auto');

for k = 1:n_sel
    ax      = subplot(1, n_sel, k);
    m       = sel_idx(k);
    col     = SELECT_COLORS(k,:);
    null_r  = mantel_null(m,:);
    rho_obs = mantel_r(m);
    pp      = mantel_p(m);

    histogram(ax, null_r, 35, 'FaceColor',col,'FaceAlpha',0.50, ...
              'EdgeColor','none','Normalization','probability');
    hold(ax,'on');

    ci_lo = prctile(null_r, 2.5);
    ci_hi = prctile(null_r, 97.5);
    yl = ylim(ax);
    plot(ax,[ci_lo ci_lo], yl,'--','Color',[0.6 0.6 0.6],'LineWidth',0.9);
    plot(ax,[ci_hi ci_hi], yl,'--','Color',[0.6 0.6 0.6],'LineWidth',0.9);
    plot(ax,[0 0], yl,'-','Color',[0.8 0.8 0.8],'LineWidth',0.7);
    plot(ax,[rho_obs rho_obs], yl,'-','Color',col,'LineWidth',2.5);

    pct = mean(null_r <= rho_obs) * 100;
    if pp == 0, ps = sprintf('p<%.4f',1/size(mantel_null,2));
    else,       ps = sprintf('p=%.3f',pp); end
    if pp < 0.05, sig = '*'; else, sig = 'ns'; end

    text(ax, 0.97, 0.97, ...
         sprintf('\\rho = %.3f\n%s  %s\npctile: %.0f%%', rho_obs, ps, sig, pct), ...
         'Units','normalized','HorizontalAlignment','right', ...
         'VerticalAlignment','top','FontSize',FONT_SIZE_ST, ...
         'FontName',FONT_NAME,'Color','k', ...
         'BackgroundColor','white','Margin',1);

    title(ax, SELECT_LABELS{k}, 'FontSize',8.5,'FontName',FONT_NAME, ...
          'Color','k','FontWeight','bold','Interpreter','none');
    xlabel(ax,'Mantel \rho (permuted)','FontSize',7,'FontName',FONT_NAME,'Color','k');
    if k == 1
        ylabel(ax,'Proportion','FontSize',7,'FontName',FONT_NAME,'Color','k');
    end
    ax.Box = 'off'; ax.FontSize = FONT_SIZE_AX; ax.FontName = FONT_NAME;
    ax.XColor = 'k'; ax.YColor = 'k';
    hold(ax,'off');
end

sgtitle({'Mantel permutation test (N=5000):  null distribution vs. observed \rho', ...
         'Dashed = null 95% CI   ·   Vertical line = observed Mantel \rho'}, ...
        'FontSize',10,'FontName',FONT_NAME,'FontWeight','bold');

print(fig2, fullfile(OUT_DIR,'panel_E_main_RSA_mantel.pdf'),'-dpdf','-r200','-bestfit');
print(fig2, fullfile(OUT_DIR,'panel_E_main_RSA_mantel.png'),'-dpng','-r200');
fprintf('Saved panel_E_main_RSA_mantel.pdf/.png\n');
