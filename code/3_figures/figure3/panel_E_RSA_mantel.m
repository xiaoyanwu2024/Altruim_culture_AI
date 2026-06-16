% =============================================================================
% Author : Xiaoyan Wu
% Date   : June 2026
% Description: Figure 3 panel E — Mantel test results (AI vs human RSA matrices)
% =============================================================================

% panel_E_RSA_mantel.m
% =========================================================================
% RSA: 8×8 country similarity matrices + Mantel permutation test
%
% Layout (two figures):
%   Figure 1 — RDM heatmaps: Human (reference) + 8 AI models, 3 per row
%              Colormap: blue (dissimilar) → white → red (similar)
%              Country tick labels colored by COUNTRY_COLORS
%
%   Figure 2 — Mantel null distributions: 2×4 panels
%              Histogram of 5000 permuted Mantel ρ values
%              Observed ρ marked; stats annotated
%
% All Mantel p > .05 → confirms cultural blindness at structural level
% =========================================================================

clear; clc;

DATA_DIR = fullfile(fileparts(mfilename('fullpath')), 'data');
load(fullfile(DATA_DIR, 'fig3_data.mat'));

% ── Parameters ────────────────────────────────────────────────────────────
FONT_NAME    = 'Helvetica';
FONT_SIZE_AX = 7;
FONT_SIZE_TT = 9;
FONT_SIZE_ST = 7;
N_COLS       = 3;   % RDM panels per row

% Country colors — ALPHABETICAL (Chile Greece Italy Mexico Poland Portugal S.Africa Spain)
COUNTRY_COLORS = [
    0.831  0.106  0.224;   % Chile
    0.000  0.333  0.855;   % Greece
    0.000  0.776  0.553;   % Italy
    0.996  0.929  0.255;   % Mexico
    1.000  0.000  0.322;   % Poland
    0.349  0.698  0.573;   % Portugal
    1.000  0.796  0.302;   % South Africa
    0.000  0.380  0.376;   % Spain (darkened for visibility on white)
];
COUNTRY_COLORS(8,:) = [1.000  0.380  0.376];   % restore Spain

% Model colors — alphabetical order (prob + logprobs):
%  1 DeepSeek-V4-Flash_logprobs   2 DeepSeek-V4-Pro_logprobs
%  3 DeepSeek-V4-Pro_prob         4 GPT-4o-mini_logprobs
%  5 GPT-4o-mini_prob             6 GPT-4o_logprobs
%  7 GPT-4o_prob                  8 GPT-5.4-mini_prob
%  9 GPT-5.4_prob                10 Gemini-2.5-Flash_prob
% 11 Gemini-3.5-Flash_prob       12 Mistral-Small-4_prob
MODEL_COLORS = [
    0.45  0.25  0.60;   % dark purple (logprobs)  DeepSeek-V4-Flash [lp]
    0.72  0.52  0.88;   % light purple (logprobs) DeepSeek-V4-Pro [lp]
    0.60  0.40  0.72;   % medium purple           DeepSeek-V4-Pro
    0.80  0.45  0.05;   % dark orange (logprobs)  GPT-4o-mini [lp]
    0.96  0.63  0.13;   % orange                  GPT-4o-mini
    0.65  0.20  0.10;   % dark coral (logprobs)   GPT-4o [lp]
    0.92  0.35  0.22;   % coral-red               GPT-4o
    0.96  0.63  0.72;   % pink                    GPT-5.4-mini
    0.60  0.55  0.80;   % lavender                GPT-5.4
    0.20  0.62  0.55;   % dark teal               Gemini-2.5-Flash
    0.94  0.78  0.19;   % yellow                  Gemini-3.5-Flash
    0.72  0.79  0.41;   % yellow-green            Mistral-Small-4
];

n_ai      = double(n_ai_delta);
% Full country names (alphabetical)
full_labs = {'Chile','Greece','Italy','Mexico','Poland','Portugal','South Africa','Spain'};

n_c = 256;

% ── Two MATLAB built-in colormaps to compare ──────────────────────────────
CMAP_WINTER = winter(n_c);   % blue → green  (cool)
CMAP_SPRING = spring(n_c);   % magenta → yellow (warm)

% ── Active colormap: Spring (magenta → yellow) ────────────────────────────
CMAP_RDM = CMAP_SPRING;

clean_name = @(s) strrep(strrep(s,'_prob',''),'_logprobs',' [lp]');

clean_name = @(s) strrep(strrep(s,'_prob',''),'_logprobs',' [lp]');

% ── Console output ────────────────────────────────────────────────────────
fprintf('\n%s\n', repmat('=',1,70));
fprintf('RSA Mantel test — Human vs AI country similarity (Spearman ρ)\n');
fprintf('%s\n', repmat('=',1,70));
fprintf('%-32s  %8s  %8s  %10s\n','Model','Mantel ρ','perm_p','Verdict');
fprintf('%s\n', repmat('-',1,65));
for m = 1:n_ai
    r   = mantel_r(m);
    p   = mantel_p(m);
    if p < 0.001, sig = '***';
    elseif p < 0.01, sig = '**';
    elseif p < 0.05, sig = '*';
    else, sig = 'ns'; end
    if p == 0, ps = sprintf('<%.4f', 1/size(mantel_null,2));
    else,      ps = sprintf('%.4f', p); end
    fprintf('  %-30s  %8.3f  %8s  %10s\n', ...
        clean_name(delta_ai_labels{m}), r, ps, sig);
end
fprintf('%s\n\n', repmat('=',1,70));

% ═════════════════════════════════════════════════════════════════════════
% FIGURE 1 — RDM heatmaps
% ═════════════════════════════════════════════════════════════════════════
n_panels = 1 + n_ai;          % Human + 8 AI
N_ROWS_R  = ceil(n_panels / N_COLS);
figW1 = N_COLS * 7.5;
figH1 = N_ROWS_R * 7.0;

% Compute global CLim from lower-triangle values only (consistent across all panels)
tril_mask = logical(tril(ones(8),-1));   % lower triangle, no diagonal
all_vals  = h_rdm(tril_mask);
for m = 1:n_ai
    tmp      = squeeze(ai_rdm(m,:,:));
    all_vals = [all_vals; tmp(tril_mask)]; %#ok<AGROW>
end
all_vals = all_vals(~isnan(all_vals));
CLIM     = [min(all_vals), max(all_vals)];
fprintf('Global CLim: [%.3f, %.3f]\n', CLIM(1), CLIM(2));

% Full country names (alphabetical order from data)
full_labs = {'Chile','Greece','Italy','Mexico','Poland','Portugal','South Africa','Spain'};

% Leave right margin for standalone colorbar
fig1 = figure('Units','centimeters','Position',[1 1 figW1+3 figH1], ...
              'Color','white','PaperPositionMode','auto');

for p = 1:n_panels
    ax = subplot(N_ROWS_R, N_COLS, p);

    if p == 1
        rdm   = h_rdm;
        ttl   = 'Human (reference)';
        fw    = 'bold';
        r_str = '';
    else
        m     = p - 1;
        rdm   = squeeze(ai_rdm(m,:,:));
        ttl   = clean_name(delta_ai_labels{m});
        fw    = 'normal';
        rho   = mantel_r(m);
        pp    = mantel_p(m);
        if pp == 0, ps = sprintf('<%.4f',1/size(mantel_null,2));
        else,       ps = sprintf('%.4f',pp); end
        r_str = sprintf('Mantel \\rho = %.3f,  p = %s', rho, ps);
    end

    % Lower triangle only: use AlphaData so upper triangle is fully transparent
    rdm_disp = rdm;
    rdm_disp(~tril_mask) = 0;          % dummy value; masked by alpha anyway
    alpha_data = double(tril_mask);    % 1 = opaque, 0 = fully transparent

    h_img = imagesc(ax, rdm_disp, CLIM);
    h_img.AlphaData = alpha_data;
    colormap(ax, CMAP_RDM);
    ax.Color      = 'white';           % axes background = white (shows through alpha=0 cells)
    ax.TickLength = [0 0];
    ax.FontSize   = FONT_SIZE_AX;
    ax.FontName   = FONT_NAME;
    ax.XColor     = 'k';
    ax.YColor     = 'k';

    % X-axis: columns 1–7
    ax.XTick      = 1:7;
    ax.XTickLabel = full_labs(1:7);
    ax.XTickLabelRotation = 40;

    % Y-axis: rows 2–8
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

% ── Standalone colorbar — placed to the right of all subplots ─────────────
% Get right edge of last subplot in first row
sp_ref = subplot(N_ROWS_R, N_COLS, N_COLS);
pos    = get(sp_ref, 'Position');   % [left bottom width height]

cb_w  = 0.018;
cb_h  = 0.55;
cb_x  = pos(1) + pos(3) + 0.045;   % right of last top panel
cb_y  = pos(2) + (pos(4) - cb_h)/2;

cb_ax = axes(fig1, 'Position',[cb_x, cb_y, cb_w, cb_h]);
imagesc(cb_ax, (1:n_c)');
colormap(cb_ax, CMAP_RDM);
set(cb_ax, 'YDir','normal','XTick',[], ...
    'YTick',linspace(1,n_c,5), ...
    'YTickLabel',arrayfun(@(v) sprintf('%.2f',v), ...
        linspace(CLIM(1),CLIM(2),5),'UniformOutput',false), ...
    'FontSize',FONT_SIZE_AX,'FontName',FONT_NAME,'YColor','k','XColor','k');
ylabel(cb_ax,'Pearson r  (country pair similarity)', ...
       'FontSize',7,'FontName',FONT_NAME,'Color','k');

sgtitle({'RSA: Country similarity matrices (Pearson r across 100 conditions)', ...
         'Human reference vs. all AI models  |  lower triangle  |  shared color scale'}, ...
        'FontSize',11,'FontName',FONT_NAME,'FontWeight','bold','Color','k');

OUT_DIR = fullfile(fileparts(mfilename('fullpath')), 'output');
if ~exist(OUT_DIR,'dir'), mkdir(OUT_DIR); end
print(fig1, fullfile(OUT_DIR,'panel_E_RSA_rdm.pdf'), '-dpdf','-r200','-bestfit');
print(fig1, fullfile(OUT_DIR,'panel_E_RSA_rdm.png'), '-dpng','-r200');
fprintf('Saved panel_E_RSA_rdm.pdf/.png\n');

% ═════════════════════════════════════════════════════════════════════════
% FIGURE 2 — Mantel null distributions
% ═════════════════════════════════════════════════════════════════════════
N_COLS2 = 4;
N_ROWS2 = ceil(n_ai / N_COLS2);
figW2 = N_COLS2 * 5.5;
figH2 = N_ROWS2 * 4.8;

fig2 = figure('Units','centimeters','Position',[2 2 figW2 figH2], ...
              'Color','white','PaperPositionMode','auto');

for m = 1:n_ai
    ax  = subplot(N_ROWS2, N_COLS2, m);
    col = MODEL_COLORS(m,:);
    null_r = mantel_null(m,:);
    rho_obs = mantel_r(m);
    pp      = mantel_p(m);

    % Histogram
    histogram(ax, null_r, 35, 'FaceColor',col,'FaceAlpha',0.50, ...
              'EdgeColor','none','Normalization','probability');
    hold(ax,'on');

    % 95% CI of null
    ci_lo = prctile(null_r, 2.5);
    ci_hi = prctile(null_r, 97.5);
    yl = ylim(ax);
    plot(ax,[ci_lo ci_lo], yl,'--','Color',[0.6 0.6 0.6],'LineWidth',0.9);
    plot(ax,[ci_hi ci_hi], yl,'--','Color',[0.6 0.6 0.6],'LineWidth',0.9);

    % Zero line
    plot(ax,[0 0], yl,'-','Color',[0.8 0.8 0.8],'LineWidth',0.7);

    % Observed ρ
    plot(ax,[rho_obs rho_obs], yl,'-','Color',col,'LineWidth',2.5);

    % Stats text
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

    % Style — all text black
    title(ax, clean_name(delta_ai_labels{m}), ...
          'FontSize',8.5,'FontName',FONT_NAME,'Color','k', ...
          'FontWeight','bold','Interpreter','none');
    xlabel(ax, 'Mantel \rho (permuted)', 'FontSize',7,'FontName',FONT_NAME,'Color','k');
    if mod(m-1,N_COLS2) == 0
        ylabel(ax,'Proportion','FontSize',7,'FontName',FONT_NAME,'Color','k');
    end
    ax.Box = 'off'; ax.FontSize = FONT_SIZE_AX; ax.FontName = FONT_NAME;
    ax.XColor = 'k'; ax.YColor = 'k';
    hold(ax,'off');
end

sgtitle({'Mantel permutation test (N=5000):  null distribution vs. observed \rho', ...
         'Dashed lines = null 95% CI   |   Vertical line = observed Mantel \rho'}, ...
        'FontSize',11,'FontName',FONT_NAME,'FontWeight','bold');

print(fig2, fullfile(OUT_DIR,'panel_E_RSA_mantel_null.pdf'), '-dpdf','-r200','-bestfit');
print(fig2, fullfile(OUT_DIR,'panel_E_RSA_mantel_null.png'), '-dpng','-r200');
fprintf('Saved panel_E_RSA_mantel_null.pdf/.png\n');
