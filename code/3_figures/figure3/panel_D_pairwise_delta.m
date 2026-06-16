% =============================================================================
% Author : Xiaoyan Wu
% Date   : June 2026
% Description: Figure 3 panel D — pairwise cross-national delta comparison
% =============================================================================

% panel_D_pairwise_delta.m
% =========================================================================
% Panel D: Cross-cultural sensitivity — pairwise country differences
%
% For each of 28 country pairs × 100 conditions = 2800 signed differences:
%   x-axis : Human ΔP(intervene)  (country_i – country_j, one condition)
%   y-axis : AI ΔP(intervene)     (same pair, same condition)
%
% One subplot per AI model (4 per row), Pearson r in corner,
% regression line + 95% CI band.
%
% Colors: matched to task2_cultural_cond_r.pdf reference palette
% =========================================================================

clear; clc;

DATA_DIR = fullfile(fileparts(mfilename('fullpath')), 'data');
load(fullfile(DATA_DIR, 'fig3_data.mat'));

% ── USER PARAMETERS ───────────────────────────────────────────────────────
FONT_NAME     = 'Helvetica';
FONT_SIZE_AX  = 7.5;
FONT_SIZE_LB  = 8.5;
FONT_SIZE_TT  = 9;
FONT_SIZE_STAT= 7;

MARKER_SIZE  = 8;
MARKER_ALPHA = 0.35;
CI_ALPHA     = 0.12;
LINE_WIDTH   = 1.8;
N_COLS       = 4;

X_LIM = [-1.05, 1.05];
Y_LIM = [-1.10, 1.10];

% Model colors — alphabetical order (prob + logprobs, matches delta_ai_labels):
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

clean_name = @(s) strtrim(strrep(strrep(s,' (logprobs)',''),'_logprobs',' (logprobs)'));
% Keep method suffix but trim underscores
pretty_name = @(s) strrep(strrep(strrep(s,'_logprobs',' [logprobs]'),'_prob',' [prob]'),'_','–');

n_ai = double(n_ai_delta);

% ── Layout ────────────────────────────────────────────────────────────────
N_ROWS = ceil(n_ai / N_COLS);
figW   = N_COLS * 7.0;
figH   = N_ROWS * 6.5;

fig = figure('Units','centimeters','Position',[1 1 figW figH], ...
             'Color','white','PaperPositionMode','auto');

hx_all = pair_delta_human(:);

fprintf('\n=== Panel D: Pairwise Δ correlation (Human vs AI) ===\n');

for m = 1:n_ai
    col  = MODEL_COLORS(mod(m-1, size(MODEL_COLORS,1))+1, :);
    ay   = pair_delta_ai(m, :)';
    hx   = hx_all;

    valid = ~isnan(hx) & ~isnan(ay);
    hx_v  = hx(valid); ay_v = ay(valid); n_pts = sum(valid);

    if n_pts < 20
        subplot(N_ROWS, N_COLS, m);
        title(delta_ai_labels{m},'Interpreter','none');
        continue
    end

    % OLS regression + 95% CI
    X     = [ones(n_pts,1), hx_v];
    b     = X \ ay_v;
    r_val = corr(hx_v, ay_v, 'Type','Pearson');
    t_val = r_val * sqrt((n_pts-2)/(1-r_val^2));
    p_val = 2*(1-tcdf(abs(t_val), n_pts-2));

    x_fit  = linspace(-1, 1, 300)';
    y_fit  = b(1) + b(2)*x_fit;
    hm     = mean(hx_v);
    SS_x   = sum((hx_v-hm).^2);
    resid  = ay_v - (b(1) + b(2)*hx_v);
    MSE    = sum(resid.^2) / (n_pts-2);
    se_fit = sqrt(MSE * (1/n_pts + (x_fit-hm).^2/SS_x));
    t95    = tinv(0.975, n_pts-2);

    if p_val < 0.001, ps = 'p<.001'; else, ps = sprintf('p=%.3f',p_val); end
    fprintf('  %2d. %-40s  r=%.3f  %s\n', m, delta_ai_labels{m}, r_val, ps);

    % ── Draw subplot ──────────────────────────────────────────────────────
    ax = subplot(N_ROWS, N_COLS, m);
    hold(ax,'on');

    % Zero reference lines
    plot(ax,[-1 1],[0 0],'-','Color',[0.85 0.85 0.85],'LineWidth',0.6);
    plot(ax,[0 0],[-1 1],'-','Color',[0.85 0.85 0.85],'LineWidth',0.6);
    % 1:1 diagonal
    plot(ax,[-1 1],[-1 1],'--','Color',[0.75 0.75 0.75],'LineWidth',0.9);

    % CI band
    fill(ax,[x_fit; flipud(x_fit)], ...
         [y_fit+t95*se_fit; flipud(y_fit-t95*se_fit)], col, ...
         'FaceAlpha',CI_ALPHA,'EdgeColor','none');

    % Scatter (subsample for speed if very large)
    scatter(ax, hx_v, ay_v, MARKER_SIZE, col, 'filled', ...
            'MarkerFaceAlpha',MARKER_ALPHA,'MarkerEdgeColor','none');

    % Regression line
    plot(ax, x_fit, y_fit, '-', 'Color',col, 'LineWidth',LINE_WIDTH);

    % r/p text
    text(ax, X_LIM(1)+0.05, Y_LIM(2)-0.05, ...
         sprintf('r = %.3f\n%s', r_val, ps), ...
         'FontSize',FONT_SIZE_STAT,'FontName',FONT_NAME, ...
         'VerticalAlignment','top','Color',col*0.75, ...
         'BackgroundColor','white','Margin',1);

    % Title: model name (prettified)
    lbl_str = delta_ai_labels{m};
    lbl_str = strrep(lbl_str, '_logprobs', ' [logprobs]');
    lbl_str = strrep(lbl_str, '_prob',     ' [prob]');
    lbl_str = strrep(lbl_str, '_',         '-');

    title(ax, lbl_str, 'FontSize',FONT_SIZE_TT, 'FontName',FONT_NAME, ...
          'FontWeight','normal', 'Color','k', 'Interpreter','none');

    % Axes
    xlim(ax, X_LIM);  ylim(ax, Y_LIM);
    ax.XTick = -1:0.5:1;  ax.YTick = -1:0.5:1;
    ax.YGrid = 'on';
    ax.GridColor = [0.88 0.88 0.88];
    ax.GridLineStyle = '--';
    ax.GridAlpha = 0.6;
    ax.Box = 'off';  ax.Layer = 'top';
    ax.FontSize = FONT_SIZE_AX;  ax.FontName = FONT_NAME;
    ax.XColor = 'k';  ax.YColor = 'k';

    if mod(m-1, N_COLS) == 0
        ylabel(ax, 'AI  \DeltaP(intervene)', ...
               'FontSize',FONT_SIZE_LB,'FontName',FONT_NAME);
    end
    if m > (N_ROWS-1)*N_COLS
        xlabel(ax, 'Human  \DeltaP(intervene)', ...
               'FontSize',FONT_SIZE_LB,'FontName',FONT_NAME);
    end
end

sgtitle({'Cross-cultural sensitivity: Human vs. AI pairwise country differences', ...
         '(28 country pairs \times 100 conditions = 2800 data points per model)'}, ...
        'FontSize',11,'FontName',FONT_NAME,'FontWeight','bold');

% ── Save ──────────────────────────────────────────────────────────────────
OUT_DIR = fullfile(fileparts(mfilename('fullpath')), 'output');
if ~exist(OUT_DIR,'dir'), mkdir(OUT_DIR); end
print(fig, fullfile(OUT_DIR,'panel_D_pairwise_delta.pdf'), '-dpdf', '-r200', '-bestfit');
print(fig, fullfile(OUT_DIR,'panel_D_pairwise_delta.png'), '-dpng', '-r200');
fprintf('Saved panel_D_pairwise_delta.pdf/.png\n');
