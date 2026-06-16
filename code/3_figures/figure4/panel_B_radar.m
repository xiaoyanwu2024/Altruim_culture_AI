% =============================================================================
% Author : Xiaoyan Wu
% Date   : June 2026
% Description: Figure 4 panel B — radar/spider chart of M8 parameter profiles
% =============================================================================

% panel_B_radar.m
% =========================================================================
% Panel B — Motivational profile radar chart
%           Human vs. GPT-4o-mini [lp] vs. GPT-5.4 vs. Gemini-3.5-Flash
%           (All three AI models selected M8 Full as best cognitive model)
%
% Parameters: gama, envy, guilt, lamda, oumiga, kapa, etak, etaa (M8, first 8)
% Normalization: per-axis, shift floor to 0 then scale to max=1
%   (handles kapa which can be negative)
% =========================================================================

clear; clc;

DATA_DIR = fullfile(fileparts(mfilename('fullpath')), 'data');
load(fullfile(DATA_DIR, 'fig4_data.mat'));

% ── USER PARAMETERS ───────────────────────────────────────────────────────
FONT_NAME       = 'Helvetica';
FONT_SIZE_PARAM = 10;
FONT_SIZE_TT    = 12;
FONT_SIZE_LEG   = 9;

COLOR_HUMAN    = [0.80 0.12 0.12];   % red
COLOR_MINI_LP  = [0.80 0.45 0.05];   % orange  (GPT-4o-mini [lp])
COLOR_GPT54    = [0.60 0.55 0.80];   % purple  (GPT-5.4)
COLOR_GEM35    = [0.10 0.40 0.82];   % blue    (Gemini-3.5-Flash)

FILL_ALPHA       = 0.18;
LINE_WIDTH_HUMAN = 3.0;
LINE_WIDTH_AI    = 2.0;
SHOW_SE_BAND     = true;
SE_ALPHA         = 0.10;

figW = 13;
figH = 12;

% ── Load raw means, force column vectors ──────────────────────────────────
h_m    = h_mean_norm(:);      h_se_v  = h_se_norm(:);
ml_m   = mini_lp_mean_norm(:); ml_se  = mini_lp_se_norm(:);
g54_m  = gpt54_mean_norm(:);  g54_se  = gpt54_se_norm(:);
g35_m  = gem35_mean_norm(:);  g35_se  = gem35_se_norm(:);

N_PARAMS = length(h_m);

% ── Per-axis normalization ────────────────────────────────────────────────
% For each axis: floor = min(0, min of all group lower bounds)
%                ceiling = max of all group upper bounds
ax_lo  = zeros(N_PARAMS, 1);
ax_hi  = zeros(N_PARAMS, 1);
for k = 1:N_PARAMS
    lo = min([h_m(k)-h_se_v(k), ml_m(k)-ml_se(k), ...
              g54_m(k)-g54_se(k), g35_m(k)-g35_se(k)]);
    hi = max([h_m(k)+h_se_v(k), ml_m(k)+ml_se(k), ...
              g54_m(k)+g54_se(k), g35_m(k)+g35_se(k)]);
    ax_lo(k) = min(lo, 0);
    ax_hi(k) = max(hi, 1e-6);
end
ax_rng = ax_hi - ax_lo;
ax_rng(ax_rng < 1e-9) = 1;

nm = @(v, se_flag) (v - ax_lo) ./ ax_rng;          % shift+scale means
ns = @(v)           v           ./ ax_rng;           % scale SEs only

h_s    = nm(h_m,   0);   h_se_s  = ns(h_se_v);
ml_s   = nm(ml_m,  0);   ml_se_s = ns(ml_se);
g54_s  = nm(g54_m, 0);   g54_se_s= ns(g54_se);
g35_s  = nm(g35_m, 0);   g35_se_s= ns(g35_se);

% ── Radar geometry ────────────────────────────────────────────────────────
angles = linspace(0, 2*pi, N_PARAMS+1)';
angles = angles - pi/2;   % start at top
close_poly = @(v) [v(:); v(1)];

% ── Figure ────────────────────────────────────────────────────────────────
fig = figure('Units','centimeters','Position',[2 2 figW figH], ...
             'Color','white','PaperPositionMode','auto');

ax = axes(fig,'Units','normalized','Position',[0.05 0.05 0.90 0.85]);
axis(ax,'off');
hold(ax,'on');
ax.DataAspectRatio = [1 1 1];
xlim(ax, [-1.55, 1.55]);
ylim(ax, [-1.55, 1.55]);

% ── Radial grid ───────────────────────────────────────────────────────────
theta_full = linspace(0, 2*pi, 360);
for r = [0.25, 0.5, 0.75]
    plot(ax, r*cos(theta_full), r*sin(theta_full), '-', ...
         'Color',[0.82 0.82 0.82], 'LineWidth',0.5);
    ang_lbl = angles(1) + 0.18;
    text(ax, (r+0.03)*cos(ang_lbl), (r+0.03)*sin(ang_lbl), ...
         sprintf('%d%%', round(r*100)), ...
         'FontSize',7,'FontName',FONT_NAME,'Color',[0.55 0.55 0.55], ...
         'HorizontalAlignment','left');
end
plot(ax, cos(theta_full), sin(theta_full), '-', ...
     'Color',[0.55 0.55 0.55],'LineWidth',1.0);

% Spokes
for k = 1:N_PARAMS
    plot(ax, [0, cos(angles(k))], [0, sin(angles(k))], '-', ...
         'Color',[0.78 0.78 0.78],'LineWidth',0.5);
end

% ── Parameter labels ──────────────────────────────────────────────────────
param_display = {'\gamma (risk)', 'envy', 'guilt', '\lambda (loss)', ...
                 '\omega (SCI)', '\kappa (RP)', '\eta_k', '\eta_a'};
label_r = 1.25;
for k = 1:N_PARAMS
    xk = label_r * cos(angles(k));
    yk = label_r * sin(angles(k));
    if     cos(angles(k)) >  0.3, ha = 'left';
    elseif cos(angles(k)) < -0.3, ha = 'right';
    else,                          ha = 'center'; end
    if     sin(angles(k)) >  0.3, va = 'bottom';
    elseif sin(angles(k)) < -0.3, va = 'top';
    else,                          va = 'middle'; end
    text(ax, xk, yk, param_display{k}, ...
         'HorizontalAlignment',ha,'VerticalAlignment',va, ...
         'FontSize',FONT_SIZE_PARAM,'FontName',FONT_NAME,'FontWeight','bold', ...
         'Color',[0.15 0.15 0.15],'Interpreter','tex');
end

% ── Draw profiles (AI first, Human last = on top) ─────────────────────────
profiles = {
    ml_s,  ml_se_s,  COLOR_MINI_LP, LINE_WIDTH_AI,    'GPT-4o-mini [lp]  (N=16)';
    g54_s, g54_se_s, COLOR_GPT54,   LINE_WIDTH_AI,    'GPT-5.4  (N=16)';
    g35_s, g35_se_s, COLOR_GEM35,   LINE_WIDTH_AI,    'Gemini-3.5-Flash  (N=16)';
    h_s,   h_se_s,   COLOR_HUMAN,   LINE_WIDTH_HUMAN, 'Human  (N=875)';
};

leg_handles = gobjects(size(profiles,1), 1);
for p = 1:size(profiles,1)
    vals  = profiles{p,1};
    svals = profiles{p,2};
    col   = profiles{p,3};
    lw    = profiles{p,4};

    rv  = close_poly(vals);
    xv  = rv .* cos(angles);
    yv  = rv .* sin(angles);

    fill(ax, xv, yv, col, 'FaceAlpha',FILL_ALPHA, 'EdgeColor','none');

    if SHOW_SE_BAND
        rv_hi = close_poly(min(vals + svals, 1.0));
        rv_lo = close_poly(max(vals - svals, 0.0));
        xhi = rv_hi .* cos(angles);   yhi = rv_hi .* sin(angles);
        xlo = rv_lo .* cos(angles);   ylo = rv_lo .* sin(angles);
        fill(ax, [xhi; flipud(xlo)], [yhi; flipud(ylo)], col, ...
             'FaceAlpha',SE_ALPHA, 'EdgeColor','none');
    end

    h_line = plot(ax, xv, yv, '-', 'Color',col, 'LineWidth',lw);
    scatter(ax, xv(1:end-1), yv(1:end-1), 28, col, 'filled', ...
            'MarkerEdgeColor','white','LineWidth',0.8,'ZData',zeros(N_PARAMS,1)+2);

    leg_handles(p) = h_line;
end

% ── Legend ────────────────────────────────────────────────────────────────
legend(ax, leg_handles, profiles(:,5), ...
       'Location','southoutside','Orientation','horizontal', ...
       'FontSize',FONT_SIZE_LEG,'FontName',FONT_NAME,'Box','off','NumColumns',2);

% ── Title ─────────────────────────────────────────────────────────────────
title(ax, 'B   Motivational profiles: Human vs. M8-Full AI models', ...
      'FontSize',FONT_SIZE_TT,'FontName',FONT_NAME,'FontWeight','bold', ...
      'Units','normalized','Position',[0.5 1.04 0]);

text(ax, 0, -1.50, ...
     'Each axis: independently scaled to its range across all groups (100% = group maximum).  Shaded = ±1 SE.', ...
     'HorizontalAlignment','center','VerticalAlignment','top', ...
     'FontSize',7,'FontName',FONT_NAME,'Color',[0.50 0.50 0.50]);

% ── Save ──────────────────────────────────────────────────────────────────
OUT_DIR = fullfile(fileparts(mfilename('fullpath')), 'output');
if ~exist(OUT_DIR,'dir'), mkdir(OUT_DIR); end
print(fig, fullfile(OUT_DIR,'panel_B_radar.pdf'), '-dpdf', '-r300', '-bestfit');
print(fig, fullfile(OUT_DIR,'panel_B_radar.png'), '-dpng', '-r300');
fprintf('Saved panel_B_radar.pdf/.png\n');
