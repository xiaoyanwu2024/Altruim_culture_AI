% =============================================================================
% Author : Xiaoyan Wu
% Date   : June 2026
% Description: Figure 4 — spider chart of motive-cocktail parameter profiles
% =============================================================================

% panel_spider.m
% =========================================================================
% Motivational parameter profiles: Human reference vs. AI models (M8 Full)
%
% Design:
%   - Human ±1SD shown as gray filled band (reference)
%   - Human mean shown as dark dashed octagon
%   - Axis boundary = Human mean + 2SD (dashed circle)
%   - Each AI model: colored filled polygon + ±1SE band
%   - One shared legend at bottom
%   - Values beyond axis boundary clipped to 1.05; noted in caption
% =========================================================================

clear; clc;

DATA_DIR = fullfile(fileparts(mfilename('fullpath')), 'data');
load(fullfile(DATA_DIR, 'fig5_data.mat'));

% ── PARAMETERS ────────────────────────────────────────────────────────────
FONT_NAME       = 'Helvetica';
FONT_SIZE_PARAM = 9.5;
FONT_SIZE_TT    = 10;
FONT_SIZE_LEG   = 8.5;
FONT_SIZE_TICK  = 6.5;

COLOR_HUMAN_FILL = [0.80 0.80 0.80];   % ±1SD band
COLOR_HUMAN_LINE = [0.30 0.30 0.30];   % mean octagon
COLOR_MINI_LP    = [0.80 0.35 0.05];   % orange-red
COLOR_GPT54      = [0.45 0.35 0.75];   % purple
COLOR_GEM35      = [0.10 0.40 0.82];   % blue

AI_LINE_WIDTH  = 2.0;
HUM_LINE_WIDTH = 1.6;
AI_FILL_ALPHA  = 0.20;
AI_SE_ALPHA    = 0.12;
BAND_ALPHA     = 0.55;

figW = 22;
figH = 8.5;

% ── Data ──────────────────────────────────────────────────────────────────
N = length(param_names);

h_m   = h_mean_n(:);        h_sd  = h_std_n(:);
ml_m  = mini_lp_mean_n(:);  ml_s  = mini_lp_se_n(:);
g54_m = gpt54_mean_n(:);    g54_s = gpt54_se_n(:);
g35_m = gem35_mean_n(:);    g35_s = gem35_se_n(:);

% Clip AI values to [-0.05, 1.10] for display (±2SD boundary = 1.0)
CLIP_LO = -0.05;   CLIP_HI = 1.05;
ml_m_cl  = min(max(ml_m,  CLIP_LO), CLIP_HI);
g54_m_cl = min(max(g54_m, CLIP_LO), CLIP_HI);
g35_m_cl = min(max(g35_m, CLIP_LO), CLIP_HI);

% ── Geometry ──────────────────────────────────────────────────────────────
angles     = linspace(0, 2*pi, N+1)' - pi/2;
close_poly = @(v) [v(:); v(1)];
theta_full = linspace(0, 2*pi, 360);

param_lbl = {'\gamma', '\alpha', '\beta', '\lambda', '\omega', '\kappa', '\eta_{no}', '\eta_{yes}'};

% Human polygons
h_mean_poly  = close_poly(h_m);
h_p1sd_poly  = close_poly(min(h_m + h_sd, CLIP_HI));
h_m1sd_poly  = close_poly(max(h_m - h_sd, CLIP_LO));
boundary_r   = 1.0;   % = Human mean + 2SD

% ── Figure ────────────────────────────────────────────────────────────────
fig = figure('Units','centimeters','Position',[1 2 figW figH], ...
             'Color','white','PaperPositionMode','auto');

panels = {
    g54_m_cl, g54_s, g54_m, COLOR_GPT54,   'GPT-5.4';
    g35_m_cl, g35_s, g35_m, COLOR_GEM35,   'Gemini-3.5-Flash';
};
panel_titles = {
    'GPT-5.4';
    'Gemini-3.5-Flash';
};

% Store legend handles from last panel
leg_h  = [];
leg_lb = {};

for pi_ = 1:2
    ai_cl  = panels{pi_,1};   % clipped mean
    ai_se  = panels{pi_,2};
    ai_raw = panels{pi_,3};   % original (for out-of-range check)
    col    = panels{pi_,4};

    left_pos = 0.06 + (pi_-1)*0.44;
    ax = axes(fig,'Units','normalized', ...
              'Position',[left_pos 0.13 0.35 0.78]); %#ok<LAXES>
    axis(ax,'off');
    hold(ax,'on');
    ax.DataAspectRatio = [1 1 1];
    xlim(ax,[-1.48 1.48]);
    ylim(ax,[-1.48 1.48]);

    % ── Background: white fill inside boundary ──────────────────────────
    fill(ax, cos(theta_full), sin(theta_full), 'w', ...
         'FaceAlpha',1,'EdgeColor','none');

    % ── Human ±1SD band ─────────────────────────────────────────────────
    xhi = h_p1sd_poly .* cos(angles);   yhi = h_p1sd_poly .* sin(angles);
    xlo = h_m1sd_poly .* cos(angles);   ylo = h_m1sd_poly .* sin(angles);
    h_band = fill(ax, [xhi; flipud(xlo)], [yhi; flipud(ylo)], ...
                  COLOR_HUMAN_FILL, 'FaceAlpha',BAND_ALPHA, 'EdgeColor','none');

    % ── Radial grid lines (0.25, 0.5, 0.75) ─────────────────────────────
    for r = [0.25, 0.50, 0.75]
        plot(ax, r*cos(theta_full), r*sin(theta_full), '-', ...
             'Color',[0.78 0.78 0.78],'LineWidth',0.4);
    end

    % ── Boundary circle = Human mean+2SD ────────────────────────────────
    plot(ax, boundary_r*cos(theta_full), boundary_r*sin(theta_full), '--', ...
         'Color',[0.50 0.50 0.50],'LineWidth',0.9);

    % ── Spokes ──────────────────────────────────────────────────────────
    for k = 1:N
        plot(ax,[0, boundary_r*cos(angles(k))], ...
                [0, boundary_r*sin(angles(k))], '-', ...
             'Color',[0.72 0.72 0.72],'LineWidth',0.4);
    end

    % ── Human mean octagon (dashed) ─────────────────────────────────────
    xhm = h_mean_poly .* cos(angles);
    yhm = h_mean_poly .* sin(angles);
    h_hum = plot(ax, xhm, yhm, '--', 'Color',COLOR_HUMAN_LINE, ...
                 'LineWidth',HUM_LINE_WIDTH);

    % ── Radial tick labels (panel 1 only) ───────────────────────────────
    if pi_ == 1
        ang_lbl = angles(2) - 0.18;
        tick_v  = [0.25, 0.50, 0.75, 1.00];
        tick_lb = {'-1SD','mean','+1SD','+2SD'};
        for ti = 1:4
            text(ax, (tick_v(ti)+0.02)*cos(ang_lbl), ...
                     (tick_v(ti)+0.02)*sin(ang_lbl), tick_lb{ti}, ...
                 'FontSize',FONT_SIZE_TICK,'FontName',FONT_NAME, ...
                 'Color',[0.40 0.40 0.40],'HorizontalAlignment','right');
        end
    end

    % ── AI profile ──────────────────────────────────────────────────────
    ai_poly    = close_poly(ai_cl);
    ai_poly_hi = close_poly(min(ai_cl + ai_se, CLIP_HI));
    ai_poly_lo = close_poly(max(ai_cl - ai_se, CLIP_LO));

    xai  = ai_poly    .* cos(angles);   yai  = ai_poly    .* sin(angles);
    xahi = ai_poly_hi .* cos(angles);   yahi = ai_poly_hi .* sin(angles);
    xalo = ai_poly_lo .* cos(angles);   yalo = ai_poly_lo .* sin(angles);

    fill(ax, xai, yai, col, 'FaceAlpha',AI_FILL_ALPHA,'EdgeColor','none');
    fill(ax, [xahi; flipud(xalo)], [yahi; flipud(yalo)], col, ...
         'FaceAlpha',AI_SE_ALPHA,'EdgeColor','none');
    h_ai = plot(ax, xai, yai, '-', 'Color',col,'LineWidth',AI_LINE_WIDTH);
    scatter(ax, xai(1:end-1), yai(1:end-1), 28, col, 'filled', ...
            'MarkerEdgeColor','white','LineWidth',0.7, ...
            'ZData', ones(N,1)*3);

    % Mark params that exceed boundary with arrow-like label
    for k = 1:N
        if ai_raw(k) > 1.0
            ang_k = angles(k);
            text(ax, 1.13*cos(ang_k), 1.13*sin(ang_k), ...
                 ['\uparrow' sprintf('%.2f', ai_raw(k))], ...
                 'FontSize',6.5,'Color',col,'FontWeight','bold', ...
                 'HorizontalAlignment','center','VerticalAlignment','middle', ...
                 'Interpreter','tex');
        elseif ai_raw(k) < 0
            ang_k = angles(k);
            text(ax, 0.10*cos(ang_k), 0.10*sin(ang_k), ...
                 ['\downarrow' sprintf('%.2f', ai_raw(k))], ...
                 'FontSize',6.5,'Color',col,'FontWeight','bold', ...
                 'HorizontalAlignment','center','VerticalAlignment','middle', ...
                 'Interpreter','tex');
        end
    end

    % ── Parameter labels ────────────────────────────────────────────────
    label_r = 1.22;
    for k = 1:N
        xk = label_r * cos(angles(k));
        yk = label_r * sin(angles(k));
        if     cos(angles(k)) >  0.2, ha = 'left';
        elseif cos(angles(k)) < -0.2, ha = 'right';
        else,                           ha = 'center'; end
        if     sin(angles(k)) >  0.2, va = 'bottom';
        elseif sin(angles(k)) < -0.2, va = 'top';
        else,                           va = 'middle'; end
        if k >= 7   % eta_no, eta_yes — subscripts don't need bold
            fw = 'normal';
        else
            fw = 'bold';
        end
        text(ax, xk, yk, param_lbl{k}, ...
             'HorizontalAlignment',ha,'VerticalAlignment',va, ...
             'FontSize',FONT_SIZE_PARAM,'FontName',FONT_NAME, ...
             'FontWeight',fw,'Color',[0.10 0.10 0.10], ...
             'Interpreter','tex');
    end

    % ── Panel title ─────────────────────────────────────────────────────
    title(ax, panel_titles{pi_}, ...
          'FontSize',FONT_SIZE_TT,'FontName',FONT_NAME,'FontWeight','bold', ...
          'Units','normalized','Position',[0.5 1.05 0],'Color','k');

    % Collect handles for shared legend (panel 3)
    if pi_ == 3
        leg_h  = [h_hum, h_band, h_ai];
        leg_lb = {'Human mean', 'Human \pm1SD', panels{pi_,5}};
    end
    if pi_ == 1 || pi_ == 2
        % invisible dummy for uniform legend entries
    end
end

% ── Shared legend (right side, vertical) ─────────────────────────────────
colors_leg = {COLOR_GPT54, COLOR_GEM35};
ai_labels  = {'GPT-5.4  (N=16)', 'Gemini-3.5-Flash  (N=16)'};

ax_leg = axes(fig,'Units','normalized','Position',[0.91 0.1 0.01 0.8],'Visible','off');
hold(ax_leg,'on');
h_hum_leg  = plot(ax_leg, NaN, NaN, '--', 'Color',COLOR_HUMAN_LINE,'LineWidth',1.6);
h_band_leg = patch(ax_leg,[0 1 1 0],[0 0 1 1],COLOR_HUMAN_FILL, ...
                   'FaceAlpha',BAND_ALPHA,'EdgeColor','none');
ai_handles = gobjects(2,1);
for ci = 1:2
    ai_handles(ci) = plot(ax_leg, NaN, NaN, '-o', ...
        'Color',colors_leg{ci},'LineWidth',1.8,'MarkerSize',5, ...
        'MarkerFaceColor',colors_leg{ci},'MarkerEdgeColor','white');
end

legend(ax_leg, [h_hum_leg, h_band_leg, ai_handles(1), ai_handles(2)], ...
       {'Human mean', 'Human \pm1SD', ai_labels{1}, ai_labels{2}}, ...
       'Orientation','vertical','Location','east', ...
       'FontSize',FONT_SIZE_LEG,'FontName',FONT_NAME,'Box','off', ...
       'Interpreter','tex');

% ── Save ──────────────────────────────────────────────────────────────────
OUT_DIR = fullfile(fileparts(mfilename('fullpath')), 'output');
if ~exist(OUT_DIR,'dir'), mkdir(OUT_DIR); end
print(fig, fullfile(OUT_DIR,'panel_spider.pdf'), '-dpdf', '-r300', '-bestfit');
print(fig, fullfile(OUT_DIR,'panel_spider.png'), '-dpng', '-r300');
fprintf('Saved panel_spider.pdf/.png\n');
