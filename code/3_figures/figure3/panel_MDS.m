% =============================================================================
% Author : Xiaoyan Wu
% Date   : June 2026
% Description: Figure 3 — MDS 2D configuration plot for human and AI cultural spaces
% =============================================================================

% panel_MDS.m
% =========================================================================
% MDS plot of 8 countries in 100-condition behavioral space
% Bootstrap (N=1000) with Procrustes alignment → 95% confidence ellipses
%
% Country colors match panel_A_human_country.m
% =========================================================================

clear; clc;

DATA_DIR = fullfile(fileparts(mfilename('fullpath')), 'data');
load(fullfile(DATA_DIR, 'mds_data.mat'));

% ── PARAMETERS ────────────────────────────────────────────────────────────
FONT_NAME      = 'Helvetica';
FONT_SIZE_LBL  = 10;
FONT_SIZE_AX   = 8.5;
FONT_SIZE_TT   = 11;
FONT_SIZE_ANNOT= 7.5;

DOT_SIZE     = 80;
DOT_EDGE_W   = 1.5;
ELLIPSE_ALPHA= 0.18;   % fill transparency of 95% CI ellipse
CHI2_95      = 5.991;  % chi2 critical value (df=2, p=0.05)

figW = 11;
figH = 10;

% Country order (alphabetical from preprocess):
% Chile, Greece, Italy, Mexico, Poland, Portugal, South Africa, Spain
% Country colors — ALPHABETICAL order (from panel_A_DG_bar.m)
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

n_c = length(countries);

% ── Figure ────────────────────────────────────────────────────────────────
fig = figure('Units','centimeters','Position',[2 2 figW figH], ...
             'Color','white','PaperPositionMode','auto');

ax = axes(fig,'Units','normalized','Position',[0.12 0.12 0.80 0.78], ...
          'FontName',FONT_NAME,'FontSize',FONT_SIZE_AX,'Box','off');
hold(ax,'on');

ax.XGrid = 'on';  ax.YGrid = 'on';
ax.GridLineStyle = ':';
ax.GridAlpha = 0.28;
ax.LineWidth = 0.8;
ax.Box = 'off';

xline(ax, 0, '-', 'Color',[0.82 0.82 0.82], 'LineWidth',0.6);
yline(ax, 0, '-', 'Color',[0.82 0.82 0.82], 'LineWidth',0.6);

% ── 95% confidence ellipses from bootstrap covariance ─────────────────────
theta_e = linspace(0, 2*pi, 200);
unit_circle = [cos(theta_e); sin(theta_e)];   % (2, 200)

for ci = 1:n_c
    col  = COUNTRY_COLORS(ci,:);
    cov_ci = squeeze(boot_cov(ci,:,:));   % (2,2)
    mu_ci  = coords2(ci,:)';             % center on original MDS point

    % Ellipse axes from eigendecomposition of covariance
    [V, D_eig] = eig(cov_ci);
    % Scale to 95% CI: multiply by sqrt(chi2_95)
    radii = sqrt(diag(D_eig) * CHI2_95);
    ellipse_pts = V * diag(radii) * unit_circle + mu_ci;

    fill(ax, ellipse_pts(1,:), ellipse_pts(2,:), col, ...
         'FaceAlpha',ELLIPSE_ALPHA, 'EdgeColor','none');
end

% ── Country dots (MDS point estimate) ────────────────────────────────────
for ci = 1:n_c
    col = COUNTRY_COLORS(ci,:);
    scatter(ax, coords2(ci,1), coords2(ci,2), DOT_SIZE, col, 'filled', ...
            'MarkerEdgeColor','white', 'LineWidth',DOT_EDGE_W, 'ZData',2);
end

% ── Country labels ────────────────────────────────────────────────────────
label_nudge = [
     0.00  -0.028;   % Chile
    -0.01  -0.028;   % Greece
     0.00   0.025;   % Italy
     0.00   0.025;   % Mexico
    -0.01  -0.028;   % Poland
     0.018  0.012;   % Portugal
     0.00   0.025;   % South Africa
     0.018  0.012;   % Spain
];

for ci = 1:n_c
    col = COUNTRY_COLORS(ci,:);
    nx  = coords2(ci,1) + label_nudge(ci,1);
    ny  = coords2(ci,2) + label_nudge(ci,2);
    va  = 'bottom';
    if label_nudge(ci,2) < 0, va = 'top'; end

    text(ax, nx, ny, countries{ci}, ...
         'HorizontalAlignment','center','VerticalAlignment',va, ...
         'FontSize',FONT_SIZE_LBL,'FontName',FONT_NAME, ...
         'FontWeight','bold','Color',col*0.82);
end

% ── Labels & title ────────────────────────────────────────────────────────
xlabel(ax,'MDS Dimension 1','FontSize',FONT_SIZE_AX+0.5,'FontName',FONT_NAME);
ylabel(ax,'MDS Dimension 2','FontSize',FONT_SIZE_AX+0.5,'FontName',FONT_NAME);
title(ax,'Cultural Distance in Third-Party Intervention', ...
      'FontSize',FONT_SIZE_TT,'FontName',FONT_NAME,'FontWeight','bold','Color','k');

% ── Save ──────────────────────────────────────────────────────────────────
OUT_DIR = fullfile(fileparts(mfilename('fullpath')), 'output');
if ~exist(OUT_DIR,'dir'), mkdir(OUT_DIR); end
print(fig, fullfile(OUT_DIR,'panel_MDS.pdf'), '-dpdf', '-r300', '-bestfit');
print(fig, fullfile(OUT_DIR,'panel_MDS.png'), '-dpng', '-r300');
fprintf('Saved panel_MDS.pdf/.png\n');
