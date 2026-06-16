% =============================================================================
% Author : Xiaoyan Wu
% Date   : June 2026
% Description: Figure 3 panel F — DeepSeek vs human cultural geometry (MDS)
% =============================================================================

% panel_F_deepseek_vs_human.m
% =========================================================================
% Spotlight: DeepSeek-V4-Pro [logprobs] vs Human — "best" AI model comparison
%
% Three panels:
%   A. Human country profiles (8 lines × 100 conditions)
%   B. DeepSeek-V4-Pro [lp] country profiles (same layout, same scale)
%   C. RSA scatter: 28 country-pair similarities (Human vs AI)
%
% Purpose: show that even the best-performing model (Mantel ρ=0.523)
%          collapses cultural variation relative to humans
% =========================================================================

clear; clc;

DATA_DIR = fullfile(fileparts(mfilename('fullpath')), 'data');
load(fullfile(DATA_DIR, 'fig3_data.mat'));

FONT_NAME    = 'Helvetica';
FONT_SIZE_AX = 8;
FONT_SIZE_TT = 10;
FONT_SIZE_ST = 7.5;

% ── Country colors (alphabetical) ─────────────────────────────────────────
COUNTRY_COLORS = [
    0.831  0.106  0.224;   % Chile
    0.000  0.333  0.855;   % Greece
    0.000  0.700  0.450;   % Italy
    0.850  0.780  0.100;   % Mexico
    1.000  0.000  0.322;   % Poland
    0.349  0.698  0.573;   % Portugal
    0.900  0.600  0.100;   % South Africa
    1.000  0.380  0.376;   % Spain
];
full_labs = {'Chile','Greece','Italy','Mexico','Poland','Portugal','South Africa','Spain'};
N_CTRY = 8;

% ── Find DeepSeek-V4-Pro_logprobs index ───────────────────────────────────
model_target = 'DeepSeek-V4-Pro_logprobs';
target_idx = find(strcmp(delta_ai_labels, model_target));
if isempty(target_idx)
    error('Model %s not found in delta_ai_labels', model_target);
end
fprintf('Target model: %s  (index %d)\n', model_target, target_idx);
fprintf('  Mantel ρ = %.3f,  p = %.4f\n', mantel_r(target_idx), mantel_p(target_idx));

% ── Extract data ──────────────────────────────────────────────────────────
% h_cond_country : (8 × 100)  — already in fig3_data.mat
% ai_cond_country: (n_ai × 8 × 100)
h_mat  = h_cond_country;                              % (8 × 100)
ai_mat = squeeze(ai_cond_country(target_idx, :, :));  % (8 × 100)

% Sort 100 conditions by human mean (ascending) for cleaner display
cond_mean_h = nanmean(h_mat, 1);     % (1 × 100)
[~, sort_idx] = sort(cond_mean_h);
h_sorted  = h_mat(:, sort_idx);
ai_sorted = ai_mat(:, sort_idx);

% ── Compute spread statistics ──────────────────────────────────────────────
h_sd_per_cond  = nanstd(h_mat,  0, 1);  % (1×100) SD across 8 countries per condition
ai_sd_per_cond = nanstd(ai_mat, 0, 1);
mean_h_sd  = nanmean(h_sd_per_cond);
mean_ai_sd = nanmean(ai_sd_per_cond);

fprintf('\nMean SD across countries per condition:\n');
fprintf('  Human:               %.4f\n', mean_h_sd);
fprintf('  DeepSeek-V4-Pro [lp]: %.4f\n', mean_ai_sd);
fprintf('  Compression ratio:   %.1fx\n', mean_h_sd / mean_ai_sd);

% ── RSA: lower triangle (28 pairs) ────────────────────────────────────────
tril_mask = logical(tril(ones(N_CTRY), -1));
h_rdm_vec  = h_rdm(tril_mask);
ai_rdm_m   = squeeze(ai_rdm(target_idx, :, :));
ai_rdm_vec = ai_rdm_m(tril_mask);
valid_rsa   = ~isnan(h_rdm_vec) & ~isnan(ai_rdm_vec);

% Country pair labels for scatter
pair_labels = {};
pair_colors = zeros(N_CTRY*(N_CTRY-1)/2, 3);
ki = 0;
for i = 2:N_CTRY
    for j = 1:i-1
        ki = ki + 1;
        pair_labels{end+1} = sprintf('%s–%s', full_labs{i}(1:3), full_labs{j}(1:3)); %#ok
        pair_colors(ki,:) = (COUNTRY_COLORS(i,:) + COUNTRY_COLORS(j,:)) / 2;
    end
end

% ═══════════════════════════════════════════════════════════════════════════
% FIGURE
% ═══════════════════════════════════════════════════════════════════════════
fig = figure('Units','centimeters','Position',[1 1 40 14], ...
             'Color','white','PaperPositionMode','auto');

% ── Panel A: Human profiles ───────────────────────────────────────────────
ax1 = subplot(1, 3, 1);
hold(ax1, 'on');

% Shaded range (min–max across countries) per sorted condition
h_lo = min(h_sorted, [], 1);
h_hi = max(h_sorted, [], 1);
x_conds = 1:100;

fill(ax1, [x_conds, fliplr(x_conds)], [h_lo, fliplr(h_hi)], ...
     [0.85 0.85 0.85], 'EdgeColor','none', 'FaceAlpha',0.5);

for c = 1:N_CTRY
    plot(ax1, x_conds, h_sorted(c,:), '-', ...
         'Color',COUNTRY_COLORS(c,:), 'LineWidth',1.3, 'DisplayName',full_labs{c});
end

% Annotation: mean SD
text(ax1, 0.03, 0.97, sprintf('Mean SD = %.3f', mean_h_sd), ...
     'Units','normalized','HorizontalAlignment','left','VerticalAlignment','top', ...
     'FontSize',FONT_SIZE_ST,'FontName',FONT_NAME,'Color','k', ...
     'BackgroundColor','white','Margin',1);

title(ax1, 'Human (reference)', 'FontSize',FONT_SIZE_TT,'FontName',FONT_NAME, ...
      'FontWeight','bold','Color','k');
xlabel(ax1,'Conditions (sorted by human mean)','FontSize',FONT_SIZE_AX,'FontName',FONT_NAME);
ylabel(ax1,'P(intervene)','FontSize',FONT_SIZE_AX,'FontName',FONT_NAME);
ylim(ax1, [-0.05, 1.05]);
xlim(ax1, [1, 100]);
ax1.Box = 'off'; ax1.XColor = 'k'; ax1.YColor = 'k';
ax1.FontSize = FONT_SIZE_AX; ax1.FontName = FONT_NAME;

legend(ax1, full_labs, 'Location','northwest','FontSize',6,'FontName',FONT_NAME, ...
       'Box','off','NumColumns',2);

% ── Panel B: DeepSeek profiles ────────────────────────────────────────────
ax2 = subplot(1, 3, 2);
hold(ax2, 'on');

ai_lo = min(ai_sorted, [], 1, 'omitnan');
ai_hi = max(ai_sorted, [], 1, 'omitnan');

fill(ax2, [x_conds, fliplr(x_conds)], [ai_lo, fliplr(ai_hi)], ...
     [0.85 0.85 0.85], 'EdgeColor','none', 'FaceAlpha',0.5);

for c = 1:N_CTRY
    plot(ax2, x_conds, ai_sorted(c,:), '-', ...
         'Color',COUNTRY_COLORS(c,:), 'LineWidth',1.3);
end

text(ax2, 0.03, 0.97, sprintf('Mean SD = %.3f', mean_ai_sd), ...
     'Units','normalized','HorizontalAlignment','left','VerticalAlignment','top', ...
     'FontSize',FONT_SIZE_ST,'FontName',FONT_NAME,'Color','k', ...
     'BackgroundColor','white','Margin',1);

% Compression ratio badge
compress_str = sprintf('%.1f× narrower than human', mean_h_sd / mean_ai_sd);
text(ax2, 0.97, 0.03, compress_str, ...
     'Units','normalized','HorizontalAlignment','right','VerticalAlignment','bottom', ...
     'FontSize',FONT_SIZE_ST,'FontName',FONT_NAME,'Color',[0.7 0.1 0.1], ...
     'BackgroundColor','white','Margin',1);

title(ax2,'DeepSeek-V4-Pro  [logprobs]  ← "best" model', ...
      'FontSize',FONT_SIZE_TT,'FontName',FONT_NAME,'FontWeight','normal','Color',[0.45 0.25 0.60]);
xlabel(ax2,'Conditions (sorted by human mean)','FontSize',FONT_SIZE_AX,'FontName',FONT_NAME);
ylabel(ax2,'P(intervene)','FontSize',FONT_SIZE_AX,'FontName',FONT_NAME);
ylim(ax2, [-0.05, 1.05]);
xlim(ax2, [1, 100]);
ax2.Box = 'off'; ax2.XColor = 'k'; ax2.YColor = 'k';
ax2.FontSize = FONT_SIZE_AX; ax2.FontName = FONT_NAME;

% ── Panel C: RSA scatter ──────────────────────────────────────────────────
ax3 = subplot(1, 3, 3);
hold(ax3, 'on');

% OLS fit line
hv = h_rdm_vec(valid_rsa);
av = ai_rdm_vec(valid_rsa);
b  = [ones(sum(valid_rsa),1), hv] \ av;
x_fit = linspace(min(hv), max(hv), 200)';
y_fit = b(1) + b(2)*x_fit;

% 95% CI band
n_pts = sum(valid_rsa);
hm    = mean(hv);
resid = av - (b(1) + b(2)*hv);
MSE   = sum(resid.^2)/(n_pts-2);
se_fit = sqrt(MSE*(1/n_pts + (x_fit-hm).^2/sum((hv-hm).^2)));
t95    = tinv(0.975, n_pts-2);

fill(ax3, [x_fit; flipud(x_fit)], ...
     [y_fit+t95*se_fit; flipud(y_fit-t95*se_fit)], ...
     [0.45 0.25 0.60], 'FaceAlpha',0.12, 'EdgeColor','none');

% Scatter: each of 28 pairs colored by avg of two countries
ki = 0;
for i = 2:N_CTRY
    for j = 1:i-1
        ki = ki + 1;
        if valid_rsa(ki)
            scatter(ax3, h_rdm_vec(ki), ai_rdm_vec(ki), 55, ...
                    pair_colors(ki,:), 'filled', 'MarkerFaceAlpha',0.75, ...
                    'MarkerEdgeColor','w','LineWidth',0.4);
        end
    end
end

% Fit line
plot(ax3, x_fit, y_fit, '-','Color',[0.45 0.25 0.60],'LineWidth',2.2);

% 1:1 reference
lims = [min([hv;av])-0.02, max([hv;av])+0.02];
plot(ax3, lims, lims, '--', 'Color',[0.7 0.7 0.7],'LineWidth',1.0);

% Stats annotation
[rho_s, p_s] = corr(hv, av, 'Type','Spearman');
if mantel_p(target_idx) == 0
    p_str = sprintf('p < %.4f', 1/size(mantel_null,2));
else
    p_str = sprintf('p = %.3f', mantel_p(target_idx));
end

text(ax3, 0.05, 0.97, ...
     sprintf('Mantel \\rho = %.3f\n%s\nn = 28 country pairs', ...
             mantel_r(target_idx), p_str), ...
     'Units','normalized','HorizontalAlignment','left','VerticalAlignment','top', ...
     'FontSize',FONT_SIZE_ST,'FontName',FONT_NAME,'Color','k', ...
     'BackgroundColor','white','Margin',2);

% Note on effect size
text(ax3, 0.05, 0.05, ...
     sprintf('Only %.0f%% of variance\nin human structure explained', ...
             mantel_r(target_idx)^2 * 100), ...
     'Units','normalized','HorizontalAlignment','left','VerticalAlignment','bottom', ...
     'FontSize',FONT_SIZE_ST,'FontName',FONT_NAME,'Color',[0.7 0.1 0.1], ...
     'BackgroundColor','white','Margin',1);

title(ax3, 'Country-pair similarity structure (RSA)', ...
      'FontSize',FONT_SIZE_TT,'FontName',FONT_NAME,'FontWeight','bold','Color','k');
xlabel(ax3,'Human: country-pair similarity (Pearson r)', ...
       'FontSize',FONT_SIZE_AX,'FontName',FONT_NAME);
ylabel(ax3,'AI: country-pair similarity (Pearson r)', ...
       'FontSize',FONT_SIZE_AX,'FontName',FONT_NAME);
xlim(ax3, lims); ylim(ax3, lims);
axis(ax3,'square');
ax3.Box = 'off'; ax3.XColor = 'k'; ax3.YColor = 'k';
ax3.FontSize = FONT_SIZE_AX; ax3.FontName = FONT_NAME;

% Country legend for RSA panel
legend_h = gobjects(N_CTRY,1);
for c = 1:N_CTRY
    legend_h(c) = scatter(ax3, NaN, NaN, 30, COUNTRY_COLORS(c,:), 'filled');
end
legend(ax3, legend_h, full_labs, 'Location','southeast','FontSize',6, ...
       'FontName',FONT_NAME,'Box','off','NumColumns',2);

% ── Supertitle ────────────────────────────────────────────────────────────
sgtitle({'DeepSeek-V4-Pro [logprobs]: Best model at cultural structure reproduction', ...
         'Yet cultural variation is drastically compressed vs. human baseline'}, ...
        'FontSize',11,'FontName',FONT_NAME,'FontWeight','bold','Color','k');

% ── Save ──────────────────────────────────────────────────────────────────
OUT_DIR = fullfile(fileparts(mfilename('fullpath')), 'output');
if ~exist(OUT_DIR,'dir'), mkdir(OUT_DIR); end
print(fig, fullfile(OUT_DIR,'panel_F_deepseek_vs_human.pdf'), '-dpdf','-r200','-bestfit');
print(fig, fullfile(OUT_DIR,'panel_F_deepseek_vs_human.png'), '-dpng','-r200');
fprintf('\nSaved panel_F_deepseek_vs_human.pdf/.png\n');
