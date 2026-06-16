% =============================================================================
% Author : Xiaoyan Wu
% Date   : June 2026
% Description: Supplementary Figure S2 — TPI bar charts (by country) and
%              scatter plots (AI vs human condition means) for all 9 AI models
% =============================================================================

% FigureS2_TPI_bar_allmodels.m
% =========================================================================
% Supplementary Fig S2: TPI — bar (by country) + scatter (vs human) per model
% Layout: 3 model-units per row, each unit = [bar | scatter] side by side
%         6 subplots per row; Human = bar only (scatter slot = country legend)
% Format: Arial, 8pt axis labels, 9pt panel letters (top-left, no bold)
% =========================================================================

clear; clc;

DATA_DIR = fullfile(fileparts(mfilename('fullpath')), 'data');
load(fullfile(DATA_DIR, 'tpp_data.mat'));

% ── FORMAT PARAMETERS ─────────────────────────────────────────────────────
FONT_NAME       = 'Arial';
FONT_SIZE_AX    = 8;
FONT_SIZE_LB    = 8;
FONT_SIZE_PANEL = 9;
FONT_SIZE_STAT  = 6.5;

BAR_W     = 0.65;
BAR_ALPHA = 0.88;
Y_MIN_BAR = 0.0;
Y_MAX_BAR = 0.60;

MARKER_SIZE  = 18;
MARKER_ALPHA = 0.55;
CI_ALPHA     = 0.12;
LINE_WIDTH   = 1.5;
X_LIM_SC = [0.0, 1.05];
Y_LIM_SC = [-0.05, 1.10];

UNITS_PER_ROW = 3;   % model-units per row
N_COLS        = UNITS_PER_ROW * 2;   % 6 subplot columns

% Country colors — ALPHABETICAL (Chile Greece Italy Mexico Poland Portugal South Africa Spain)
COUNTRY_COLORS = [
    0.831  0.106  0.224;
    0.000  0.333  0.855;
    0.000  0.776  0.553;
    0.996  0.929  0.255;
    1.000  0.000  0.322;
    0.349  0.698  0.573;
    1.000  0.796  0.302;
    1.000  0.380  0.376;
];

MODEL_COLORS = [
    0.92  0.35  0.22;
    0.17  0.74  0.67;
    0.72  0.79  0.41;
    0.94  0.78  0.19;
    0.34  0.53  0.80;
    0.55  0.78  0.29;
    0.96  0.63  0.13;
    0.49  0.78  0.63;
    0.60  0.55  0.80;
];

clean_name   = @(s) strtrim(strrep(strrep(s,' (logprobs)',''),'_logprobs',''));
panel_letter = @(i) char('a' + i - 1);

% ── Filter: remove logprobs duplicates only when a prob version also exists
clean_fn = @(s) strtrim(strrep(strrep(s,' (logprobs)',''),'_logprobs',''));
keep = true(length(ai_tpp_names), 1);
for ii = 1:length(ai_tpp_names)
    if contains(ai_tpp_names{ii}, 'logprobs', 'IgnoreCase', true)
        base = clean_fn(ai_tpp_names{ii});
        prob_exists = any(cellfun(@(s) strcmpi(clean_fn(s), base) && ...
                          ~contains(s,'logprobs','IgnoreCase',true), ai_tpp_names));
        if prob_exists, keep(ii) = false; end
    end
end
ai_tpp_names        = ai_tpp_names(keep);
ai_tpp_r            = ai_tpp_r(keep);
ai_tpp_mae          = ai_tpp_mae(keep);
ai_tpp_country_mean = ai_tpp_country_mean(keep, :);
ai_tpp_mean         = ai_tpp_mean(keep, :);

n_models = double(length(ai_tpp_names));

% ── Sort AI models by r ───────────────────────────────────────────────────
[r_sorted, r_idx]   = sort(ai_tpp_r, 'descend');
name_sorted         = ai_tpp_names(r_idx);
country_mean_sorted = ai_tpp_country_mean(r_idx, :);
cond_mean_sorted    = ai_tpp_mean(r_idx, :);

% ── Country sort by human mean ────────────────────────────────────────────
n_ctry = length(countries);
[~, sort_ord] = sort(h_tpp_country_mean, 'descend');
h_sorted   = h_tpp_country_mean(sort_ord);
ctry_sorted = countries(sort_ord);
col_sorted  = COUNTRY_COLORS(sort_ord, :);

% ── Human condition vector ────────────────────────────────────────────────
if ~exist('h_vec','var')
    h_vec = h_tpp_cond_mean(:);
end

% ── Layout ────────────────────────────────────────────────────────────────
% unit index: 1=Human, 2..n_models+1 = AI models
N_UNITS = 1 + n_models;
N_ROWS  = ceil(N_UNITS / UNITS_PER_ROW);
figW    = N_COLS * 6.8;    % 6 cols × 6.8 cm
figH    = N_ROWS * 6.2;

fig = figure('Units','centimeters','Position',[1 1 figW figH], ...
             'Color','white','PaperPositionMode','auto');

fprintf('\n=== FigureS2: TPI bar + scatter (9 prob models) ===\n');

for u = 1:N_UNITS
    % Grid position of this unit
    row_u    = ceil(u / UNITS_PER_ROW);          % 1-based row
    col_u    = mod(u-1, UNITS_PER_ROW);          % 0-based unit column
    bar_slot = (row_u-1)*N_COLS + col_u*2 + 1;
    sct_slot = (row_u-1)*N_COLS + col_u*2 + 2;

    % ── BAR ───────────────────────────────────────────────────────────────
    ax_b = subplot(N_ROWS, N_COLS, bar_slot);
    hold(ax_b, 'on');

    if u == 1
        patch(ax_b, [0.5 n_ctry+0.5 n_ctry+0.5 0.5], ...
              [Y_MIN_BAR Y_MIN_BAR Y_MAX_BAR Y_MAX_BAR], [0.94 0.94 0.94], ...
              'FaceAlpha',0.5,'EdgeColor','none');
        vals = h_sorted;
        ttl  = 'Human';
    else
        m    = u - 1;
        vals = country_mean_sorted(m, sort_ord);
        ttl  = clean_name(name_sorted{m});
    end

    for c = 1:n_ctry
        bar(ax_b, c, vals(c), BAR_W, ...
            'FaceColor',col_sorted(c,:),'EdgeColor','none','FaceAlpha',BAR_ALPHA);
    end

    if u > 1 && contains(clean_name(name_sorted{u-1}), 'DeepSeek-V4-Flash', 'IgnoreCase', true)
        ylim(ax_b, [0, 0.10]);
        ax_b.YTick = [0 0.05 0.10];
    else
        ylim(ax_b, [Y_MIN_BAR Y_MAX_BAR]);
        ax_b.YTick = 0:0.2:Y_MAX_BAR;
    end
    xlim(ax_b, [0.3, n_ctry+0.7]);
    ax_b.XTick = 1:n_ctry;
    ax_b.XTickLabel = ctry_sorted;
    ax_b.XTickLabelRotation = 45;
    ax_b.YGrid = 'on';
    ax_b.GridColor = [0.85 0.85 0.85];
    ax_b.GridLineStyle = '--';
    ax_b.GridAlpha = 0.55;
    ax_b.Box = 'off';
    ax_b.Layer = 'top';
    ax_b.FontSize = FONT_SIZE_AX;
    ax_b.FontName = FONT_NAME;
    ax_b.XColor = 'k'; ax_b.YColor = 'k';
    ax_b.TickLabelInterpreter = 'none';

    title(ax_b, ttl, 'FontSize',FONT_SIZE_AX, 'FontName',FONT_NAME, ...
          'FontWeight','normal', 'Color','k', 'Interpreter','none');
    ylabel(ax_b, 'P(intervene)', 'FontSize',FONT_SIZE_LB, 'FontName',FONT_NAME);

    text(ax_b, 0.02, 0.97, panel_letter((u-1)*2+1), ...
         'Units','normalized', 'FontSize',FONT_SIZE_PANEL, ...
         'FontName',FONT_NAME, 'FontWeight','normal', ...
         'HorizontalAlignment','left', 'VerticalAlignment','top', 'Color','k');

    % ── SCATTER ───────────────────────────────────────────────────────────
    ax_s = subplot(N_ROWS, N_COLS, sct_slot);

    if u == 1
        % Country legend in human's scatter slot
        axis(ax_s, 'off');
        leg_h = gobjects(n_ctry,1);
        for c = 1:n_ctry
            leg_h(c) = patch(ax_s, NaN, NaN, col_sorted(c,:), ...
                             'EdgeColor','none','FaceAlpha',BAR_ALPHA);
        end
        legend(ax_s, leg_h, ctry_sorted, ...
               'FontSize',FONT_SIZE_AX, 'FontName',FONT_NAME, ...
               'Box','off','NumColumns',1,'Location','west');
        continue
    end

    m   = u - 1;
    col = MODEL_COLORS(mod(m-1, size(MODEL_COLORS,1))+1, :);
    ay  = cond_mean_sorted(m, :);  ay = ay(:);
    hx  = h_vec(:);
    n_min = min(numel(hx), numel(ay));
    hx = hx(1:n_min);  ay = ay(1:n_min);
    valid = ~isnan(hx) & ~isnan(ay);
    hx_v = hx(valid);  ay_v = ay(valid);  n_pts = sum(valid);

    X     = [ones(n_pts,1), hx_v];
    b     = X \ ay_v;
    r_val = corr(hx_v, ay_v);
    t_val = r_val * sqrt((n_pts-2)/(1-r_val^2));
    p_val = 2*(1-tcdf(abs(t_val), n_pts-2));

    x_fit  = linspace(0, 1, 200)';
    y_fit  = b(1) + b(2)*x_fit;
    hm     = mean(hx_v);
    SS_x   = sum((hx_v-hm).^2);
    se_fit = sqrt(mean((ay_v-(b(1)+b(2)*hx_v)).^2) .* ...
                  (1/n_pts + (x_fit-hm).^2/SS_x));
    t95    = tinv(0.975, n_pts-2);

    if p_val < 0.001, ps='p<.001'; else, ps=sprintf('p=%.3f',p_val); end
    fprintf('  %2d. %-30s  r=%.3f  %s\n', m, name_sorted{m}, r_val, ps);

    hold(ax_s, 'on');
    plot(ax_s,[0 1],[0 1],'--','Color',[0.75 0.75 0.75],'LineWidth',0.8);
    fill(ax_s,[x_fit; flipud(x_fit)], ...
         [y_fit+t95*se_fit; flipud(y_fit-t95*se_fit)], col, ...
         'FaceAlpha',CI_ALPHA,'EdgeColor','none');
    scatter(ax_s, hx_v, ay_v, MARKER_SIZE, col, 'filled', ...
            'MarkerFaceAlpha',MARKER_ALPHA,'MarkerEdgeColor','none');
    plot(ax_s, x_fit, y_fit, '-', 'Color',col, 'LineWidth',LINE_WIDTH);

    text(ax_s, 0.12, 0.97, sprintf('r=%.3f\n%s', r_val, ps), ...
         'Units','normalized', 'FontSize',FONT_SIZE_STAT, ...
         'FontName',FONT_NAME, 'VerticalAlignment','top', 'Color',col*0.75);

    xlim(ax_s, X_LIM_SC);  ylim(ax_s, Y_LIM_SC);
    ax_s.XTick = 0:0.5:1;  ax_s.YTick = 0:0.5:1;
    ax_s.YGrid = 'on';
    ax_s.GridColor = [0.88 0.88 0.88];
    ax_s.GridLineStyle = '--';
    ax_s.GridAlpha = 0.6;
    ax_s.Box = 'off';
    ax_s.Layer = 'top';
    ax_s.FontSize = FONT_SIZE_AX;
    ax_s.FontName = FONT_NAME;
    ax_s.XColor = 'k'; ax_s.YColor = 'k';

    xlabel(ax_s,'Human  P(int.)','FontSize',FONT_SIZE_LB,'FontName',FONT_NAME);
    ylabel(ax_s,'AI  P(int.)','FontSize',FONT_SIZE_LB,'FontName',FONT_NAME);

    text(ax_s, 0.02, 0.97, panel_letter((u-1)*2+2), ...
         'Units','normalized', 'FontSize',FONT_SIZE_PANEL, ...
         'FontName',FONT_NAME, 'FontWeight','normal', ...
         'HorizontalAlignment','left', 'VerticalAlignment','top', 'Color','k');
end

% ── Save ──────────────────────────────────────────────────────────────────
OUT_DIR   = fullfile(fileparts(mfilename('fullpath')), 'output');
PAPER_DIR = '/Users/wuxiaoyan/Desktop/TPP_culture_AI/manuscript/AI_Culture_altruism';
if ~exist(OUT_DIR,'dir'), mkdir(OUT_DIR); end

set(fig, 'PaperUnits','centimeters', 'PaperSize',[figW figH], ...
         'PaperPosition',[0 0 figW figH]);
print(fig, fullfile(OUT_DIR,'FigureS2_TPI_bar_allmodels.pdf'), '-dpdf', '-r300');
print(fig, fullfile(OUT_DIR,'FigureS2_TPI_bar_allmodels.png'), '-dpng', '-r300');
print(fig, fullfile(PAPER_DIR,'FigureS2.pdf'), '-dpdf', '-r300');
fprintf('Saved FigureS2.pdf\n');
