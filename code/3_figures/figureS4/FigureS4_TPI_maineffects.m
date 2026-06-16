% =============================================================================
% Author : Xiaoyan Wu
% Date   : June 2026
% Description: Supplementary Figure S4 — TPI directional sensitivity (Scenario/Inequality/Ratio/Cost)
% =============================================================================

% FigureS4_TPI_maineffects.m
% =========================================================================
% Supplementary Fig S4: TPI directional sensitivity
% Layout: 5 rows × 2 model-groups side by side (left: units 1-5, right: 6-10)
%         subplot(5,9,...) — cols 1-4 = left, col 5 = gap, cols 6-9 = right
% Significance brackets: Scenario/Ratio = paired t-test (n=50)
%                        Inequality/Cost = adjacent paired t-tests (n=5)
% Format: Arial, 8pt axis, 9pt panel letters (left column)
% =========================================================================

clear; clc;

DATA_DIR = fullfile(fileparts(mfilename('fullpath')), '..', 'Figure2', 'data');
load(fullfile(DATA_DIR, 'tpp_data.mat'));

% ── FORMAT PARAMETERS ─────────────────────────────────────────────────────
FONT_NAME       = 'Arial';
FONT_SIZE_AX    = 7;
FONT_SIZE_LB    = 8;
FONT_SIZE_TT    = 7.5;
FONT_SIZE_ROW   = 8;
FONT_SIZE_PANEL = 9;
FONT_SIZE_STAR  = 8;

C_PUNISH  = [0.910, 0.592, 0.357];
C_HELP    = [0.522, 0.757, 0.914];
C_INEQ    = [0.996 0.850 0.200;
             0.600 0.820 0.200;
             0.200 0.680 0.200;
             0.100 0.580 0.320;
             0.000 0.407 0.216];
C_RATIO_1 = [0.610, 0.350, 0.714];
C_RATIO_2 = [0.380, 0.200, 0.460];
C_COST    = [0.941, 0.380, 0.573];

SCENARIO_LABELS = {'Punish','Help'};
INEQ_LABELS     = {'50:50','60:40','70:30','80:20','90:10'};
RATIO_LABELS_TX = {'\times1.5','\times3'};
COST_LABELS     = {'10','20','30','40','50'};
COL_TITLES      = {'Scenario','Inequality','Ratio','Cost'};

clean_name   = @(s) strtrim(strrep(strrep(s,' (logprobs)',''),'_logprobs',''));
panel_letter = @(i) char('a' + i - 1);

% ── Filter: remove logprobs duplicates only when a prob version also exists
% (keeps logprobs-only models like DeepSeek-V4-Flash)
clean_fn = @(s) strtrim(strrep(strrep(s,' (logprobs)',''),'_logprobs',''));
keep = true(length(ai_tpp_names), 1);
for ii = 1:length(ai_tpp_names)
    if contains(ai_tpp_names{ii}, 'logprobs', 'IgnoreCase', true)
        base = clean_fn(ai_tpp_names{ii});
        prob_exists = any(cellfun(@(s) strcmpi(clean_fn(s), base) && ...
                          ~contains(s,'logprobs','IgnoreCase',true), ai_tpp_names));
        if prob_exists
            keep(ii) = false;
        end
    end
end
ai_tpp_names = ai_tpp_names(keep);
ai_tpp_r     = ai_tpp_r(keep);
ai_heatmap   = ai_heatmap(keep, :, :, :, :);

[r_sorted, r_idx] = sort(ai_tpp_r, 'descend');
ai_tpp_names = ai_tpp_names(r_idx);
ai_heatmap   = ai_heatmap(r_idx, :, :, :, :);
n_models     = double(length(ai_tpp_names));

% ── Marginal means ────────────────────────────────────────────────────────
h_scenario   = squeeze(mean(mean(mean(h_heatmap, 4), 3), 2));
h_ratio      = squeeze(mean(mean(mean(h_heatmap, 4), 3), 1));
h_cost       = squeeze(mean(mean(mean(h_heatmap, 4), 2), 1));
h_inequality = flip(squeeze(mean(mean(mean(h_heatmap, 3), 2), 1)));   % flip: 50:50→90:10

tmp  = mean(mean(ai_heatmap, 5), 4);
ai_scenario   = squeeze(mean(tmp, 3));
ai_ratio      = squeeze(mean(tmp, 2));
tmp2 = mean(mean(ai_heatmap, 5), 3);
ai_cost       = squeeze(mean(tmp2, 2));
tmp3 = mean(mean(ai_heatmap, 4), 3);
ai_inequality = fliplr(squeeze(mean(tmp3, 2)));   % flip: 50:50→90:10

% ── Layout ────────────────────────────────────────────────────────────────
% N_UNITS = 10 (Human + 9 models), 5 rows × 2 groups side by side
% Manual axes positioning for tight margins and compact group gap
N_UNITS     = 1 + n_models;   % 10
N_COLS      = 4;               % panels per model unit
N_GRID_ROWS = 5;
figW        = 40;
figH        = N_GRID_ROWS * 7.5;

% Normalized layout parameters (figure coordinates 0–1)
ml      = 0.055;   % left margin
mr      = 0.012;   % right margin
mt      = 0.018;   % top margin
mb      = 0.042;   % bottom margin
gap_col = 0.010;   % gap between panels within a group
gap_row = 0.055;   % gap between rows (must fit row title annotation)

% gap_grp = panel_w/2  →  solve: panel_w=(1-ml-mr-gap_grp-6*gap_col)/8, gap_grp=panel_w/2
% → 17*gap_grp = 1-ml-mr-6*gap_col
gap_grp = (1 - ml - mr - 6*gap_col) / 17;   % exactly half a panel width
panel_w = gap_grp * 2;                        % = (1-ml-mr-gap_grp-6*gap_col)/8
panel_h = (1 - mt - mb - 4*gap_row) / 5;

fig = figure('Units','centimeters','Position',[1 1 figW figH], ...
             'Color','white','PaperPositionMode','auto');

ax_handles  = gobjects(N_UNITS, N_COLS);
unit_labels = cell(N_UNITS, 1);

fprintf('\n=== FigureS4: TPI directional sensitivity ===\n');

for u = 1:N_UNITS
    % ── Map unit → grid row and x-origin ──────────────────────────────────
    % Units 1-5 → left group; units 6-10 → right group (same rows 1-5)
    if u <= 5
        grid_row = u - 1;   % 0-based
        x_origin = ml;
    else
        grid_row = u - 6;   % 0-based
        x_origin = ml + 4*(panel_w + gap_col) - gap_col + gap_grp;
    end
    y_bottom_row = 1 - mt - (grid_row+1)*panel_h - grid_row*gap_row;

    % ── Data for this unit ────────────────────────────────────────────────
    if u == 1
        hm4d           = h_heatmap;
        sc             = h_scenario(:)';
        ineq           = h_inequality(:)';
        rt             = h_ratio(:)';
        co             = h_cost(:)';
        unit_labels{u} = 'Human';
    else
        m              = u - 1;
        hm4d           = squeeze(ai_heatmap(m,:,:,:,:));
        sc             = ai_scenario(m,:);
        ineq           = ai_inequality(m,:);
        rt             = ai_ratio(m,:);
        co             = ai_cost(m,:);
        unit_labels{u} = clean_name(ai_tpp_names{m});
        fprintf('  %2d. %s  r=%.3f\n', m, ai_tpp_names{m}, r_sorted(m));
    end

    % ── Stats ─────────────────────────────────────────────────────────────
    v1 = hm4d(1,:,:,:);  v1 = v1(:);
    v2 = hm4d(2,:,:,:);  v2 = v2(:);
    [~, psc] = ttest(v1, v2);
    r1 = hm4d(:,1,:,:);  r1 = r1(:);
    r2 = hm4d(:,2,:,:);  r2 = r2(:);
    [~, prt] = ttest(r1, r2);
    tmp_iq = squeeze(mean(mean(hm4d, 1), 2));
    p_ineq = zeros(1,4);  p_cost = zeros(1,4);
    for k = 1:4
        [~, p_ineq(k)] = ttest(tmp_iq(:,k), tmp_iq(:,k+1));
        [~, p_cost(k)] = ttest(tmp_iq(k,:)', tmp_iq(k+1,:)');
    end

    panel_vals   = {sc, ineq, rt, co};
    xlabels_cell = {SCENARIO_LABELS, INEQ_LABELS, RATIO_LABELS_TX, COST_LABELS};
    bar_colors   = {[C_PUNISH; C_HELP], C_INEQ, [C_RATIO_1; C_RATIO_2], repmat(C_COST,5,1)};

    for col = 1:N_COLS
        ax_left = x_origin + (col-1)*(panel_w + gap_col);
        ax = axes(fig, 'Position', [ax_left, y_bottom_row, panel_w, panel_h]); %#ok<LAXES>
        ax_handles(u, col) = ax;
        hold(ax, 'on');

        vals  = panel_vals{col};
        n_grp = length(vals);
        cols  = bar_colors{col};

        for g = 1:n_grp
            bar(ax, g, vals(g), 0.65, ...
                'FaceColor', cols(g,:), 'EdgeColor','none', 'FaceAlpha',0.85);
        end
        if col == 2 || col == 4
            plot(ax, 1:n_grp, vals, '-o', 'Color',[0.2 0.2 0.2], ...
                 'LineWidth',1.1,'MarkerSize',2.5,'MarkerFaceColor',[0.2 0.2 0.2]);
        end

        % Significance brackets
        if col == 1
            y_top = max(vals) + 0.06;
            draw_bracket(ax, 1, 2, y_top, pval2star(psc), FONT_SIZE_STAR, FONT_NAME);
        elseif col == 3
            y_top = max(vals) + 0.06;
            draw_bracket(ax, 1, 2, y_top, pval2star(prt), FONT_SIZE_STAR, FONT_NAME);
        elseif col == 2
            for k = 1:4
                y_k = max(vals(k), vals(k+1)) + 0.06;
                draw_bracket(ax, k, k+1, y_k, pval2star(p_ineq(k)), FONT_SIZE_STAR, FONT_NAME);
            end
        elseif col == 4
            for k = 1:4
                y_k = max(vals(k), vals(k+1)) + 0.06;
                draw_bracket(ax, k, k+1, y_k, pval2star(p_cost(k)), FONT_SIZE_STAR, FONT_NAME);
            end
        end

        % Axes
        ax.XTick = 1:n_grp;
        ax.XTickLabel = xlabels_cell{col};
        if col == 2
            ax.XTickLabelRotation = 35;
        else
            ax.XTickLabelRotation = 0;
        end
        ax.TickLabelInterpreter = 'tex';
        if contains(unit_labels{u}, 'DeepSeek-V4-Flash', 'IgnoreCase', true)
            ax.YLim    = [0, 0.26];
            ax.YTick   = [0 0.1 0.2];
            ax.YTickLabel = {'0','0.1','0.2'};
        else
            ax.YLim    = [0, 1.65];
            ax.YTick   = [0 0.25 0.50 0.75 1.00];
            ax.YTickLabel = {'0','','0.5','','1.0'};
        end
        ax.YGrid = 'on';
        ax.GridColor = [0.88 0.88 0.88];
        ax.GridLineStyle = '--';
        ax.GridAlpha = 0.6;
        ax.Box = 'off';
        ax.Layer = 'top';
        ax.FontSize = FONT_SIZE_AX;
        ax.FontName = FONT_NAME;
        ax.XColor = 'k'; ax.YColor = 'k';

        % Column subtitle (every subplot)
        title(ax, COL_TITLES{col}, 'FontSize',FONT_SIZE_TT, ...
              'FontName',FONT_NAME, 'FontWeight','normal', 'Color',[0.4 0.4 0.4]);

        if col == 1
            ylabel(ax, 'P(intervene)', 'FontSize',FONT_SIZE_LB, 'FontName',FONT_NAME);
            text(ax, 0.02, 0.97, panel_letter(u), ...
                 'Units','normalized', 'FontSize',FONT_SIZE_PANEL, ...
                 'FontName',FONT_NAME, 'FontWeight','normal', ...
                 'HorizontalAlignment','left', 'VerticalAlignment','top', 'Color','k');
        end
    end

    % ── Row title annotation (placed immediately after this unit is drawn) ──
    pos1 = get(ax_handles(u, 1),      'Position');
    pos4 = get(ax_handles(u, N_COLS), 'Position');
    ann_left   = pos1(1);
    ann_bottom = pos1(2) + pos1(4) + 0.003;
    ann_width  = (pos4(1) + pos4(3)) - pos1(1);
    ann_height = 0.028;
    annotation(fig, 'textbox', [ann_left, ann_bottom, ann_width, ann_height], ...
               'String', unit_labels{u}, ...
               'EdgeColor','none', 'BackgroundColor','none', ...
               'FontSize',FONT_SIZE_ROW, 'FontName',FONT_NAME, ...
               'FontWeight','bold', 'HorizontalAlignment','center', ...
               'VerticalAlignment','bottom', 'FitBoxToText',false, ...
               'Interpreter','none');
end

% ── Vertical divider line between left and right groups ───────────────────
x_divider = ml + 4*(panel_w + gap_col) - gap_col + gap_grp/2;
annotation(fig, 'line', [x_divider x_divider], [mb, 1-mt], ...
           'Color', [0.75 0.75 0.75], 'LineWidth', 0.8, 'LineStyle', '--');

% ── Save ──────────────────────────────────────────────────────────────────
OUT_DIR   = fileparts(mfilename('fullpath'));
PAPER_DIR = '/Users/wuxiaoyan/Desktop/TPP_culture_AI/manuscript/AI_Culture_altruism';

if ~exist(OUT_DIR,'dir'), mkdir(OUT_DIR); end

set(fig, 'PaperUnits','centimeters', 'PaperSize',[figW figH], ...
         'PaperPosition',[0 0 figW figH]);
print(fig, fullfile(OUT_DIR,'FigureS4_TPI_maineffects.pdf'), '-dpdf', '-r300');
print(fig, fullfile(OUT_DIR,'FigureS4_TPI_maineffects.png'), '-dpng', '-r300');
print(fig, fullfile(PAPER_DIR,'FigureS4.pdf'), '-dpdf', '-r300');
fprintf('Saved FigureS4.pdf to manuscript folder\n');

% =========================================================================
function s = pval2star(p)
    if p < 0.001,     s = '***';
    elseif p < 0.01,  s = '**';
    elseif p < 0.05,  s = '*';
    else,             s = 'ns';
    end
end

function draw_bracket(ax, x1, x2, y, stars, fsize, fname)
    tick_h = 0.025;
    plot(ax, [x1 x1 x2 x2], [y-tick_h y y y-tick_h], 'k-', 'LineWidth',0.7);
    text(ax, (x1+x2)/2, y+0.01, stars, ...
         'HorizontalAlignment','center', 'VerticalAlignment','bottom', ...
         'FontSize',fsize, 'FontName',fname, 'Color','k', 'FontWeight','bold');
end
