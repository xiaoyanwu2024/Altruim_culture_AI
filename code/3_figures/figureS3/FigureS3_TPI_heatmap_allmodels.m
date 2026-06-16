% =============================================================================
% Author : Xiaoyan Wu
% Date   : June 2026
% Description: Supplementary Figure S3 — TPI heatmaps for all 9 AI models
% =============================================================================

% FigureS3_TPI_heatmap_allmodels.m
% =========================================================================
% Supplementary Fig S3: TPI condition heatmaps — Human + 9 prob AI models
% Layout: 4 panels per row, each panel = 2×2 inner heatmap (Block × Ratio)
% Format: Arial, 8pt axis, 9pt panel letters (top-left, no bold)
% =========================================================================

clear; clc;

DATA_DIR = fullfile(fileparts(mfilename('fullpath')), 'data');
load(fullfile(DATA_DIR, 'tpp_data.mat'));

% ── FORMAT PARAMETERS ─────────────────────────────────────────────────────
FONT_NAME       = 'Arial';
FONT_SIZE_AX    = 6.5;
FONT_SIZE_TT    = 8;
FONT_SIZE_PANEL = 9;
CMIN = 0.0;  CMAX = 1.0;
SHOW_VALUES = false;
N_COLS      = 4;

% Colormap: white → #C13383
n_colors = 256;
dark_red = [0.757, 0.200, 0.514];
CMAP = [linspace(1, dark_red(1), n_colors)', ...
        linspace(1, dark_red(2), n_colors)', ...
        linspace(1, dark_red(3), n_colors)'];

BLOCK_LABELS  = {'Punishment', 'Helping'};
RATIO_LABELS  = {'\times1.5', '\times3'};
OFFER_XLABELS = {'90:10','80:20','70:30','60:40','50:50'};
COST_YLABELS  = {'10','20','30','40','50'};

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
ai_tpp_names = ai_tpp_names(keep);
ai_heatmap   = ai_heatmap(keep, :, :, :, :);

% ── Sort by r (best first) ────────────────────────────────────────────────
ai_tpp_r_filt = ai_tpp_r(keep);
[~, r_idx]    = sort(ai_tpp_r_filt, 'descend');
ai_tpp_names  = ai_tpp_names(r_idx);
ai_heatmap    = ai_heatmap(r_idx, :, :, :, :);

% ── Combine Human + 9 AI ─────────────────────────────────────────────────
clean_names  = cellfun(clean_name, ai_tpp_names(:)', 'UniformOutput', false);
all_names    = [{'Human'}, clean_names];
all_heatmap  = cat(1, reshape(h_heatmap,[1,2,2,5,5]), ai_heatmap);
n_total      = size(all_heatmap, 1);

% ── Figure layout ─────────────────────────────────────────────────────────
N_ROWS = ceil(n_total / N_COLS);
figW   = 26;
figH   = N_ROWS * 7.5;

fig = figure('Units','centimeters','Position',[1 1 figW figH], ...
             'Color','white','PaperPositionMode','auto');

margin_l   = 0.08;
margin_r   = 0.06;
margin_t   = 0.03;
margin_b   = 0.04;
gap_col    = 0.035;
gap_row    = 0.055;
title_frac = 0.10;

panel_w = (1 - margin_l - margin_r - (N_COLS-1)*gap_col) / N_COLS;
panel_h = (1 - margin_t - margin_b - (N_ROWS-1)*gap_row) / N_ROWS;

inner_gap_h = 0.008;
inner_gap_v = 0.028;

for p = 1:n_total
    row = ceil(p / N_COLS) - 1;   % 0-based
    col = mod(p-1, N_COLS);

    px = margin_l + col * (panel_w + gap_col);
    py = 1 - margin_t - (row+1)*panel_h - row*gap_row;

    content_h = panel_h * (1 - title_frac);
    content_y = py;

    iw = (panel_w - inner_gap_h)  / 2;
    ih = (content_h - inner_gap_v) / 2;

    % ── 2×2 inner heatmaps ────────────────────────────────────────────────
    for bi = 1:2
        for ri = 1:2
            left   = px + (ri-1)*(iw + inner_gap_h);
            bottom = content_y + (2-bi)*(ih + inner_gap_v);

            ax = axes(fig, 'Position',[left, bottom, iw, ih]); %#ok<LAXES>
            data = squeeze(all_heatmap(p, bi, ri, :, :));
            imagesc(ax, data, [CMIN CMAX]);
            colormap(ax, CMAP);
            ax.TickLength = [0 0];
            ax.Box = 'off';
            ax.XColor = 'k';  ax.YColor = 'k';
            ax.FontSize = FONT_SIZE_AX;
            ax.FontName = FONT_NAME;

            if bi == 2
                ax.XTick = 1:5;
                ax.XTickLabel = OFFER_XLABELS;
                ax.XTickLabelRotation = 30;
            else
                ax.XTick = [];
            end

            if ri == 1
                ax.YTick = 1:5;
                ax.YTickLabel = COST_YLABELS;
            else
                ax.YTick = [];
            end

            if bi == 1
                title(ax, RATIO_LABELS{ri}, 'FontSize',FONT_SIZE_AX-0.5, ...
                      'FontName',FONT_NAME, 'Color','k', ...
                      'FontAngle','italic', 'FontWeight','normal', ...
                      'Interpreter','tex');
            end
            if ri == 1
                ylabel(ax, BLOCK_LABELS{bi}, 'FontSize',FONT_SIZE_AX, ...
                       'FontName',FONT_NAME, 'Color','k', 'FontWeight','bold');
            end

            if SHOW_VALUES
                for r2=1:5
                    for c2=1:5
                        v=data(r2,c2);
                        if ~isnan(v)
                            text(ax,c2,r2,sprintf('%.2f',v),'FontSize',4,...
                                 'HorizontalAlignment','center','Color','k');
                        end
                    end
                end
            end
        end
    end

    % ── Panel title ───────────────────────────────────────────────────────
    % p==1: Human; p==2,3,4: top-3 AI (already sorted by r descending)
    if p == 1
        ttl = all_names{p};  fw = 'bold';
    elseif p <= 4   % top-3 AI models
        ttl = ['[★]  ' all_names{p}];  fw = 'bold';
    else
        ttl = all_names{p};  fw = 'normal';
    end
    title_y = content_y + content_h + 0.004;
    annotation(fig, 'textbox', [px, title_y, panel_w, panel_h*title_frac], ...
               'String', ttl, 'Color','k', 'EdgeColor','none', ...
               'FontSize',FONT_SIZE_TT, 'FontName',FONT_NAME, ...
               'FontWeight',fw, 'HorizontalAlignment','center', ...
               'VerticalAlignment','bottom', 'FitBoxToText',false, ...
               'Interpreter','none');

    % ── Panel letter (top-left of panel area) ─────────────────────────────
    annotation(fig, 'textbox', [px, py+panel_h-0.04, 0.04, 0.04], ...
               'String', panel_letter(p), 'Color','k', 'EdgeColor','none', ...
               'FontSize',FONT_SIZE_PANEL, 'FontName',FONT_NAME, ...
               'FontWeight','normal', 'HorizontalAlignment','left', ...
               'VerticalAlignment','top', 'FitBoxToText',false);
end

% ── Colorbar in last empty slot ───────────────────────────────────────────
last_row = ceil(n_total / N_COLS) - 1;
last_col = mod(n_total, N_COLS);
if last_col == 0
    cb_px = 1 - margin_r + 0.005;
    cb_py = margin_b + 0.10;
    cb_h  = 0.25;
    cb_w  = 0.014;
else
    cb_px = margin_l + last_col * (panel_w + gap_col);
    cb_py = 1 - margin_t - (last_row+1)*panel_h - last_row*gap_row;
    cb_w  = 0.018;
    cb_h  = panel_h * 0.65;
    cb_px = cb_px + panel_w*0.15;
    cb_py = cb_py + panel_h*0.18;
end

cb_ax = axes(fig, 'Position',[cb_px, cb_py, cb_w, cb_h]); %#ok<LAXES>
imagesc(cb_ax, (1:n_colors)');
colormap(cb_ax, CMAP);
set(cb_ax, 'YDir','normal', 'XTick',[], ...
    'YTick',linspace(1,n_colors,6), ...
    'YTickLabel',{'0','0.2','0.4','0.6','0.8','1.0'}, ...
    'FontSize',FONT_SIZE_AX, 'FontName',FONT_NAME, 'YColor','k');
ylabel(cb_ax, 'P(intervene)', 'FontSize',FONT_SIZE_AX+0.5, ...
       'FontName',FONT_NAME, 'Color','k');

% ── Save ──────────────────────────────────────────────────────────────────
OUT_DIR   = fullfile(fileparts(mfilename('fullpath')), 'output');
PAPER_DIR = '/Users/wuxiaoyan/Desktop/TPP_culture_AI/manuscript/AI_Culture_altruism';
if ~exist(OUT_DIR,'dir'), mkdir(OUT_DIR); end

set(fig, 'PaperUnits','centimeters', 'PaperSize',[figW figH], ...
         'PaperPosition',[0 0 figW figH]);
print(fig, fullfile(OUT_DIR,'FigureS3_TPI_heatmap_allmodels.pdf'), '-dpdf', '-r300');
print(fig, fullfile(OUT_DIR,'FigureS3_TPI_heatmap_allmodels.png'), '-dpng', '-r300');
print(fig, fullfile(PAPER_DIR,'FigureS3.pdf'), '-dpdf', '-r300');
fprintf('Saved FigureS3_TPI_heatmap_allmodels.pdf/.png\nSaved FigureS3.pdf to manuscript folder\n');
