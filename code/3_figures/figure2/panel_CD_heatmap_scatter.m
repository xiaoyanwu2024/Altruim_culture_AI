% =============================================================================
% Author : Xiaoyan Wu
% Date   : June 2026
% Description: Figure 2 panels C+D — combined heatmap and scatter
% =============================================================================

% panel_CD_heatmap_scatter.m
% =========================================================================
% Combined Panel C+D:
%   LEFT  : Human heatmap (large, full height, 2×2 inner grid)
%   RIGHT : 3 AI columns, each = heatmap (top) + scatter vs Human (bottom)
%
% Colormap : white → #760031 (dark burgundy) for ALL heatmaps
% Scatter  : 100-condition points, colored by model (country colors from
%            task1_cultural_sensitivity reference, darkened for transparency)
%
% HOW TO MODIFY:
%   DARK_RED        : target color of heatmap (#760031)
%   MODEL_COLORS    : 3×3 RGB for AI scatter (one per top-3 model)
%   MARKER_ALPHA    : scatter dot transparency
%   HEATMAP_FRAC    : fraction of right column height used for heatmap
%   figW / figH     : figure size in cm
%   SHOW_VALUES     : true = print numbers inside heatmap cells
% =========================================================================

clear; clc;

DATA_DIR = fullfile(fileparts(mfilename('fullpath')), 'data');
load(fullfile(DATA_DIR, 'tpp_data.mat'));

% ── USER PARAMETERS ───────────────────────────────────────────────────────
FONT_NAME     = 'Helvetica';
FONT_SIZE_AX  = 7;
FONT_SIZE_LB  = 8;
FONT_SIZE_TT  = 9;
FONT_SIZE_STAT= 7;

% Heatmap colormap: white → #760031
DARK_RED  = [1.000, 0.000, 0.518];   % rgb(255,0,132)
n_colors  = 256;
CMAP = [linspace(1, DARK_RED(1), n_colors)', ...
        linspace(1, DARK_RED(2), n_colors)', ...
        linspace(1, DARK_RED(3), n_colors)'];
CMIN = 0.0;  CMAX = 1.0;

SHOW_VALUES = false;

% Model scatter colors — darker versions matching task1_cultural_sensitivity:
%   Mistral-Small-4   : dark purple  (ref: lavender)
%   GPT-4o-mini       : dark orange  (ref: peach)
%   GPT-4o (logprobs) : dark crimson (ref: coral-red)
MODEL_COLORS = [
    0.42  0.18  0.62;   % Mistral-Small-4      dark purple
    0.85  0.32  0.05;   % GPT-4o-mini          dark orange
    0.72  0.08  0.12;   % GPT-4o (logprobs)    dark crimson
];
MARKER_SIZE  = 22;
MARKER_ALPHA = 0.55;
CI_ALPHA     = 0.12;
LINE_WIDTH   = 1.8;

% Layout fractions
HEATMAP_FRAC = 0.58;   % fraction of right-column height for heatmap

figW = 24;
figH = 13;

BLOCK_LABELS  = {'Punishment', 'Helping'};
RATIO_LABELS  = {'\times1.5', '\times3'};
OFFER_XLABELS = {'90:10','80:20','70:30','60:40','50:50'};
OFFER_SHORT   = {'90','80','70','60','50'};
COST_YLABELS  = {'10','20','30','40','50'};

X_LIM = [0.0, 1.05];
Y_LIM = [-0.05, 1.10];

% ── Layout ────────────────────────────────────────────────────────────────
top3   = top3_tpp_idx;
clean  = @(s) strtrim(strrep(strrep(s,' (logprobs)',''),'_logprobs',''));

margin_l  = 0.07;
margin_r  = 0.06;
margin_b  = 0.09;
margin_t  = 0.10;
cb_w      = 0.016;
gap_hcol  = 0.04;   % gap: human → AI section
gap_ai    = 0.025;  % gap between AI columns
gap_hs    = 0.045;  % gap: heatmap → scatter within column

panel_h_total = 1 - margin_t - margin_b;   % 0.81
human_w  = 0.24;
ai_total = 1 - margin_l - human_w - gap_hcol - margin_r - cb_w - gap_hcol;
ai_col_w = (ai_total - 2*gap_ai) / 3;

hm_h  = panel_h_total * HEATMAP_FRAC;
sc_h  = panel_h_total * (1 - HEATMAP_FRAC) - gap_hs;
hm_b  = margin_b + sc_h + gap_hs;
sc_b  = margin_b;

human_left = margin_l;
ai_left    = @(k) margin_l + human_w + gap_hcol + (k-1)*(ai_col_w+gap_ai);

fig = figure('Units','centimeters','Position',[1 1 figW figH], ...
             'Color','white','PaperPositionMode','auto');

% ── 1. Human heatmap (left, full height) ──────────────────────────────────
h_pos = [human_left, margin_b, human_w, panel_h_total];
draw_heatmap(fig, h_pos, h_heatmap, 'Human data', ...
             FONT_NAME, FONT_SIZE_AX+1, FONT_SIZE_LB, FONT_SIZE_TT+1, ...
             OFFER_XLABELS, COST_YLABELS, BLOCK_LABELS, RATIO_LABELS, ...
             SHOW_VALUES, CMAP, CMIN, CMAX, true);

% ── 2. AI columns (heatmap top + scatter bottom) ──────────────────────────
for k = 1:3
    idx  = top3(k);
    col  = MODEL_COLORS(k,:);
    name = clean(ai_tpp_names{idx});
    xl   = ai_left(k);

    % Heatmap
    hm_pos = [xl, hm_b, ai_col_w, hm_h];
    draw_heatmap(fig, hm_pos, squeeze(ai_heatmap(idx,:,:,:,:)), name, ...
                 FONT_NAME, FONT_SIZE_AX, FONT_SIZE_LB, FONT_SIZE_TT, ...
                 OFFER_SHORT, COST_YLABELS, BLOCK_LABELS, RATIO_LABELS, ...
                 SHOW_VALUES, CMAP, CMIN, CMAX, false);

    % Scatter
    sc_pos = [xl, sc_b, ai_col_w, sc_h];
    h_v    = h_vec(:);
    ai_v   = ai_tpp_mean(idx,:)';
    valid  = ~isnan(h_v) & ~isnan(ai_v);
    hx = h_v(valid);  ay = ai_v(valid);  n_pts = sum(valid);

    X    = [ones(n_pts,1), hx];
    b    = X \ ay;
    r_val = corr(hx, ay);
    t_val = r_val * sqrt((n_pts-2)/(1-r_val^2));
    p_val = 2*(1-tcdf(abs(t_val), n_pts-2));
    x_fit = linspace(0, 1, 200)';
    y_fit = b(1) + b(2)*x_fit;
    hm    = mean(hx);  ssx = sum((hx-hm).^2);
    se    = sqrt(mean((ay-(b(1)+b(2)*hx)).^2) .* (1/n_pts + (x_fit-hm).^2/ssx));
    t95   = tinv(0.975, n_pts-2);

    ax_s = axes(fig,'Position',sc_pos,'FontName',FONT_NAME,'FontSize',FONT_SIZE_AX,...
                'Box','off','LineWidth',0.7,'XColor','k','YColor','k');
    hold(ax_s,'on');
    plot(ax_s,[0 1],[0 1],'--','Color',[0.75 0.75 0.75],'LineWidth',0.8);
    fill(ax_s,[x_fit;flipud(x_fit)],[y_fit+t95*se;flipud(y_fit-t95*se)],col,...
         'FaceAlpha',CI_ALPHA,'EdgeColor','none');
    scatter(ax_s,hx,ay,MARKER_SIZE,col,'filled',...
            'MarkerFaceAlpha',MARKER_ALPHA,'MarkerEdgeColor','none');
    plot(ax_s,x_fit,y_fit,'-','Color',col,'LineWidth',LINE_WIDTH);

    if p_val < 0.001, ps='p<.001'; elseif p_val<0.01, ps=sprintf('p=%.3f',p_val);
    else, ps=sprintf('p=%.3f',p_val); end
    text(ax_s,0.04,Y_LIM(2)-0.02,sprintf('r=%.3f\n%s',r_val,ps),...
         'FontSize',FONT_SIZE_STAT,'FontName',FONT_NAME,'VerticalAlignment','top',...
         'Color',col*0.75,'BackgroundColor','white');

    xlim(ax_s,X_LIM);  ylim(ax_s,Y_LIM);
    ax_s.XTick = 0:0.5:1;  ax_s.YTick = 0:0.5:1;
    ax_s.YGrid='on'; ax_s.GridColor=[0.88 0.88 0.88];
    ax_s.GridLineStyle='--'; ax_s.GridAlpha=0.6;
    if k==1
        ylabel(ax_s,'AI  P(intervene)','FontSize',FONT_SIZE_LB,'FontName',FONT_NAME);
    end
    xlabel(ax_s,'Human  P(intervene)','FontSize',FONT_SIZE_LB,'FontName',FONT_NAME);
end

% ── 3. Colorbar ───────────────────────────────────────────────────────────
cb_left = ai_left(3) + ai_col_w + gap_ai*0.6;
cb_ax   = axes(fig,'Position',[cb_left, margin_b+0.08, cb_w, panel_h_total-0.16]);
imagesc(cb_ax,(1:n_colors)');
colormap(cb_ax,CMAP);
set(cb_ax,'YDir','normal','XTick',[],...
    'YTick',linspace(1,n_colors,6),'YTickLabel',{'0','0.2','0.4','0.6','0.8','1.0'},...
    'FontSize',FONT_SIZE_AX,'FontName',FONT_NAME,'YColor','k');
ylabel(cb_ax,'Intervention rate','FontSize',FONT_SIZE_AX+0.5,'FontName',FONT_NAME,'Color','k');

% ── 4. Offer-direction arrow (below Human heatmap) ─────────────────────────
arr_y = max(margin_b - 0.045, 0.015);
annotation(fig,'arrow',[human_left+0.01, human_left+human_w*0.88],[arr_y,arr_y],...
    'Color','k','LineWidth',1.0,'HeadStyle','vback2','HeadLength',5,'HeadWidth',4);
annotation(fig,'textbox',[human_left, max(arr_y-0.036,0.001), human_w*1.1, 0.034],...
    'String','Offer:  90:10 (unfair)  \rightarrow  50:50 (fair)',...
    'EdgeColor','none','FontSize',6.5,'FontName',FONT_NAME,...
    'HorizontalAlignment','left','FontAngle','italic','Color',[0.35 0.35 0.35],...
    'VerticalAlignment','bottom');


% ── Save ──────────────────────────────────────────────────────────────────
OUT_DIR = fullfile(fileparts(mfilename('fullpath')), 'output');
if ~exist(OUT_DIR,'dir'), mkdir(OUT_DIR); end
print(fig,fullfile(OUT_DIR,'panel_CD_heatmap_scatter.pdf'),'-dpdf','-r300','-bestfit');
print(fig,fullfile(OUT_DIR,'panel_CD_heatmap_scatter.png'),'-dpng','-r300');
fprintf('Saved panel_CD_heatmap_scatter.pdf/.png\n');

% =========================================================================
% LOCAL FUNCTION — must be at end of file
% =========================================================================
function draw_heatmap(fig, pos, hm4d, title_str, ...
                      FONT_NAME, FONT_SIZE_AX, FONT_SIZE_LB, FONT_SIZE_TT, ...
                      OFFER_XLABELS, COST_YLABELS, BLOCK_LABELS, RATIO_LABELS, ...
                      SHOW_VALUES, CMAP, CMIN, CMAX, is_human)

    inner_gap_h = 0.010;
    inner_gap_v = 0.038;
    iw = (pos(3) - inner_gap_h) / 2;
    ih = (pos(4) - inner_gap_v) / 2;

    for bi = 1:2
        for ri = 1:2
            left   = pos(1) + (ri-1)*(iw + inner_gap_h);
            bottom = pos(2) + (2-bi)*(ih + inner_gap_v);
            ax = axes(fig,'Position',[left, bottom, iw, ih]);

            data = squeeze(hm4d(bi, ri, :, :));
            imagesc(ax, data, [CMIN CMAX]);
            colormap(ax, CMAP);
            ax.TickLength = [0 0];
            ax.XTick = 1:5;  ax.YTick = 1:5;
            ax.XColor = 'k'; ax.YColor = 'k';
            ax.Box = 'off';
            ax.FontSize = FONT_SIZE_AX;
            ax.FontName = FONT_NAME;

            if bi == 2
                ax.XTickLabel = OFFER_XLABELS;
                ax.XTickLabelRotation = is_human * 0;
            else
                ax.XTickLabel = {};
            end

            if ri == 1
                ax.YTickLabel = COST_YLABELS;
            else
                ax.YTickLabel = {};
            end

            if bi == 1
                title(ax, RATIO_LABELS{ri}, 'FontSize', max(FONT_SIZE_AX-0.5,6), ...
                      'FontName',FONT_NAME,'Color','k',...
                      'FontAngle','italic','FontWeight','normal');
            end

            if ri == 1
                fs_blk = FONT_SIZE_AX + 0.5*is_human;
                ylabel(ax, BLOCK_LABELS{bi}, 'FontSize', fs_blk, ...
                       'FontName',FONT_NAME,'Color','k','FontWeight','bold');
            end

            if SHOW_VALUES
                for r=1:5
                    for c=1:5
                        v=data(r,c);
                        if ~isnan(v)
                            text(ax,c,r,sprintf('%.2f',v),'FontSize',4,...
                                 'HorizontalAlignment','center','Color','k');
                        end
                    end
                end
            end
        end
    end

    annotation(fig,'textbox',...
               [pos(1), pos(2)+pos(4)+0.003, pos(3), 0.038],...
               'String',title_str,'Color','k','EdgeColor','none',...
               'FontSize',FONT_SIZE_TT,'FontName',FONT_NAME,...
               'FontWeight','bold','HorizontalAlignment','center',...
               'VerticalAlignment','bottom','FitBoxToText',false,'Interpreter','none');
end
