% =============================================================================
% Author : Xiaoyan Wu
% Date   : June 2026
% Description: Model fitting — Bayesian model comparison (PXP, protected exceedance probability)
% =============================================================================

clear; clc;

load('data_ai_all.mat')
load('results_ai.mat')

% ── Build model+Method combination labels ───────────────────────────────────
allModels  = {data_ai.model};
allMethods = {data_ai.Method};
allLabels  = strcat(allModels, '_', allMethods);   % e.g. 'GPT-4o_logprobs'
ModelList  = unique(allLabels);                     % unique model+method combinations

modelname = {'Baseline','SI','SI+SCI','SI+SCI+VCI', ...
             'SI+SCI+VCI+EC','SI+SCI+VCI+EC+RP', ...
             'SI+SCI+VCI+EC+RP+II','SI+SCI+VCI+EC+RP+II+lapse'};

AICC      = [];
deltaAICC = [];
PXP       = [];

for num_ai = 1:length(ModelList)

    % subjects for this model+method combination
    idx = strcmp(ModelList{num_ai}, allLabels);

    clear Results

    for m = 1:length(results_ai)
        Results(m).subid     = string({data_ai(idx).subid})';
        Results(m).loglik    = results_ai(m).loglik(idx)';
        Results(m).logp      = results_ai(m).logp(idx)';
        Results(m).p         = results_ai(m).p(idx)';
        Results(m).x         = results_ai(m).x(idx,:);   % N_subj × N_params, no transpose
        Results(m).aic       = results_ai(m).aic(idx)';
        Results(m).aicc      = results_ai(m).aicc(idx)';
        Results(m).ModelName = results_ai(m).ModelName;
        Results(m).AImodel   = ModelList{num_ai};
    end

    Results = ForModelComparison(Results);

    saveName = sprintf('Results_%s.mat', strrep(ModelList{num_ai}, '-', '_'));
    save(saveName, 'Results');

    AICC(num_ai,:)      = [Results.mean_AICc];
    deltaAICC(num_ai,:) = [Results.mean_detaAICc];
    PXP(num_ai,:)       = [Results.pxp];
end


% ── Plot parameters ─────────────────────────────────────────────────────────
n_ai     = length(ModelList);
n_cog    = length(modelname);
n_cols   = 3;
n_rows   = ceil(n_ai / n_cols);

bar_color_aicc  = [0.40 0.60 0.80];
bar_color_delta = [0.90 0.50 0.30];
bar_color_pxp   = [0.30 0.70 0.40];

short_names = strrep(ModelList, '-', ' ');   % display label: remove hyphens
short_names = strrep(short_names, '_', ' '); % replace underscores with spaces



% ── Three summary figures ───────────────────────────────────────────────────
datasets = {AICC, deltaAICC, PXP};
ylabels  = {'Mean AICc', '\DeltaAICc', 'PXP'};
fignames = {'fig_ModelComp_AICc', 'fig_ModelComp_deltaAICc', 'fig_ModelComp_PXP'};
colors   = {bar_color_aicc, bar_color_delta, bar_color_pxp};

for fig_i = 1:3
    DATA = datasets{fig_i};

    figure('Color','w','Position',[50 50 n_cols*320 n_rows*260]);

    for num_ai = 1:n_ai
        ax = subplot(n_rows, n_cols, num_ai);

        vals = DATA(num_ai, :);

        b = bar(ax, vals, 0.65, 'FaceColor', colors{fig_i}, 'EdgeColor', 'none');

        % x-axis: cognitive model abbreviations
        set(ax, 'XTick', 1:n_cog, ...
                'XTickLabel', modelname, ...
                'XTickLabelRotation', 35, ...
                'FontSize', 7.5, ...
                'TickDir', 'out');
        ylabel(ax, ylabels{fig_i}, 'FontSize', 8);
        title(ax, short_names{num_ai}, 'FontSize', 9, 'FontWeight', 'bold', 'Interpreter', 'none');
        grid(ax, 'on'); box(ax, 'off');
        ax.GridAlpha = 0.3;

        % reference line
        if fig_i == 2
            yline(ax, 0, 'k--', 'LineWidth', 0.8, 'Alpha', 0.5);
        elseif fig_i == 3
            ylim(ax, [0 1]);
            yline(ax, 1/n_cog, 'k--', 'LineWidth', 0.8, 'Alpha', 0.5);
        end

        % mark the best-fitting model
        if fig_i == 3
            [~, best_idx] = max(vals);   % PXP: highest value
        else
            [~, best_idx] = min(vals);   % AICc / ΔAICc: lowest value
        end
        mark_best(ax, best_idx, vals);
    end

    sgtitle(ylabels{fig_i}, 'FontSize', 13, 'FontWeight', 'bold');
    set(gcf, 'PaperPositionMode', 'auto');
    exportgraphics(gcf, [fignames{fig_i} '.pdf'], 'ContentType', 'vector');
    saveas(gcf, [fignames{fig_i} '.png']);
    fprintf('✓ %s\n', fignames{fig_i});
end



% ── Helper: highlight best model (min AICc / max PXP) ─────────────────────
function mark_best(ax, best_idx, y_vals)
    hold(ax, 'on');
    y_star = y_vals(best_idx);
    y_off  = (max(y_vals) - min(y_vals)) * 0.06;
    plot(ax, best_idx, y_star - y_off, ...
         'p', 'MarkerSize', 14, ...
         'MarkerFaceColor', [1 0.85 0], ...
         'MarkerEdgeColor', [0.8 0.6 0], ...
         'LineWidth', 1.2);
    % yellow circle marker
    plot(ax, best_idx, y_star, ...
         'o', 'MarkerSize', 18, ...
         'MarkerFaceColor', 'none', ...
         'MarkerEdgeColor', [1 0.85 0], ...
         'LineWidth', 2.0);
end