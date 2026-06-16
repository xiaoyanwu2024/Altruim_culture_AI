% =============================================================================
% Author : Xiaoyan Wu
% Date   : June 2026
% Description: Model fitting — main script to fit motive-cocktail models M1–M8 to AI TPI data
% =============================================================================

%clear space
clc;clear;

%load data
path = pwd;
load('data_ai_all.mat');
load('results_human_aligned.mat');

% model fitting of model 1 to model 8
core = 9;
p = parpool(core);
parfor m = 1:length(results_aligned)
    model = results_aligned(m).likfun;
    param = results_aligned(m).param;
    results_ai(m) = optimizeAllsubs(model, param, data_ai);
    disp(['Model:',num2str(m)]);
end
delete(p);

modelname = {'Baseline','SI','SI+SCI','SI+SCI+VCI','SI+SIC+VCI+EC','SI+SIC+VCI+EC+RP','SI+SIC+VCI+EC+RP+II','SI+SIC+VCI+EC+RP+II+lapse'};
for m = 1:length(results_ai)
    results_ai(m).ModelName = modelname(m);
end
results_ai = ForModelComparison(results_ai);
save('results_ai.mat','results_ai');