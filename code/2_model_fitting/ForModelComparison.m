% =============================================================================
% Author : Xiaoyan Wu
% Date   : June 2026
% Description: Model fitting — computes AICc/BIC for model comparison
% =============================================================================

function results = ForModelComparison(results)

%--------------------------------------------------------------------------
% Function Name: ForModelComparison
% Author: Xiaoyan Wu (xiaoyan.psych@gmail.com)
% Created: February 12, 2024
% Description:
%   Calculates Akaike Information Criterion (AIC),corrected AIC (AICc), and protected 
%   exceedance probability (PEP) for all models across all subjects. This function is 
%   designed to support comprehensive model comparison.
%
% Input:
%   results - A structure containing fitting results for all models applied to
%             the subjects. Each row corresponds to one model, and includes:
%       .K - Number of free parameters in the model.
%       .param - The structure of free parameters in the model.
%       .subid - Subject IDs
%       .loglik - Log likelihood for each subject, double check of the log likelihood calculated by a different method.
%       .logp -  Log likelihood for each subject.
%       .p - Exponential of the log likelihood (exp(loglik)).
%       .x - Values of the free parameters.
%       .aic - Akaike Information Criterion (AIC) values.
%       .aicc - Corrected AIC for sample size.
%
% Output:
%   The function updates the input 'results' structure with the following fields:
%       .sum_AICc - Sum of the AICc across all subjects.
%       .mean_AICc - Mean of the AICc across all subjects.
%       .mean_detaAICc - Mean of the ΔAICc across all subjects.
%       .pxp - PEP based on AIC, providing the probability of each model being the best model among the set
%
% Note:
%   ΔAICc are calculated for each model by subtracting the lowest
%   AICc observed for a given participant from the AICc of the other models. 
%   This provides a relative measure of model fit.
%--------------------------------------------------------------------------

for s = 1:length(results(1).subid) % loop across subjects
    for i = 1:length(results) % loop across models
        subaicc(i,1) = (results(i).aicc(s));
    end
    subaicc = subaicc-min(subaicc);
    
    for i = 1:length(subaicc)
        results(i).detaAICc(s,1) = subaicc(i);
    end
end

% calculate the mean and sum value for AICc and detaAICc
for i = 1:length(results)  % loop across models  
    results(i).sum_AICc = sum(results(i).aicc);
    results(i).mean_AICc = mean(results(i).aicc);
    
    results(i).sum_detaAICc = sum(results(i).detaAICc);
    results(i).mean_detaAICc = mean(results(i).detaAICc);
end

% PEP
for m = 1:length(results)
    for s = 1:length(results(1).logp)
        sublogp = results(m).logp(s);
        subAICc = results(m).aicc(s);
        subAICc = -(subAICc/2); % log evidence
        lmeaic(s,m) = subAICc;
    end
end

[~,~,~,pxp,~] = spm_BMS(lmeaic);
for m = 1:length(results)
    results(m).pxp = pxp(m);
end
