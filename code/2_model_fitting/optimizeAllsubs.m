% =============================================================================
% Author : Xiaoyan Wu
% Date   : June 2026
% Description: Model fitting — batch optimisation across all subjects
% =============================================================================

function results = optimizeAllsubs(likfun,param,data)

%--------------------------------------------------------------------------
% Function Name: optimizeAllsubs
% Author: Xiaoyan Wu
% Date: February 12, 2024
% Usage: This function performs model fitting
%
% Input:
%   likfun - the loglikelihood function
%   param - the free parameter structure
%   data - the dataset of all subjects, with 300 trials for each subject
%   NumMoltiiStart - the number of multiple start for global searching
%   NumMaxTry - the number of attempts for the starting point of x0.

%
% Output:
% results - A structure containing fitting results for all models, which includes:
%       .K - Number of free parameters in the model.
%       .param - Parameter structure of the model.
%       .subid - Subject IDs of all subjects
%       .loglik - Log likelihood of all subjects,double check of the log likelihood calculated by a different method
%       .logp - Log likelihood of all subjects
%       .p - Exponential of the log likelihood (exp(loglik)).
%       .x - Values of the free parameters of all subjects.
%       .aic - Akaike Information Criterion (AIC) values of all subjects.
%       .aicc - Corrected AIC for sample size of all subjects.
%--------------------------------------------------------------------------

K = length(param);
results.K = K;
results.param = param;
results.likfun = likfun;
options = optimset('Display','off');

lb = [param.lb];
ub = [param.ub];

NumMaxTry = 500;
NumMoltiiStart = 1000;

gs = GlobalSearch;


for s = 1:length(data)
    
    disp(['Subject ',num2str(s)]);
    f = @(x) -likfun(x,data(s));
    numTry = 1;
    
    while numTry < NumMaxTry
        for i = 1:length(ub)
            x0(i) = lb(i)+ub(i)*rand();
        end

        problem = createOptimProblem('fmincon','objective', f,'x0',x0,'lb',lb,'ub',ub','options',options);
        gs = GlobalSearch(gs,'StartPointsToRun','bounds','Display','off','NumTrialPoints',NumMoltiiStart);
        try
            [x, nlogp] = run(gs,problem);
            % problem = createOptimProblem('fmincon','objective',...
            %     f,'x0',x0,'lb',lb,'ub',ub','options',options);
            % ms = MultiStart('PlotFcns',@gsplotbestf);
            % [x,nlogp] = run(ms,problem,500);
            break
        catch
            numTry = numTry+1;
        end
    end
    
    n = length(data(s).action);
    results.subid(s,1) = data(s).subid(1);
    logp = -nlogp;
    results.loglik(s) = likfun(x,data(s));
    results.logp(s) = logp;
    results.p(s) = exp(logp);
    results.x(s,:) = x;
    results.aic(s,1) = -2*logp+ 2*K;
    results.aicc(s,1) = (-2*logp+ 2*K)+((2*K*(K+1))/(n-K-1));
end
