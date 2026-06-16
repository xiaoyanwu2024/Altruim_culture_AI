% =============================================================================
% Author : Xiaoyan Wu
% Date   : June 2026
% Description: Model fitting — individual-level parameter estimation for
%              motive-cocktail models M1–M8
% =============================================================================

function [loglik,logp,p,x,aic,aicc] = IndividualFitting(likfun,param,subdata,NumMoltiiStart,NumMaxTry,K)
%--------------------------------------------------------------------------
% Function Name: IndividualFitting
% Author: Xiaoyan Wu (xiaoyan.psych@gmail.com)
% Date: February 13, 2024
%
% Usage: This function is used for model fitting for each subject, primarily
%        for parallel computing reasons.
%
% Input:
%   likfun - The loglikelihood function for the model.
%   param - The parameter structure for the model.
%   subdata - Data for one subject.
%   NumMoltiiStart - Number of multiple start points for global searching.
%   NumMaxTry - Maximum number of attempts for model fitting to solve the
%               failure of initial starting values.
%   K - Number of the free parameters for the model.
%   multiple - Number of the global searches.
%
% Output:
%   loglik - The loglikelihood.
%   logp - The loglikelihood transformed into the joint probability (p = exp(logp)).
%   p - The joint probability.
%   x - The estimated parameter values.
%   aic - The Akaike information criterion score.
%   aicc - The corrected version of small sample of AIC.
%
%--------------------------------------------------------------------------
options = optimset('Display','off');
lb = [param.lb];
ub = [param.ub];
f = @(x) -likfun(x,subdata);
gs = GlobalSearch;
numTry = 1;
while numTry < NumMaxTry
    for i = 1:length(ub)
        if ub(i) == inf
            x0(i) = lb(i)+10*rand();
        else
            x0(i) = lb(i)+ub(i)*rand();
        end
    end
    problem = createOptimProblem('fmincon','objective',...
        f,'x0',x0,'lb',lb,'ub',ub','options',options);
    gs = GlobalSearch(gs,'StartPointsToRun','bounds','Display','off','NumTrialPoints',NumMoltiiStart);
    try
        [x,nlogp] = run(gs,problem);
        break;
    catch
        numTry = numTry+1;
    end
end

n = length(subdata);
loglik = likfun(x,subdata);
logp = -nlogp;
p = exp(logp);
aic = -2*logp+ 2*K;
aicc = (-2*logp+ 2*K)+((2*K*(K+1))/(n-K-1));
