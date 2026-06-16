% =============================================================================
% Author : Xiaoyan Wu
% Date   : June 2026
% Description: Model M2 — adds self-interest (SI) motive
% =============================================================================

function lik = m2(x,data)

%--------------------------------------------------------------------------
% Function Name: m2
% Author: Xiaoyan Wu
% Date: February 12, 2024
%
% Usage:
%   Computes the log likelihood of the SI model (model 2)
%
% Inputs:
%   - x: The value of the free parameter of one subject.
%   - data: The dataset of one subject, containing 300 trials in total.
%
% Output:
%   - lik: Log likelihood of observing the actions.
%--------------------------------------------------------------------------

% setup parameter values
lambda = x(1);
% setup initial values
lik = 0;
%likelihood function
for t = 1:length(data.block)
    x3 = 50; % money of the third-party (the participant)
    cost = data.cost(t); % intervention cost
    a = data.action(t); % action = 1 if yes, action = 0 if no
    
    % utility function
    Uyes = (x3-cost); % utility of "yes"
    Uno = x3; % utility of  "No"
    
    p_yes = 1 / (1 + exp(lambda * (Uno - Uyes)));
    p_yes = max(min(p_yes, 1-1e-6), 1e-6);

    lik = lik + a * log(p_yes) + (1 - a) * log(1 - p_yes);
    % %choice probability
    % if action == 1
    %     lik = lik + log(1/(1+exp(lamda*(Uno-Uyes))));
    % elseif action ==0
    %     lik = lik + log(1/(1+exp(lamda*(Uyes-Uno))));
    % end
end