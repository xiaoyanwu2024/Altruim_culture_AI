% =============================================================================
% Author : Xiaoyan Wu
% Date   : June 2026
% Description: Model M3 — adds shared cost inequality (SCI)
% =============================================================================

function lik = m3(x,data)

%--------------------------------------------------------------------------
% Function Name: m3
% Author: Xiaoyan Wu
% Date: February 12, 2024
%
% Usage:
%   Computes the log likelihood of the SI + SCI model (model 3)
%
% Inputs:
%   - x: The value of the free parameter of one subject.
%   - data: The dataset of one subject, containing 300 trials in total.
%
% Output:
%   - lik: Log likelihood of observing the actions.
%--------------------------------------------------------------------------


% setup parameter values
envy = x(1);
guilt = x(2);
lambda =x(3);

% setup initial values
lik = 0;

% likelihood function
for t = 1:length(data.block)
    block = data.block(t); % block = 1 if in the punishment scenatio, block = 2 if in the help scenatio
    i = data.ratio(t); % impact ratio
    x1 = data.violator(t); % money of the violator
    x2 = data.victim(t); % money of the victim
    x3 = 50; % money of the third-party (the participant)
    cost = data.cost(t); % intervention cost
    a = data.action(t);
    
    % utility function
    x3s = x3-cost;
    if block == 1
        x1s = x1-cost*i; x2s = x2;
    elseif block == 2
        x1s = x1; x2s = x2+i*cost;
    end
    disad = max(x1s-x3s,0)+max(x2s-x3s,0); % self-centered: disadvantageous ineuquality
    ad = max(x3s-x1s,0)+max(x3s-x2s,0); % self-centered: advantageous inequality
    
    Uyes = x3s - envy*disad-guilt*ad; % utility of "yes"
    Uno = x3 - envy*(max(x1-x3,0)+max(x2-x3,0)) - guilt*(max(x3-x1,0) + max(x3-x2,0)); % utility of "no"

    p_yes = 1 / (1 + exp(lambda * (Uno - Uyes)));
    p_yes = max(min(p_yes, 1-1e-6), 1e-6);

    lik = lik + a * log(p_yes) + (1 - a) * log(1 - p_yes);
    
    % %choice probability
    % if action == 1
    %     lik = lik + log(1/(1+exp(lamda*(Uno-Uyes))));
    % elseif action == 0
    %     lik = lik + log(1/(1+exp(lamda*(Uyes-Uno))));
    % end
end