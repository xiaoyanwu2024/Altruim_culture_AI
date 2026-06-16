% =============================================================================
% Author : Xiaoyan Wu
% Date   : June 2026
% Description: Model M7 — adds inequality aversion (II)
% =============================================================================

function lik = m7(x,data)

%--------------------------------------------------------------------------
% Function Name: m7
% Author: Xiaoyan Wu
% Date: February 12, 2024
%
% Usage:
%   Computes the log likelihood of the SI + SCI + VCI + EC + RP + II model (model 7)
%
% Inputs:
%   - x: The value of the free parameter of one subject.
%   - data: The dataset of one subject, containing 300 trials in total.
%
% Output:
%   - lik: Log likelihood of observing the actions.
%--------------------------------------------------------------------------

% setup parameter values
gama = x(1);
envy = x(2);
guilt = x(3);
lambda =x(4);
oumiga = x(5);
kapa = x(6);
etak = x(7);
etaa = x(8);

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
    ad = max(x3s-x1s,0)+max(x3s-x2s,0); % self-centered: advantageous ineuquality
    inqua = max(x1s-x2s,0); % victim-centered inequality aversion
    EP = (x1s+x2s); % efficiency concern
    RP = max(x2s-x1s,0); % reversual preference
    IIk = 2/(1+(exp(etak*(cost/50)))); % inaction inequality attention
    IIa = 2/(1+(exp(etaa*(cost/50)))); % action inequality attention
    
    Uyes = x3s - envy*disad - guilt*ad - gama*inqua*IIa + oumiga*EP + kapa*RP; % utility of "yes"
    Uno = x3 - envy*(max(x1-x3,0)+max(x2-x3,0)) - guilt*(max(x3-x1,0)+max(x3-x2,0)) - gama*(max(x1-x2,0)*IIk)  + oumiga*(x1+x2) + kapa*max(x2-x1,0); % utility of "no"
    
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