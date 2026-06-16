% =============================================================================
% Author : Xiaoyan Wu
% Date   : June 2026
% Description: Model M8 — full motive-cocktail (all 8 motives: SI, SCI, VCI, EC, RP, II, η_no, η_yes)
% =============================================================================

function lik = m8(x,data)

%--------------------------------------------------------------------------
% Function Name: m8
% Author: Xiaoyan Wu
% Date: February 12, 2024
%
% Usage:
%   Computes the log likelihood of the SI + SCI + VCI + EC + RP + II + lapse model (model 8)
%
% Inputs:
%   - x: The value of the free parameter of one subject.
%   - data: The dataset of one subject, containing 300 trials in total.
%
% Output:
%   - lik: Log likelihood of observing the actions.
%--------------------------------------------------------------------------

%setup parameter values
gama = x(1);
envy = x(2);
guilt = x(3);
lambda =x(4);
oumiga = x(5);
kapa = x(6);
etak = x(7);
etaa = x(8);
minp = x(9);
maxp = x(10);

% setup initial values
lik = 0;

% likelihood function
for t = 1:length(data.block)
    block = data.block(t);
    i = data.ratio(t);
    x1 = data.violator(t);
    x2 = data.victim(t);
    x3 = 50;
    cost = data.cost(t);
    a = data.action(t);
    
    % utility function
    x3s = x3-cost;
    if block == 1
        x1s = x1-cost*i; x2s = x2;
    elseif block == 2
        x1s = x1; x2s = x2+i*cost;
    end
    disad = max(x1s-x3s,0)+max(x2s-x3s,0);
    ad = max(x3s-x1s,0)+max(x3s-x2s,0);
    inqua = max(x1s-x2s,0);
    EP = (x1s+x2s);
    RP = max(x2s-x1s,0);
    IIk = 2/(1+(exp(etak*(cost/50))));
    IIa = 2/(1+(exp(etaa*(cost/50))));
    Uyes = x3s - envy*disad - guilt*ad - gama*inqua*IIa + oumiga*EP + kapa*RP; % utility of "yes"
    Uno = x3 - envy*(max(x1-x3,0)+max(x2-x3,0)) - guilt*(max(x3-x1,0)+max(x3-x2,0)) - gama*(max(x1-x2,0)*IIk)  + oumiga*(x1+x2) + kapa*max(x2-x1,0); % utility of "no"
    p_yes =  1/(1+exp(lambda*(Uno-Uyes)));
    p_yes = minp+(1-maxp-minp)*p_yes;
    p_yes = max(min(p_yes, 1-1e-6), 1e-6);

    lik = lik + a * log(p_yes) + (1 - a) * log(1 - p_yes);

    % %choice probability
    % if action == 1
    %     lik = lik + log(pact);
    % elseif action == 0
    %     lik = lik + log(1-pact);
    % end
end