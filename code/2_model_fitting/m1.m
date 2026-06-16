% =============================================================================
% Author : Xiaoyan Wu
% Date   : June 2026
% Description: Model M1 — baseline (fixed intervention probability)
% =============================================================================

function lik = m1(x, data)

lik = 0;
p = x(1);
p = max(min(p, 1-1e-6), 1e-6);  % clip to avoid log(0)

for t = 1:length(data.block)
    a = data.action(t);  % AI: continuous p_yes in [0,1]
    lik = lik + a * log(p) + (1 - a) * log(1 - p);
end