% =============================================================================
% Author : Xiaoyan Wu
% Date   : June 2026
% Description: Model M9 — extended model (exploratory)
% =============================================================================

% setup parameter values
lamda = x(1);
bs = x(2);
bi = x(3);
bc = x(4);
br = x(5);
maxp = x(6);
minp = x(7);

lik = 0;

for t = 1:length(data.block)
    block = data.block(t);
    if block == 2
        block = -1;
    else
        block = 1;
    end
    ratio  = data.multi(t);       % ← ratio → multi
    x1     = data.MA(t);          % ← violator → MA
    x2     = data.MB(t);          % ← victim → MB
    inequa = max(x1 - x2, 0);
    cost   = data.cost(t);
    a      = data.action(t);      % continuous p_yes in [0,1]

    p = 1 / (1 + exp(lamda * (bs*block + bi*inequa + bc*cost + br*ratio)));
    p = minp + (1 - maxp - minp) * p;
    p = max(min(p, 1-1e-6), 1e-6);

    lik = lik + a * log(p) + (1 - a) * log(1 - p);
end