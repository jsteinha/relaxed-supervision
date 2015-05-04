global theta beta W L;
options = sdpsettings('solver', '', 'verbose', 2, 'usex0', 1, 'fmincon.UseParallel', 'always', ...
    'fmincon.Algorithm', 'interior-point', 'fmincon.Diagnostics', 'on', 'fmincon.TolCon', 1e-2, 'fmincon.TolFun', 1e-3);
W = 21; % number of words
L = 12; % length of sentence
n = 50;
S = 10;
theta = zeros(W, W);
beta = zeros(W, 1);
[X, Y] = generate_data(W, L, n); % generate n data points
%%
for t = 1:20
    % create variables
    theta_s = sdpvar(W, W, 'full');
    beta_s = sdpvar(W,1);
    assign(theta_s, theta);
    assign(beta_s, beta);
    Constraint = 0;
    Objective = 0;
    total_samples = 0;
    xcount = zeros(W, 1);
    for i=1:n
        %disp(i);
        xi = X(i,:);
        yi = Y(i,:);
        diff = 0;
        logZtar = 0;
        num_samples = 0;
        indices = zeros(W*W, 1);
        for s = 1:S
            [zi_cur, diff_cur, logZtar_cur,num_samples_cur] = sample(xi, yi);
            indices_cur = sub2ind(size(theta), xi, zi_cur);
            indices(indices_cur) = indices(indices_cur) + 1.0 / S;
            diff = diff + (diff_cur * 1.0 / S);
            logZtar = logZtar + (logZtar_cur * 1.0 / S);
            num_samples = num_samples + num_samples_cur;
        end
        total_samples = total_samples + num_samples;
        if mod(i, 20) == 0
            fprintf(1, '\t%d samples\n', total_samples);
        end
        % update constraint
        LogConstraint = - sum((theta_s(:) - theta(:)) .* indices) - logZtar - logZsrc2(xi, yi, theta);
%        for j=1:L
%            Constraint = Constraint - (theta_s(xi(j), zi(j)) - theta(xi(j), zi(j))) - logZtar;
%        end
        LogConstraint = LogConstraint + sum(beta_s .* diff);
        LogConstraint = LogConstraint + logZsrc2(xi, yi, theta_s);
        Constraint = Constraint + exp(LogConstraint);
        % update objective
        Objective = Objective - sum((theta_s(:) - theta(:)) .* indices) - logZtar - logZsrc2(xi, yi, theta);
%        for j=1:L
%            Objective = Objective - (theta_s(xi(j), zi(j)) - theta(xi(j), zi(j))) - logZtar;
%        end
        Objective = Objective + sum(beta_s .* diff);
        xcount = xcount + accumarray(xi', 1, [W 1]);
    end
    % Note: we can speed up all of these by computing all at once
    Objective = Objective + n * sum(log(1 + (L-1) * exp(-beta_s)));
    Objective = Objective + logZsrc(xcount, theta_s);
    NegEntropy = Ephi(xcount, theta_s) - logZsrc(xcount, theta_s);
    Objective = Objective + 0.1 * NegEntropy;
    fprintf(1, 'total samples: %d\n', total_samples);
    % optimize
    Regularizer = 0;
    for i=1:W
        CurDist = sum(theta_s(i,:) - theta(i,:));
        Regularizer = Regularizer + 0.5 * CurDist * CurDist;
    end
    fprintf(1, 'submitting optimization instance...\n');
    solvesdp([(Constraint/n) <= 200.0; beta_s >= 1.0/L; beta_s <= 5.0], Objective/n + Regularizer, options);
    %solvesdp([], Objective/n, options);
    % update parameters
    theta = double(theta_s);
    beta = double(beta_s);
    disp(beta);
    p = zeros(W,W);
    for j=1:W
        p(j,:) = exp(theta(j,:) - logsumexp(theta(j,:)));
    end
    disp(p);
    disp(sum(diag(p)));
end