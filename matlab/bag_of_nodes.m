global theta beta W L;
options = sdpsettings('solver', '', 'verbose', 2, 'usex0', 1, 'fmincon.UseParallel', 'always', ...
    'fmincon.Algorithm', 'interior-point', 'fmincon.Diagnostics', 'on', 'fmincon.TolCon', 1e-2, 'fmincon.TolFun', 1e-3);
W = 42; % number of words
L = 24; % length of sentence
n = 400;
theta = zeros(W, W);
beta = zeros(W, 1);
[X, Y] = generate_data(W, L, n); % generate n data points
for t = 1:20
    % create variables
    theta_s = sdpvar(W, W);
    beta_s = sdpvar(W,1);
    assign(theta_s, theta);
    assign(beta_s, beta);
    Constraint = 0;
    Objective = 0;
    total_samples = 0;
    for i=1:n
        %disp(i);
        xi = X(i,:);
        yi = Y(i,:);
        [zi, diff, logZtar,num_samples] = sample(xi, yi);
        total_samples = total_samples + num_samples;
        if mod(i, 20) == 0
            fprintf(1, '\t%d samples\n', total_samples);
        end
        % update constraint
        indices = sub2ind(size(theta), xi, zi);
        LogConstraint = - sum(theta_s(indices) - theta(indices)) - logZtar - logZsrc2(xi, yi, theta);
%        for j=1:L
%            Constraint = Constraint - (theta_s(xi(j), zi(j)) - theta(xi(j), zi(j))) - logZtar;
%        end
        LogConstraint = LogConstraint + sum(diff .* beta_s);
        LogConstraint = LogConstraint + logZsrc2(xi, yi, theta_s);
        Constraint = Constraint + exp(LogConstraint);
        % update objective
        Objective = Objective - sum(theta_s(indices) - theta(indices)) - logZtar - logZsrc2(xi, yi, theta);
%        for j=1:L
%            Objective = Objective - (theta_s(xi(j), zi(j)) - theta(xi(j), zi(j))) - logZtar;
%        end
        Objective = Objective + sum(diff .* beta_s);
        Objective = Objective + sum(log(1 + (W-1) * exp(-beta_s)));
        Objective = Objective + logZsrc(xi, theta_s);
    end
    fprintf(1, 'total samples: %d\n', total_samples);
    % optimize
    solvesdp([(Constraint/n) <= 200.0; beta_s >= 0], Objective/n, options);
    %solvesdp([], Objective/n, options);
    % update parameters
    theta = double(theta_s);
    beta = double(beta_s);
    disp(beta);
    p = zeros(W,1);
    for j=1:W
        p(j) = exp(theta(j,j) - logsumexp(theta(j,:)));
    end
    disp(p);
    disp(sum(p));
end