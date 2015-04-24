global theta beta W L;
options = sdpsettings('solver', '', 'verbose', 2, 'usex0', 1);
W = 21; % number of words
L = 6; % length of sentence
n = 20;
theta = zeros(W, W);
beta = 0;
[X, Y] = generate_data(W, L, n); % generate n data points
for t = 1:10
    cvx_begin sdp
        % create variables
        variables theta_s(W,W) beta_s;
        Constraint = 0;
        Objective = 0;
        total_samples = 0;
        for i=1:n
            %disp(i);
            xi = X(i,:);
            yi = Y(i,:);
            [zi, diff, logZtar, num_samples] = sample2(xi, yi);
            total_samples = total_samples + num_samples;
            u = unique(yi);
            logZSrc2 = 0.0;
            logZSrc20 = 0.0;
            for j=1:L
                logZSrc2 = logZSrc2 + log_sum_exp(theta_s(xi(j),u));
                logZSrc20 = logZSrc20 + log_sum_exp(theta(xi(j),u));
            end
            % update constraint
            indices = sub2ind(size(theta), xi, zi);
            Constraint = Constraint - sum(theta_s(indices) - theta(indices)) - logZtar - logZSrc20;
            Constraint = Constraint + diff * beta_s;
            Constraint = Constraint + logZSrc2;
            % update objective
            Objective = Objective - sum(theta_s(indices) - theta(indices)) - logZtar - logZSrc20;
            Objective = Objective + diff * beta_s;
            Objective = Objective + L * log(1 + (W-1) * exp(-beta_s));
            logZSrc = 0.0;
            for j=1:L
                logZSrc = logZSrc + log_sum_exp(theta_s(xi(j),:));
            end
            Objective = Objective + logZSrc;
        end
        fprintf(1, 'total samples: %d\n', total_samples);
        % optimize
        minimize( Objective/n + 0.1 * sum(sum(theta_s.^2)) );
        subject to
            (Constraint/n) <= 0.2;
            beta_s >= 0;            
    cvx_end
    % update parameters
    theta = double(theta_s);
    beta = double(beta_s);
    disp(beta);
    p = zeros(W,1);
    for j=1:W
        p(j) = exp(theta(j,j) - log_sum_exp(theta(j,:)));
    end
    disp(p);
    disp(sum(p));
end