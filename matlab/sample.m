function [z,diff,logZtar,num_samples] = sample(x, y)
    global L W beta;
    u = unique(y);
    %disp(y);
    a = accumarray(y', 1, [W 1]);
    % take a bunch of samples
    num_samples = 1;
    logZtar = -Inf;
    while 1
        z = zeros(1, L);
        for j=1:L
            z(j) = sampleOnce(x(j), u);
        end
        diff = norm(max(accumarray(z',1,[W 1])-a, 0), 1);
        logZtar = logsumexp([logZtar, -beta * diff]);
        % see if we should accept
        if rand < exp(-beta * diff)
            % if so, return estimate of normalization constant
            logZtar = logZtar - log(num_samples);
            %disp(num_samples);
            return;
        end
        num_samples = num_samples + 1;
    end
end

function zj = sampleOnce(xj, u)
    global theta;
    %fprintf(1, 'xf: %f, u: %f\n', xj, u);
    logZ = logsumexp(theta(xj, u));
    v = rand;
    cur = -Inf;
    for uu = u
        cur = logsumexp([cur, theta(xj, uu)]);
        % fprintf(1, 'uu = %d, cur = %f\n', uu, cur);
        if v < exp(cur - logZ)
            zj = uu;
            return;
        end
    end
    disp('BAD');
    zj = NaN;
end