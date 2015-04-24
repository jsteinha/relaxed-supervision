function ret = logZsrc2(x, y, theta)
%    global L;
    L = length(x);
    u = unique(y);
    %ret = sum(logsumexp(theta(x,u)'));
    ret = 0.0;
    for i=1:L
        ret = ret + logsumexp(theta(x(i),u));
    end
end