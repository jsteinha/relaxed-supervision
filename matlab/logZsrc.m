function ret = logZsrc(x, theta)
    global L;
    ret = 0.0;
    for i=1:L
        ret = ret + logsumexp(theta(x(i),:));
    end
end