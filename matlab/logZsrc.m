function ret = logZsrc(xc, theta)
    global W;
    ret = 0.0;
    for j=1:W
        ret = ret + xc(j) * logsumexp(theta(j,:));
    end
end