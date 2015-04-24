function [X,Y] = generate_data(W, L, n)
    X = zeros(n, L);
    for i=1:n
        for j=1:2:L
            X(i,j) = 3*randi(W/3,1)-2;
            X(i,j+1) = X(i,j) + randi(2,1);
        end
    end
    Y = X;
end