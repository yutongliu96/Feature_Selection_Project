function [X, obj] = solve_iteratively_L21(A, Y)
%% minimize 21-norm with equality constraints

% Solve the following equivalent problem: 
%  min_X  ||X||_21
%  s.t.   A X = Y

m = size(A,  2);
n = size(A,  1);

ITER = 10;
obj = zeros(ITER,1);
d = ones(n, 1);                                              % initialization

epsilon = 10^-5;
obj1 = -1000;
D = eye(m);
for iter = 1 : ITER
    D_inv=inv(D);
    lambda = ((A * D_inv) * A') \ Y';
    X = D_inv *(A' * lambda);
    for j=1:n
        D(j,j)=1/(2*X(n,:)*X(n,:)');
    end
    d = sqrt(sum(X .* X,2));   
     
    obj(iter) = sum(d);
    
    if abs( obj(iter) -  obj1) < epsilon
        break;
    end
    obj1 = obj(iter);
end


 return;