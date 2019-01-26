
function [X, obj] = solve_iteratively_L21(A, Y)
%% minimize 21-norm with

% Solve the following equivalent problem: 
%  min_X  ||X||_21
%  s.t.   A X = Y

n = size(A,  2);
m = size(A,  1);

ITER = 10;
obj = zeros(ITER,1);
d = ones(n, 1);                                              % initialization

epsilon = 10^-5;
obj1 = -1000;

for iter = 1 : ITER
    D = spdiags(d, 0, n, n);
    lambda = ((A * D) * A') \ Y;
    X = D *(A' * lambda);
    d = sqrt(sum(X .* X,2))  + 0.00000001;   
    % d = sqrt(sum(X .* X,2));  
     
    obj(iter) = sum(d);
    
    if abs( obj(iter) -  obj1) < epsilon
        break;
    end
    obj1 = obj(iter);
end


 return;