
function [W, b] = least_squares_regression(X,  Y,  gamma)

% X:                                     each column is a data point
% Y:                                     each column is an target data point: such as  [0, 1, 0, ..., 0]'
% gamma:                             a positive scalar

% return 
% W and b
% here we use the following equivalent model:   y = W' x + b 

[dim, N] = size (X);
[dim_reduced, N] = size(Y);

% first step,  remove the mean!
XMean = mean(X')';                                       % is a column vector
XX = X - repmat(XMean, 1, N);                    % each column is a data point

W = [];
b = [];
if dim < N
    
    t0 =  XX * XX' + gamma * eye(dim);
    W = t0 \ (XX * Y');   
    
    b = Y - W' * X;     % each column is an error vector
    b = mean(b')';        % now b is a column vector
    
else
    t0 = XX' * XX + gamma * eye(N);
    W = XX * (t0 \ Y');
     
   
     b = Y - W' * X;     % each column is an error vector
     b = mean(b')';        % now b is a column vector
    
end



