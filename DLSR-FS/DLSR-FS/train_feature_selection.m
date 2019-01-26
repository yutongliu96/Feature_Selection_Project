
function [W, b] = train_feature_selection(X, class_id, lambda_para, u_para, iters, epsilon)

% X:                            each column is a data point
% class_id:                a column vector, a column vector,  such as  [1, 2, 3, 4, 1, 3, 2, ...]'
% lambda_para:      The lambda parameter in Equation (29), in the paper 
% u_para:                  the parameter c in the theorem, it is a very large positive number, generally, it is infinite. 
% iters:                       the largest number of iterations
% epsilon:                  For convergence  control

[dim, N] = size(X);
num_class = max(class_id);

Y = zeros(num_class, N);
B = -1 * ones(N, num_class);


for i = 1 : N
    Y( class_id(i),  i) = 1.0;  
    B(i, class_id(i) )  = 1.0;
end

[W0, b0] = least_squares_regression(X,  Y,  lambda_para);   % Here we use the soultion to the standard least squares regreesion as the initial solution  
W = W0;   
b = b0;

XX = [X; u_para * ones(1, N)];                                                       % construct the new training data by using homogeneous coordinates 

for i = 1: iters
    
    %first, optimize matrix M.
    P = X' * W0 + ones(N, 1) * b0' - Y';     
    
    M = optimize_m_matrix(P, B);
    
    % optimize W and b by using the theorem
    R  = Y' + (B .* M);
    [TT, obj] = optimize_L21(XX', R, lambda_para);
    W = TT(1:dim, :);
    b =    u_para * TT(dim + 1, :)';      
    
    if ( trace ( (W - W0)' * (W - W0) )  + ( b - b0)' *  (b - b0)   <  epsilon)     % check if it reaches the convergence point 
        break;
    end
    
    W0 = W;
    b0 = b;
      
end


return;