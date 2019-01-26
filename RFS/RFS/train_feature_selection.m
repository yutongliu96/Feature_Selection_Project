
function [W, b] = train_feature_selection(X, class_id, lambda_para, iters, epsilon)

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
end

[W, b] = optimize_L21(X, Y);



return;