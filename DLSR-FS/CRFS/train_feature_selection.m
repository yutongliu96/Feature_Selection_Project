function [W, b] = train_feature_selection(X, class_id, lambda_para, iters, epsilon)

% X:                            each column is a data point
% class_id:                a column vector, a column vector,  such as  [1, 2, 3, 4, 120, 36, 2, ...]'
% iters:                       the largest number of iterations
% epsilon:                  For convergence  control

[dim, N] = size(X);
num_class = max(class_id);

Y = zeros(num_class, N);

for i = 1 : N
    Y( class_id(i),  i) = 1.0;  
end

[W, b] = solve_iteratively(X, Y, lambda_para);



return;