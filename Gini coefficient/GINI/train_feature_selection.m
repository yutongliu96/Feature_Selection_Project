
function [W, b] = train_feature_selection(X, class_id)

[dim, N] = size(X);
num_class = max(class_id);

Y = zeros(num_class, N);
for i = 1 : N
    Y( class_id(i),  i) = 1.0;  
end

[W] = solve_iteratively_L21(X, Y);



return;