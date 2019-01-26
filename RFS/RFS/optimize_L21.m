
function [W, obj] = optimize_L21(X, Y, lambda_para)
%% 21-norm loss with 21-norm regularization
%: each row is a data point

%  Note that: 
%  min_X  || A X - Y||_21 + lambda_para * ||X||_21       is equivalent to the following problem:

%  min_X  ||X||_21 + ||E||_21
%  s.t.   A X + lambda_para*E = Y





[d n] = size(X);
[W, obj] = solve_iteratively_L21([X', lambda_para * eye(n)], Y);
W = W(1:n, :);
obj = lambda_para * obj;
