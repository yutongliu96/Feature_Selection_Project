
function M = optimize_m_matrix(P, B)

% P:              the residual matrix, each row is a residual vector
% B:              construction matrix related to class label, each row is a constructtion vector

% return:        The optimized matrix

N = size(P, 1);
num_class = size(B, 2);

M1 = zeros(N, num_class);

M = max( B .* P,  M1); 

return;