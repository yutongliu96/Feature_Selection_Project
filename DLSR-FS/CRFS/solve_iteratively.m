function [U, obj] = solve_iteratively(X, Y,lambda_para)
n = size(X,  2);
d = size(X,  1);
Y=Y';%Y is n*c matrix
c=size(Y,2);
ITER = 30;
obj = zeros(ITER,1);
U=zeros(d,c);
epsilon = 10^-5;
obj1 = -1000;
P = eye(n);
Q = eye(d);
for iter = 1 : ITER
    temp=X'*U-Y;
    for k=1:n
        P(k,k)=exp(-(temp(n,:)*temp(n,:)')/768);
    end
    for j=1:d
        Q(j,j)=1/(2*(U(n,:)*U(n,:)')^0.5);
    end
    U=(X*P*X'+lambda_para*Q)\(X*P*Y);
    d = sqrt(sum(U .* U,2));  
     
    obj(iter) = sum(d);
    
    if abs( obj(iter) -  obj1) < epsilon
        break;
    end
    obj1 = obj(iter);
end


 return;