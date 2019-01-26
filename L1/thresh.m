function f=thresh(fea, gnd, lambda)
[y,A] = ProducePoNe(gnd,fea);
[p,n] = size(A);
tau=lambda;

for i =1:n
    if iqr(A(:,i))>0
    A(:,i)=(A(:,i)-median(A(:,i)))/iqr(A(:,i));
    end
end
% normalization
lam_max=max(svds(A));
g=zeros(size(y));
for i=1:p
    for j=1:n
        A(i,j)=A(i,j)/lam_max;
        g(j)=y(j)/lam_max;
    end
end
f=zeros(p,1);
for i=1:p
    f(i)=0;
end
%threshold
f0=1;
while norm(f-f0)>=norm(f)/100
    f0=f;
    f=f+A*(g-A'*f); 
    for i=1:p
        if abs(f(i))>=tau
            f(i)=f(i)-tau*sign(f(i));
        else
            f(i)=0;
        end
    end
end
