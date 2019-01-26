function score=t_test(fea,gnd)
[d,n]=size(fea);
label=unique(gnd);
k=length(label);
numclass=zeros(1,k);
idx=cell(k,1);
for i=1:k
    idx{i,1}=find(gnd==i);
    numclass(i)=length(find(gnd==i));
end

vector=cell(k,1);
mu=zeros(k,1);
M=zeros(k,1);
sum_de=0;
absx=zeros(k,1);
score=zeros(d,1);
for i=1:d
    muall=mean(fea(i,:));
    for j=1:k
        vector{j,1}=fea(i,idx{j,1});
        mu(j)=mean(vector{j,1});
        M(j)=sqrt(1/numclass(j)+1/n);
        sum_de= sum_de+sum((vector{j,1}-mu(j)).^2);
    end
    sum_de=sum_de/(n-k);
    for j=1:k
        absx(j)=abs(mu(j)-muall)/(M(j)*sum_de);
    end
    score(i)=max(absx);
end

    
        
        
        
        
        