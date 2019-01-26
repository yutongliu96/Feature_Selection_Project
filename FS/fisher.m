function score = fisher(fea,gnd)
[d,n]=size(fea);
labels=unique(gnd);
k=length(labels);
numclass=zeros(1,k);
idx=cell(k,1);
for i=1:k
    idx{i,1}=find(gnd==i);
    numclass(i)=length(find(gnd==i));
end

vector=cell(k,1);
mu=zeros(k,1);
sigma=zeros(k,1);
score=zeros(d,1);
for i=1:d
    muall=mean(fea(i,:));
    sum_mu=0;
    sum_sig=0;
    for j=1:k
        %mean and variance of i-th feature and k-th class
        vector{j,1}=fea(i,idx{j,1});
        mu(j) = mean(vector{j,1});
        sigma(j)=std(vector{j,1});
        sum_mu = sum_mu+numclass(j)*(mu(j)-muall)^2;
        sum_sig=sum_sig+numclass(j)*(sigma(j))^2;
    end
    score(i)=sum_mu/sum_sig;
end


    


