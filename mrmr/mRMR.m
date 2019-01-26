close all;

tic;
addpath('./libsvm-mat-3.0-1');
filename = ['./data/AR.mat'];
load (filename);
addpath('./fs_sup_mrmr');
% fea=im2double(fea);
fea = fea/255;
lambda = [0.01 0.1 1 10 100 1000 10000 100000];% the parameter of feature selection
length_lamada = length(lambda);
C = [0.0001 0.001 0.01 0.1 1 10 100];% the parameter of SVM
length_C = length(C);
rate=zeros(1,20);
fold = 10; % 10 fold cross validation
d = [10 20 30 40 50 60 70 80]; % the number of selected features
length_d = length(d);
epsilon = 0.0001;
iters = 30;

for i=1:20%20 trials
    temp_rate=zeros(fold,length_lamada,length_d,length_C);
    filename = strcat('./data/2Train/',num2str(i));
    load (filename);
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    fea_Train = fea(:,trainIdx);
    gnd_Train = gnd(trainIdx);
    fea_Test = fea(:,testIdx);
    gnd_Test = gnd(testIdx);
    
    %%%%%%%%%%%%%%%%%%%%%%Cross validation begins%%%%%%%%%%%%%%%%%%%%%%
    indices = crossvalind('Kfold',gnd_Train,fold);
    for j = 1:fold%1
        test = (indices == j); train = ~test;
        temp_Train = fea_Train(:,train);
        temp_Test = fea_Train(:,test);
        temp_gnd_Train = gnd_Train(train);
        temp_gnd_Test = gnd_Train(test);
        for k = 1:length_lamada%1
            % The first step: Train the model
            [W] = fsMRMR_parson(temp_Train', temp_gnd_Train);
            %   The second step:  Select the features and output the selected
            %   features
            WW = W.W .^ 2;
            W_weight = sum(WW, 2); % sum the element row-by-row
            [Weight, index_sorted_features] = sort(-W_weight); %  sort them from the largest to the smallest
            % output the features
            for n=1:length_d
                index_features_finally_seelcted = index_sorted_features(1 : d(n));
                % perform classification
                for m=1:length_C%1
                    SVMParameter=sprintf('-c %f -t 0',C(m));
                    model = svmtrain(temp_gnd_Train, temp_Train(index_features_finally_seelcted,:)', SVMParameter); %linear kernel
                    [predict_label, rate1, dec_values] = svmpredict(temp_gnd_Test, temp_Test(index_features_finally_seelcted,:)', model);
                    temp_rate(j,k,n,m) = rate1(1);
                    clear SVMParameter model predict_label rate1 dec_values;
                end
                clear index_features_finally_seelcted;
            end
            clear  W b WW W_weight Weight index_sorted_features;
        end
        clear temp_Train temp_Test temp_gnd_Train temp_gnd_Test test train;
    end
    %%%%%%%%%%%%%%%%%%%%%%Cross validation ends%%%%%%%%%%%%%%%%%%%%%%
    for n=1:length_d
        temp_rate1 = mean(temp_rate(:,:,n,:),1);
        temp_rate2 = reshape(temp_rate1,length_lamada,length_C);
        clear temp_rate1;
        ind = find(temp_rate2 == max(max(temp_rate2)));
        clear  temp_rate2;
       
        if mod(ind(1),length_lamada)==0
            kk = length_lamada;
        else
            kk = mod(ind(1),length_lamada);
        end
        mm = ceil(ind(1)/length_lamada);
        clear ind;
        
        [W] = fsMRMR_parson(fea_Train', gnd_Train);
        WW = W.W .^ 2;
        W_weight = sum(WW, 2); % sum the element row-by-row
        [Weight, index_sorted_features] = sort(-W_weight); %  sort them from the largest to the smallest
        % output the features
        index_features_finally_seelcted = index_sorted_features(1 : d(n));
        clear  W b WW W_weight Weight index_sorted_features ;
        SVMParameter=sprintf('-c %f -t 0',C(mm));
        model = svmtrain(gnd_Train, fea_Train(index_features_finally_seelcted,:)', SVMParameter); %linear kernel
        [predict_label, rate2, dec_values] = svmpredict(gnd_Test, fea_Test(index_features_finally_seelcted,:)', model);
        rate(n,i) = rate2(1);
        clear SVMParameter model predict_label rate2 dec_values index_features_finally_seelcted;
        save AR_DLSR_FS_2Train rate;
    end
    clear fea_Train gnd_Train fea_Test gnd_Test;
end
% draw the figure directly
hold on;
errorbar(d,mean(rate,2),std(rate,1,2));
toc;

