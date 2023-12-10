clear
clc
%% Abeline DATA
% Prepare (raw) data
data = load('data\Abilene.mat');

% Matching times (A week ranges from 8:14-dec-2003).
fprintf('Preparing Abilene dataset from December 8 to 14, 2003 ...... ')
smptime1 = data.utc2/(24*60*60)+datetime('1970-01-01 00:00:00'); 
timebgn1 = datetime('2003-12-08 00:00:00');  dt = 5/(60*24);
timeseries1 = timebgn1:dt:timebgn1+7-dt/2;

X1 = zeros(size(data.X2));  
for i=1:length(timeseries1)
    [v,j] = min(abs(smptime1-timeseries1(i)));
    if v<dt
        X1(i,:) = data.X2(j,:);
    end
end

% Matching origin-destination pairs.
ods = {'ATLA','CHIN','DNVR','HSTN','IPLS','KSCY','LOSA','NYCM','SNVA','STTL','WASH'};
Abilene_OD = zeros(length(ods),length(ods),length(timeseries1));
for j=1:length(ods)
    for i=1:length(ods)
        index = strcmp(data.odnames,[ods{i},'-',ods{j}]);
        Abilene_OD(i,j,:) = X1(:,index);
    end
end
fprintf('[Done]\n');
fprintf('An OD tensor of size: OxDxT = %2dx%2dx%3d = %d. \n',size(Abilene_OD),numel(Abilene_OD));
clear smptime1 v X1 dt index i j

%% DEMO
dense_tensor = reshape(Abilene_OD,[121,288,7]); R = size(dense_tensor);
idx = 1:numel(dense_tensor);
idx = idx(dense_tensor(:)>0);
mask = sort(randperm(length(idx),round(0.1*numel(dense_tensor))));
arti_miss_idx = idx;  
arti_miss_idx(mask) = [];  
arti_miss_mv = dense_tensor(arti_miss_idx);
Omega = zeros(size(dense_tensor)); Omega(mask) = 1; Omega = boolean(Omega);
RM_P = length(find(Omega(:)==1)); sample_ratio = RM_P/numel(dense_tensor);
Opts = Initial_Para(250,R,'lrstd',0.5,1,1e-5); Opts.weight = 'sum'; Opts.flag = [1,1,0]; Opts.prior = 'lrstd';
[PALM_estimation_tensor, PALM_U, PALM_info_S_LRSTD_teop] = PALM_LRSTD(dense_tensor,Omega,Opts); 
% [IALM_estimation_tensor, IALM_U, IALM_info] = IALM_LRSTD(dense_tensor,Omega,Opts); 
sparse_tensor = Omega.*dense_tensor;
est_tensor = PALM_estimation_tensor;
NMAE = norm(arti_miss_mv-est_tensor(arti_miss_idx),1) / norm(arti_miss_mv,1);
RMSE = sqrt((1/length(arti_miss_mv))*norm(arti_miss_mv-est_tensor(arti_miss_idx),2)^2);  
MAPE = (100/length(arti_miss_mv))* sum(abs((arti_miss_mv-est_tensor(arti_miss_idx))./arti_miss_mv));
%% TESTs UNDER RM
dense_tensor = reshape(Abilene_OD/1e4,[121,288,7]);
Eval_LSPTD = zeros(4,11);
DLP = [0.1:0.1:0.9,0.93,0.95];
for MR = 1:length(DLP)
    rng('default')
    sample_ratio = 1- DLP(MR);
    sample_num = round(sample_ratio*numel(dense_tensor));
    fprintf('Sampling OD tensor with %4.1f%% known elements ...... \n',100*sample_ratio);
    % Filter missing positions 
    idx = 1:numel(dense_tensor);
    idx = idx(dense_tensor(:)>0);
    % Artificial missing position
    mask = sort(randperm(length(idx),sample_num));
    arti_miss_idx = idx;  
    arti_miss_idx(mask) = [];  
    arti_miss_mv = dense_tensor(arti_miss_idx);
    Omega = zeros(size(dense_tensor)); Omega(mask) = 1; Omega = boolean(Omega);

    t0 = tic;
    Opts = Initial_Para(250,[121,288,7],'lrstd',0.5,1e-6,1e-5); Opts.weight = 'sum'; Opts.flag = [1,1,0]; Opts.prior = 'lrstd';
    [est_tensor, ~, ~] = PALM_LRSTD(dense_tensor,Omega,Opts); 

    Eval_LSPTD(4,MR) = toc(t0);
    nmae = norm(arti_miss_mv-est_tensor(arti_miss_idx),1) / norm(arti_miss_mv,1);
    RMSE = sqrt((1/length(arti_miss_mv))*norm(arti_miss_mv-est_tensor(arti_miss_idx),2)^2);  
    MAPE = (100/length(arti_miss_mv))* sum(abs((arti_miss_mv-est_tensor(arti_miss_idx))./arti_miss_mv));
    Eval_LSPTD(1,MR) = MAPE; Eval_LSPTD(2,MR) = 1e4*RMSE; Eval_LSPTD(3,MR) = nmae; 

end