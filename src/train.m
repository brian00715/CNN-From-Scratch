clear all; %#ok<CLALL>
addpath("cnn_core");
addpath("utils");

date_prefix = string(datetime('now', 'Format', 'MM-dd_HH-mm-ss'));
log_path = "../logs/"+date_prefix + "/";
mkdir(log_path);

%% load dataset
load_from_file = true;

if ~load_from_file
    data_path = "../data/";
    dataset_option.load_raw = false;
    dataset_option.shuffle = true;
    dataset_option.img_dim = 124;
    dataset_option.train_ratio = 0.8;
    dataset_option.save = true;
    dataset_option.transform = true;
    [images_train, labels_train, images_test, labels_test] = loadDataset(data_path, dataset_option);
else
    load("../data/train.mat");
    load("../data/test.mat");
end

%% define network structure

% 32x32->(conv)28x28x6->(pool)14x14x6->(conv)10x10x16->(pool)5x5x16=400->(linear)120->(linear)84
% cnn.layers = {
%               struct('type', 'input') %input layer
%               struct('type', 'Conv2D', 'filterDim', 5, 'numFilters', 6, 'poolDim', 2, 'actiFunc', 'relu')
%               struct('type', 'Conv2D', 'filterDim', 5, 'numFilters', 16, 'poolDim', 2, 'actiFunc', 'relu')
%               struct('type', 'Linear', 'hiddenUnits', 120, 'actiFunc', 'tanh')
%               struct('type', 'Linear', 'hiddenUnits', 84, 'actiFunc', 'tanh')
%               struct('type', 'output', 'softmax', 1)
%               };

% 68x68 ->(conv) 64x64x8 ->(pool) 32x32x8 ->(conv) 28x28x16 ->(pool) 14x14x16 ->(conv) 10x10x32 ->(pool) 5x5x32=1000 ->(linear) 200 ->(linear) 100
% cnn.layers = {
%               struct('type', 'input') %input layer
%               struct('type', 'Conv2D', 'filterDim', 5, 'numFilters', 8, 'poolDim', 2, 'actiFunc', 'relu')
%               struct('type', 'Conv2D', 'filterDim', 5, 'numFilters', 16, 'poolDim', 2, 'actiFunc', 'relu')
%               struct('type', 'Conv2D', 'filterDim', 5, 'numFilters', 32, 'poolDim', 2, 'actiFunc', 'relu')
%               struct('type', 'Linear', 'hiddenUnits', 200, 'actiFunc', 'tanh')
%               struct('type', 'Linear', 'hiddenUnits', 100, 'actiFunc', 'tanh')
%               struct('type', 'output', 'softmax', 1)
%               };

% 124x124->(conv)120x120x8->(pool)30x30x8->(conv)26x26x16->(pool)13x13x16->(conv)9x9x32->(pool)3x3x32=288
cnn.layers = {
              struct('type', 'input') %input layer
              struct('type', 'Conv2D', 'filterDim', 5, 'numFilters', 8, 'poolDim', 4, 'actiFunc', 'relu')
              struct('type', 'Conv2D', 'filterDim', 5, 'numFilters', 16, 'poolDim', 2, 'actiFunc', 'relu')
              struct('type', 'Conv2D', 'filterDim', 5, 'numFilters', 32, 'poolDim', 3, 'actiFunc', 'relu')
              struct('type', 'Linear', 'hiddenUnits', 100, 'actiFunc', 'relu', 'dropout', 0.2)
              struct('type', 'Linear', 'hiddenUnits', 50, 'actiFunc', 'relu')
              struct('type', 'output', 'softmax', 1)
              };

% 124x124->(conv)120x120x8->(pool)60x60x8->(conv)56x56x16->(pool)28x28x16->(conv)24x24x32->(pool)12x12x32->(conv)8x8x64->(pool)4x4x64=1024
% cnn.layers = {
%               struct('type', 'input') %input layer
%               struct('type', 'Conv2D', 'filterDim', 5, 'numFilters', 8, 'poolDim', 2, 'actiFunc', 'relu')
%               struct('type', 'Conv2D', 'filterDim', 5, 'numFilters', 16, 'poolDim', 2, 'actiFunc', 'relu')
%               struct('type', 'Conv2D', 'filterDim', 5, 'numFilters', 32, 'poolDim', 2, 'actiFunc', 'relu')
%               struct('type', 'Conv2D', 'filterDim', 5, 'numFilters', 64, 'poolDim', 2, 'actiFunc', 'relu')
%               struct('type', 'Linear', 'hiddenUnits', 512, 'actiFunc', 'tanh')
%               struct('type', 'Linear', 'hiddenUnits', 256, 'actiFunc', 'tanh')
%               struct('type', 'output', 'softmax', 1)
%               };
%% define training options
options.epochs = 40;
options.minibatch = 64;
options.lr_max = 0.05;
options.lr = options.lr_max;
options.lr_min = 1e-7;
options.lr_method = 'cosine_cyclic';
options.lr_duty = 10; % duty cycle for cosine lr
options.momentum = 0.9;
options.log_path = log_path;
options.l2_penalty = 0.01;
options.use_l2 = false;
options.save_best_acc_model = true;
options.train_mode = true;
total_iter = round(floor(size(images_train, 4) / options.minibatch) * options.epochs);
options.total_iter = total_iter;
fprintf('Total iterations: %d\n', total_iter);

%% train
numClasses = max(labels_train);
cnn = initModelParams(cnn, images_train, numClasses);
% load("logs/cnn_mnist.mat");
cnn = learn(cnn, images_train, labels_train, images_test, labels_test, options);

%% test
preds = predict(cnn, images_test, options);
acc = sum(preds == labels_test) / length(preds);
fprintf('Final accuracy: %f\n', acc);

%% save model and parameters
fileID = fopen(log_path + "results.txt", 'a');
fprintf(fileID, 'Final accuracy: %f\n', acc);
fclose(fileID);
save(log_path + "cnn.mat", 'cnn');
save(log_path + "results_on_test.mat", 'preds', 'labels_test');
save(log_path + "hyper_params.mat", 'dataset_option', 'options');
