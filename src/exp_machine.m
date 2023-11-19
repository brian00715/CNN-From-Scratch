% Description: automatically execute all the experiments

clear all;
close all;
addpath("utils");
addpath("cnn_core");

% default options
load_from_file = false;
data_path = "../data/";
dataset_options.load_raw = false;
dataset_options.shuffle = true;
dataset_options.img_dim = 124;
dataset_options.train_ratio = 0.8;
dataset_options.save = false;
dataset_options.apply_rand_tf = false;
[data_train, labels_train, data_test, labels_test] = loadDataset(data_path, dataset_options);

train_options.epochs = 60;
train_options.minibatch = 64;
train_options.lr_max = 0.01;
train_options.lr = train_options.lr_max;
train_options.lr_min = 1e-7;
train_options.lr_method = 'cosine_cyclic';
train_options.lr_duty = 20; % epoches per cycle
train_options.momentum = 0.9;
train_options.l2_penalty = 0.01;
train_options.use_l2 = false;
train_options.save_best_acc_model = true;
train_options.train_mode = true;
total_iter = round(floor(size(data_train, 4) / train_options.minibatch) * train_options.epochs);
train_options.total_iter = total_iter;

% %% 1 model design
% %% 1.1 layer design
% sprintf("1.1 layer design")
% % 32x32->(conv)28x28x6->(pool)14x14x6->(conv)10x10x16->(pool)5x5x16=400
% cnn1.layers = {
%                struct('type', 'input') %input layer
%                struct('type', 'Conv2D', 'filterDim', 5, 'numFilters', 6, 'poolDim', 2, 'actiFunc', 'relu')
%                struct('type', 'Conv2D', 'filterDim', 5, 'numFilters', 16, 'poolDim', 2, 'actiFunc', 'relu')
%                struct('type', 'Linear', 'hiddenUnits', 120, 'actiFunc', 'relu', 'dropout', 0.2)
%                struct('type', 'Linear', 'hiddenUnits', 84, 'actiFunc', 'relu')
%                struct('type', 'output', 'softmax', 1)
%                };
% % 124x124->(conv)120x120x8->(pool)30x30x8->(conv)26x26x16->(pool)13x13x16->(conv)9x9x32->(pool)3x3x32=288
% cnn2.layers = {
%                struct('type', 'input') %input layer
%                struct('type', 'Conv2D', 'filterDim', 5, 'numFilters', 4, 'poolDim', 4, 'actiFunc', 'relu')
%                struct('type', 'Conv2D', 'filterDim', 5, 'numFilters', 8, 'poolDim', 2, 'actiFunc', 'relu')
%                struct('type', 'Conv2D', 'filterDim', 5, 'numFilters', 16, 'poolDim', 3, 'actiFunc', 'relu')
%                struct('type', 'Linear', 'hiddenUnits', 128, 'actiFunc', 'relu', 'dropout', 0.2)
%                struct('type', 'Linear', 'hiddenUnits', 64, 'actiFunc', 'relu')
%                struct('type', 'output', 'softmax', 1)
%                };
% % 124x124->(conv)120x120x8->(pool)60x60x8->(conv)56x56x16->(pool)28x28x16->(conv)24x24x32->(pool)12x12x32->(conv)8x8x64->(pool)4x4x64=1024
% cnn3.layers = {
%                struct('type', 'input') %input layer
%                struct('type', 'Conv2D', 'filterDim', 5, 'numFilters', 8, 'poolDim', 2, 'actiFunc', 'relu')
%                struct('type', 'Conv2D', 'filterDim', 5, 'numFilters', 16, 'poolDim', 2, 'actiFunc', 'relu')
%                struct('type', 'Conv2D', 'filterDim', 5, 'numFilters', 32, 'poolDim', 2, 'actiFunc', 'relu')
%                struct('type', 'Conv2D', 'filterDim', 5, 'numFilters', 64, 'poolDim', 2, 'actiFunc', 'relu')
%                struct('type', 'Linear', 'hiddenUnits', 512, 'actiFunc', 'relu', 'dropout', 0.2)
%                struct('type', 'Linear', 'hiddenUnits', 256, 'actiFunc', 'relu')
%                struct('type', 'output', 'softmax', 1)
%                };
% cnns = {cnn1, cnn2, cnn3};

% for subexp = 1:3
%     sprintf("1.1 layer design subexp: %d", subexp)
%     cnn = cnns{subexp};
%     log_path = sprintf("../logs/exp1_1/model_%d/", subexp);

%     if ~exist(log_path, 'dir')
%         mkdir(log_path);
%     end

%     if subexp == 1
%         dataset_options.img_dim = 32;
%     else
%         dataset_options.img_dim = 124;
%     end

%     [data_train, labels_train, data_test, labels_test] = loadDataset(data_path, dataset_options);

%     train_options.log_path = log_path;
%     trainMachine(cnn, dataset_options, train_options, data_train, labels_train, data_test, labels_test);
% end

% %% 1.2 conv layer design

% %% 1.2.1 filter number

% sprintf("1.2.1 filter number")

% cnn1.layers = {
%                struct('type', 'input') %input layer
%                struct('type', 'Conv2D', 'filterDim', 5, 'numFilters', 2, 'poolDim', 4, 'actiFunc', 'relu')
%                struct('type', 'Conv2D', 'filterDim', 5, 'numFilters', 4, 'poolDim', 2, 'actiFunc', 'relu')
%                struct('type', 'Conv2D', 'filterDim', 5, 'numFilters', 8, 'poolDim', 3, 'actiFunc', 'relu')
%                struct('type', 'Linear', 'hiddenUnits', 128, 'actiFunc', 'relu', 'dropout', 0.2)
%                struct('type', 'Linear', 'hiddenUnits', 64, 'actiFunc', 'relu')
%                struct('type', 'output', 'softmax', 1)
%                };
% cnn2.layers = {
%                struct('type', 'input') %input layer
%                struct('type', 'Conv2D', 'filterDim', 5, 'numFilters', 4, 'poolDim', 4, 'actiFunc', 'relu')
%                struct('type', 'Conv2D', 'filterDim', 5, 'numFilters', 8, 'poolDim', 2, 'actiFunc', 'relu')
%                struct('type', 'Conv2D', 'filterDim', 5, 'numFilters', 16, 'poolDim', 3, 'actiFunc', 'relu')
%                struct('type', 'Linear', 'hiddenUnits', 128, 'actiFunc', 'relu', 'dropout', 0.2)
%                struct('type', 'Linear', 'hiddenUnits', 64, 'actiFunc', 'relu')
%                struct('type', 'output', 'softmax', 1)
%                };
% cnn3.layers = {
%                struct('type', 'input') %input layer
%                struct('type', 'Conv2D', 'filterDim', 5, 'numFilters', 8, 'poolDim', 4, 'actiFunc', 'relu')
%                struct('type', 'Conv2D', 'filterDim', 5, 'numFilters', 16, 'poolDim', 2, 'actiFunc', 'relu')
%                struct('type', 'Conv2D', 'filterDim', 5, 'numFilters', 32, 'poolDim', 3, 'actiFunc', 'relu')
%                struct('type', 'Linear', 'hiddenUnits', 128, 'actiFunc', 'relu', 'dropout', 0.2)
%                struct('type', 'Linear', 'hiddenUnits', 64, 'actiFunc', 'relu')
%                struct('type', 'output', 'softmax', 1)
%                };
% cnns = {cnn1, cnn2, cnn3};
% dataset_options.img_dim = 124;
% [data_train, labels_train, data_test, labels_test] = loadDataset(data_path, dataset_options);

% for subexp = 1:3
%     fprintf("1.2.1 filter number subexp: %d\n", subexp)
%     cnn = cnns{subexp};
%     log_path = sprintf("../logs/exp1_2/model_%d/", subexp);

%     if ~exist(log_path, 'dir')
%         mkdir(log_path);
%     end

%     train_options.log_path = log_path;
%     trainMachine(cnn, dataset_options, train_options, data_train, labels_train, data_test, labels_test);
% end

% %% 1.2.2 filter size

% fprintf("1.2.2 filter size\n")
% % 122 -> conv3 -> 120 -> pool4 -> 30 -> conv3 -> 28 -> pool2 -> 14 ->  conv3 -> 12 -> pool2 -> 6
% cnn1.layers = {
%                struct('type', 'input') %input layer
%                struct('type', 'Conv2D', 'filterDim', 3, 'numFilters', 4, 'poolDim', 4, 'actiFunc', 'relu')
%                struct('type', 'Conv2D', 'filterDim', 3, 'numFilters', 8, 'poolDim', 2, 'actiFunc', 'relu')
%                struct('type', 'Conv2D', 'filterDim', 3, 'numFilters', 16, 'poolDim', 2, 'actiFunc', 'relu')
%                struct('type', 'Linear', 'hiddenUnits', 128, 'actiFunc', 'relu', 'dropout', 0.2)
%                struct('type', 'Linear', 'hiddenUnits', 64, 'actiFunc', 'relu')
%                struct('type', 'output', 'softmax', 1)
%                };
% % 124 -> conv5 -> 120 -> pool4 -> 30 -> conv3 -> 28 -> pool2 -> 14 ->  conv3 -> 12 -> pool2 -> 6
% cnn2.layers = {
%                struct('type', 'input') %input layer
%                struct('type', 'Conv2D', 'filterDim', 5, 'numFilters', 4, 'poolDim', 4, 'actiFunc', 'relu')
%                struct('type', 'Conv2D', 'filterDim', 3, 'numFilters', 8, 'poolDim', 2, 'actiFunc', 'relu')
%                struct('type', 'Conv2D', 'filterDim', 3, 'numFilters', 16, 'poolDim', 3, 'actiFunc', 'relu')
%                struct('type', 'Linear', 'hiddenUnits', 128, 'actiFunc', 'relu', 'dropout', 0.2)
%                struct('type', 'Linear', 'hiddenUnits', 64, 'actiFunc', 'relu')
%                struct('type', 'output', 'softmax', 1)
%                };
% % 124 -> conv5 -> 120 -> pool4 -> 30 -> conv5 -> 26 -> pool2 -> 13 ->  conv5 -> 9 -> pool3 -> 3
% cnn3.layers = {
%                struct('type', 'input') %input layer
%                struct('type', 'Conv2D', 'filterDim', 5, 'numFilters', 4, 'poolDim', 4, 'actiFunc', 'relu')
%                struct('type', 'Conv2D', 'filterDim', 5, 'numFilters', 8, 'poolDim', 2, 'actiFunc', 'relu')
%                struct('type', 'Conv2D', 'filterDim', 5, 'numFilters', 16, 'poolDim', 3, 'actiFunc', 'relu')
%                struct('type', 'Linear', 'hiddenUnits', 128, 'actiFunc', 'relu', 'dropout', 0.2)
%                struct('type', 'Linear', 'hiddenUnits', 64, 'actiFunc', 'relu')
%                struct('type', 'output', 'softmax', 1)
%                };
% cnns = {cnn1, cnn2, cnn3};

% for subexp = 1:3
%     fprintf("1.2.2 filter size subexp: %d\n", subexp)
%     cnn = cnns{subexp};

%     if subexp == 1
%         dataset_options.img_dim = 122;
%     else
%         dataset_options.img_dim = 124;
%     end

%     [data_train, labels_train, data_test, labels_test] = loadDataset(data_path, dataset_options);

%     log_path = sprintf("../logs/exp1_2/model_%d/", subexp);

%     if ~exist(log_path, 'dir')
%         mkdir(log_path);
%     end

%     train_options.log_path = log_path;
%     trainMachine(cnn, dataset_options, train_options, data_train, labels_train, data_test, labels_test);
% end

% %% 1.3 fully connected layer design
% fprintf("1.3 fully connected layer design\n")
% % -> 288
% cnn1.layers = {
%                struct('type', 'input') %input layer
%                struct('type', 'Conv2D', 'filterDim', 5, 'numFilters', 4, 'poolDim', 4, 'actiFunc', 'relu')
%                struct('type', 'Conv2D', 'filterDim', 5, 'numFilters', 8, 'poolDim', 2, 'actiFunc', 'relu')
%                struct('type', 'Conv2D', 'filterDim', 5, 'numFilters', 16, 'poolDim', 3, 'actiFunc', 'relu')
%                struct('type', 'Linear', 'hiddenUnits', 100, 'actiFunc', 'relu', 'dropout', 0.2)
%                struct('type', 'Linear', 'hiddenUnits', 50, 'actiFunc', 'relu')
%                struct('type', 'output', 'softmax', 1)
%                };
% cnn2.layers = {
%                struct('type', 'input') %input layer
%                struct('type', 'Conv2D', 'filterDim', 5, 'numFilters', 4, 'poolDim', 4, 'actiFunc', 'relu')
%                struct('type', 'Conv2D', 'filterDim', 5, 'numFilters', 8, 'poolDim', 2, 'actiFunc', 'relu')
%                struct('type', 'Conv2D', 'filterDim', 5, 'numFilters', 16, 'poolDim', 3, 'actiFunc', 'relu')
%                struct('type', 'Linear', 'hiddenUnits', 200, 'actiFunc', 'relu', 'dropout', 0.2)
%                struct('type', 'Linear', 'hiddenUnits', 100, 'actiFunc', 'relu')
%                struct('type', 'output', 'softmax', 1)
%                };
% cnn3.layers = {
%                struct('type', 'input') %input layer
%                struct('type', 'Conv2D', 'filterDim', 5, 'numFilters', 4, 'poolDim', 4, 'actiFunc', 'relu')
%                struct('type', 'Conv2D', 'filterDim', 5, 'numFilters', 8, 'poolDim', 2, 'actiFunc', 'relu')
%                struct('type', 'Conv2D', 'filterDim', 5, 'numFilters', 16, 'poolDim', 3, 'actiFunc', 'relu')
%                struct('type', 'Linear', 'hiddenUnits', 200, 'actiFunc', 'relu', 'dropout', 0.2)
%                struct('type', 'Linear', 'hiddenUnits', 50, 'actiFunc', 'relu')
%                struct('type', 'output', 'softmax', 1)
%                };
% cnns = {cnn1, cnn2, cnn3};
% dataset_options.img_dim = 124;
% [data_train, labels_train, data_test, labels_test] = loadDataset(data_path, dataset_options);

% for subexp = 1:3
%     fprintf("1.3 fully connected layer design subexp: %d\n", subexp)
%     cnn = cnns{subexp};

%     log_path = sprintf("../logs/exp1_3/model_%d/", subexp);

%     if ~exist(log_path, 'dir')
%         mkdir(log_path);
%     end

%     train_options.log_path = log_path;
%     trainMachine(cnn, dataset_options, train_options, data_train, labels_train, data_test, labels_test);
% end

%% 2 hyperparameter tuning

%% 2.1 learning rate

%% 2.1.1 lr schedule scheme
fprintf("2.1.1 lr schedule scheme\n")

cnn.layers = {
              struct('type', 'input') %input layer
              struct('type', 'Conv2D', 'filterDim', 5, 'numFilters', 4, 'poolDim', 4, 'actiFunc', 'relu')
              struct('type', 'Conv2D', 'filterDim', 5, 'numFilters', 8, 'poolDim', 2, 'actiFunc', 'relu')
              struct('type', 'Conv2D', 'filterDim', 5, 'numFilters', 16, 'poolDim', 3, 'actiFunc', 'relu')
              struct('type', 'Linear', 'hiddenUnits', 100, 'actiFunc', 'relu', 'dropout', 0.2)
              struct('type', 'Linear', 'hiddenUnits', 50, 'actiFunc', 'relu')
              struct('type', 'output', 'softmax', 1)
              };
lr_sheculer = {'fixed', 'cosine', 'cosine_cyclic', "linear"};

for subexp = 1:3
    fprintf("2.1.1 lr schedule scheme subexp: %d\n", subexp)
    train_options.lr_method = lr_sheculer{subexp};
    log_path = sprintf("../logs/exp2_1_1/lr_method_%d/", subexp);

    if ~exist(log_path, 'dir')
        mkdir(log_path);
    end

    train_options.log_path = log_path;
    trainMachine(cnn, dataset_options, train_options, data_train, labels_train, data_test, labels_test);
end

%% 2.1.2 lr magnitude
fprintf("2.1.2 lr magnitude\n")

lr_max = [0.1, 0.01, 0.001];
train_options.lr_method = 'cosine_cyclic';

for subexp = 1:3
    fprintf("2.1.2 lr magnitude subexp: %d\n", subexp)
    train_options.lr_max = lr_max(subexp);
    train_options.lr = train_options.lr_max;
    log_path = sprintf("../logs/exp2_1_2/lr_max_%d/", subexp);

    if ~exist(log_path, 'dir')
        mkdir(log_path);
    end

    train_options.log_path = log_path;
    trainMachine(cnn, dataset_options, train_options, data_train, labels_train, data_test, labels_test);
end

%% 2.2 batch size

batch_size = [16, 32, 64, 128];
train_options.lr_max = 0.01;
train_options.lr = train_options.lr_max;

for subexp = 1:4
    train_options.minibatch = batch_size(subexp);
    train_options.total_iter = round(floor(size(data_train, 4) / train_options.minibatch) * train_options.epochs);
    options.lr_method = 'cosine_cyclic';
    log_path = sprintf("../logs/exp2_2/batch_size_%d/", train_options.minibatch);

    if ~exist(log_path, 'dir')
        mkdir(log_path);
    end

    train_options.log_path = log_path;
    trainMachine(cnn, dataset_options, train_options, data_train, labels_train, data_test, labels_test);
end

%% 3 data preprocessing

%% 3.1 resize

%% 3.2 random transform
fprintf("3.2 random transform\n")

tf = [
      "only_trans"
      "only_rot"
      "only_scale"
      "rot&trans"
      "rot&scale"
      "all"
      ];
train_options.lr_max = 0.01;
train_options.lr = train_options.lr_max;
train_options.minibatch = 64;
train_options.total_iter = round(floor(size(data_train, 4) / train_options.minibatch) * train_options.epochs);

dataset_option.img_dim = 124;
dataset_option.apply_rand_tf = true;
random_trans.prob = 0.5;
random_trans.trans_ratio = 0.1;
random_trans.rot_range = [-25 25];
random_trans.scale_ratio = [0.8 1.2];

for subexp = 1:6
    fprintf("3.2 random transform subexp: %d\n", subexp)
    tf_type = tf{subexp};

    switch tf_type
        case "only_trans"
            random_trans.rot_range = [0 0];
            random_trans.scale_ratio = [1 1];
            random_trans.trans_ratio = 0.1;
        case "only_rot"
            random_trans.rot_range = [-25 25];
            random_trans.scale_ratio = [1 1];
            random_trans.trans_ratio = 0;
        case "only_scale"
            random_trans.rot_range = [0 0];
            random_trans.scale_ratio = [0.8 1.2];
            random_trans.trans_ratio = 0;
        case "rot&trans"
            random_trans.rot_range = [-25 25];
            random_trans.scale_ratio = [1 1];
            random_trans.trans_ratio = 0.1;
        case "rot&scale"
            random_trans.rot_range = [-25 25];
            random_trans.scale_ratio = [0.8 1.2];
            random_trans.trans_ratio = 0;
        case "all"
            random_trans.rot_range = [-25 25];
            random_trans.scale_ratio = [0.8 1.2];
            random_trans.trans_ratio = 0.1;
    end

    dataset_options.rand_tf = random_trans;
    [data_train, labels_train, data_test, labels_test] = loadDataset(data_path, dataset_options);
    log_path = sprintf("../logs/exp3_2/rand_tf_%d/", subexp);

    if ~exist(log_path, 'dir')
        mkdir(log_path);
    end

    train_options.log_path = log_path;
    trainMachine(cnn, dataset_options, train_options, data_train, labels_train, data_test, labels_test);
end
