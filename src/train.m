clear all; %#ok<CLALL>
addpath("core");

date_prefix = string(datetime('now', 'Format', 'MM-dd_HH-mm-ss'));
log_path = "../logs/"+date_prefix + "/";
mkdir(log_path);

%% load dataset
load_from_file = false;
% data_path = "../data/p_dataset_26/";
data_path = "../data/mnist/";

if ~load_from_file
    dataset_option.shuffle = true;
    dataset_option.img_dim = 28;
    dataset_option.train_ratio = 0.8;
    dataset_option.save = true;
    dataset_option.transform = true;
    % [images_train, labels_train, images_test, labels_test] = loadDataset(data_path, dataset_option);

    imageDim3 = 1; % Dim3 is mandatory
    images_train = loadMNISTImages(data_path + 'train-images-idx3-ubyte');
    images_train = reshape(images_train, 28, 28, 1, []);
    labels_train = loadMNISTLabels(data_path + 'train-labels-idx1-ubyte');
    labels_train(labels_train == 0) = 10; % Remap 0 to 10
    images_test = loadMNISTImages(data_path + 't10k-images-idx3-ubyte');
    images_test = reshape(images_test, 28, 28, 1, []);
    labels_test = loadMNISTLabels(data_path + 't10k-labels-idx1-ubyte');
    labels_test(labels_test == 0) = 10; % Remap 0 to 10
else
    load("../data/train.mat");
    load("../data/test.mat");
end

%% define network structure
cnn.layers = {
              struct('type', 'i') %input layer
              struct('type', 'Conv2D', 'filterDim', 5, 'numFilters', 6, 'poolDim', 2, 'actiFunc', 'relu')
              struct('type', 'Conv2D', 'filterDim', 5, 'numFilters', 16, 'poolDim', 2, 'actiFunc', 'relu')
%   struct('type', 'Conv2D', 'filterDim', 3, 'numFilters', 32, 'poolDim', 2, 'actiFunc', 'relu')
              struct('type', 'Linear', 'hiddenUnits', 120, 'actiFunc', 'tanh')
              struct('type', 'Linear', 'hiddenUnits', 84, 'actiFunc', 'tanh')
              struct('type', 'o', 'softmax', 1)
              };

%% define training options
options.epochs = 30;
options.minibatch = 256;
options.lr_max = 0.1;
options.lr = options.lr_max;
options.lr_min = 1e-4;
options.lr_method = 'cosine';
options.momentum = 0.9;
options.log_path = log_path;
options.l2_penalty = 0.01;
options.use_l2 = false;
total_iter = round(floor(size(images_train, 4) / options.minibatch) * options.epochs);
options.total_iter = total_iter;
fprintf('Total iterations: %d\n', total_iter);

%% train
cnn = learn(cnn, images_train, labels_train, images_test, labels_test, options);

%% test
preds = predict(cnn, images_test);
acc = sum(preds == labels_test) / length(preds);
fprintf('Final accuracy: %f\n', acc);

%% save model and parameters
fileID = fopen(log_path + "results.txt", 'w');
fprintf(fileID, 'Final accuracy: %f\n', acc);
save(log_path + "cnn.mat", 'cnn');
save(log_path + "results_on_test.mat", 'preds', 'labels_test');
save(log_path + "hyper_params.mat", 'dataset_option', 'options');
