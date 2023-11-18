clear all; %#ok<CLALL>

addpath("core");

date_prefix = string(datetime('now', 'Format', 'MM-dd_HH-mm-ss'));
log_path = "logs/"+date_prefix + "/";
mkdir(log_path);

DEBUG = false;
% global images_train labels_train images_test labels_test;
load_from_file = false;

if ~load_from_file
    dataset_option.shuffle = true;
    dataset_option.img_dim = 128;
    dataset_option.train_ratio = 0.8;
    dataset_option.save = true;
    [images_train, labels_train, images_test, labels_test] = loadDataset(dataset_option);
else
    load("data/train.mat");
    load("data/test.mat");
end

cnn.layers = {
              struct('type', 'i') %input layer
              struct('type', 'cs', 'filterDim', 5, 'numFilters', 6, 'poolDim', 2, 'activationFunction', 'relu')
              struct('type', 'cs', 'filterDim', 5, 'numFilters', 16, 'poolDim', 2, 'activationFunction', 'relu')
              struct('type', 'cs', 'filterDim', 3, 'numFilters', 20, 'poolDim', 1, 'activationFunction', 'relu')
              struct('type', 'fc', 'hiddenUnits', 120, 'activationFunction', 'tanh')
              struct('type', 'fc', 'hiddenUnits', 84, 'activationFunction', 'tanh')
              struct('type', 'o', 'softmax', 1)
              };

options.epochs = 30;
options.minibatch = 64;
options.lr_max = 0.01;
options.lr = options.lr_max;
options.lr_min = 1e-4;
options.lr_method = 'cosine';
options.momentum = 0.9;
options.log_path = log_path;
options.l2_penalty = 0.1;
total_iter = round(floor(size(images_train, 4) / options.minibatch) * options.epochs);
options.total_iter = total_iter;

fprintf('Total iterations: %d\n', total_iter);
cnn = learn(cnn, images_train, labels_train, images_test, labels_test, options);

preds = predict(cnn, images_test);
acc = sum(preds == labels_test) / length(preds);
fprintf('Final accuracy: %f\n', acc);

% save model and parameters
save(log_path + "cnn.mat", 'cnn');
save(log_path + "hyper_params.mat", 'dataset_option', 'options');
