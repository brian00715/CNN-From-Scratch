close all;
clear all;
addpath("utils");
addpath("cnn_core");
dataset_options.load_raw = false;
dataset_options.shuffle = true;
dataset_options.img_dim = 124;
dataset_options.train_ratio = 0.8;
dataset_options.save = false;
dataset_options.apply_rand_tf = false;

train_options.epochs = 2;
train_options.minibatch = 8;
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

if 1
    [data_train, labels_train, data_test, labels_test] = loadDataset("../data/", dataset_options);
    total_iter = round(floor(size(data_train, 4) / train_options.minibatch) * train_options.epochs);
    train_options.total_iter = total_iter;

    cnn.layers = {
                  struct('type', 'input') %input layer
                  struct('type', 'Conv2D', 'filterDim', 5, 'numFilters', 4, 'poolDim', 4, 'actiFunc', 'relu')
                  struct('type', 'Conv2D', 'filterDim', 5, 'numFilters', 8, 'poolDim', 2, 'actiFunc', 'relu')
                  struct('type', 'Conv2D', 'filterDim', 5, 'numFilters', 16, 'poolDim', 3, 'actiFunc', 'relu')
                  struct('type', 'Linear', 'hiddenUnits', 100, 'actiFunc', 'relu', 'dropout', 0.2)
                  struct('type', 'Linear', 'hiddenUnits', 50, 'actiFunc', 'relu')
                  struct('type', 'output', 'softmax', 1)
                  };

    log_path = sprintf("../temp/");

    if ~exist(log_path, 'dir')
        mkdir(log_path);
    end

    train_options.log_path = log_path;
    trainMachine(cnn, dataset_options, train_options, data_train, labels_train, data_test, labels_test);

end

if 0

    [data_train, labels_train, data_test, labels_test] = loadDataset("../data/", dataset_options);
    log_path = "../logs/exp2_1_1/lr_method_1/";
    load(log_path + "acc_test.mat");
    load(log_path + "acc_train.mat");
    load(log_path + "lr_ar.mat");
    load(log_path + "loss_ar.mat");
    load(log_path + "results_on_testset.mat");
    vizExpLog(log_path, acc_test, acc_train, lr_ar, loss_ar, preds, labels_test)
end

if 0
    load("/home/simon/ProgramDev/me5411/logs/best/11-19_20-59-04/cnn.mat") % cnn
    model_json = model2json(cnn);
    fid = fopen("../temp/model.json", "w");
    fprintf(fid, model_json);
    fclose(fid);
end

if 0
    total_epoch = 40;
    lr = [];
    options.lr_max = 0.01;
    options.lr = options.lr_max;
    options.lr_min = 1e-7;
    options.lr_method = 'exp';
    options.lr_duty = 5; % duty cycle for cosine lr

    for curr_epoch = 1:total_epoch
        curr_lr = lrSchedule(curr_epoch, total_epoch, options);
        lr = [lr curr_lr];
    end

    figure;
    plot(lr);
    xlabel('epoch');
    ylabel('learning rate');
    title('lrSchedule test');
end

if 0
    data_path = "../data/p_dataset_26/";
    dataset_option.shuffle = true;
    dataset_option.img_dim = 124;
    dataset_option.train_ratio = 0.8;
    dataset_option.save = true;
    dataset_option.transform = true;
    [images_train, labels_train, images_test, labels_test] = loadDataset(data_path, dataset_option);

    run ("viz_result.m")
end
