close all;
clear all;
addpath("utils");
addpath("cnn_core");

if 1
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
