close all; % show the images that are miss classified
clear all;
addpath("cnn_core")

log_path = "E:\ProgramDev\NUS-ME5411\logs\best\lr_max_1\";
% load(log_path + "/results_on_test.mat")
load(log_path + "/cnn_best_acc.mat")
% load("../data/test.mat") % data_test, labels_test
data_path = "../data/";
dataset_options.load_raw = true;
dataset_options.shuffle = true;
dataset_options.img_dim = 124;
dataset_options.train_ratio = 0.75;
dataset_options.save = false;
dataset_options.apply_rand_tf = false;
[data_train, labels_train, data_test, labels_test] = loadDataset(data_path, dataset_options);

[preds, ~] = predict(cnn, data_test);
confu_mat = confusionmat(labels_test, preds);
confusionchart(confu_mat);

mis_classified_idx = [];

cnt = 0;

for i = 1:length(preds)

    if preds(i) ~= labels_test(i)
        mis_classified_idx = [mis_classified_idx i];

    end

end

fprintf("misclassified 5: %d\n", cnt)

for i = 1:size(mis_classified_idx,2)
    % for i = 1:100
    % idx = i;
    % mis_idx = randi(length(mis_classified_idx));
    idx = mis_classified_idx(i);
    fig = figure(idx);
    ax = axes(fig);
    imshow(data_test(:, :, :, idx), "Parent", ax);
    title(ax, "Predicted Class: " + labelNum2Char(preds(idx)) );%+ " True Class: " + labelNum2Char(labels_test(idx))+" idx: " + num2str(idx))
    imwrite(data_test(:, :, :, idx), ...
        sprintf("../temp/mis_classified/T_%c_P_%c.png", labelNum2Char(labels_test(idx)), labelNum2Char(preds(idx))))
end
