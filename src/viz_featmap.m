close all;
clear all;

% visualize features extracted by the network
addpath("cnn_core");

% find the latest log
% log_path = "../logs/";
% logs = dir(log_path);
% logs = logs(~ismember({logs.name}, {'.', '..'}));
% [~, idx] = max([logs.datenum]);
% log_path = log_path + logs(idx).name + "/";
% disp("Loading log from " + log_path);

log_path = "../logs/11-19_15-56-26/";

load(log_path + "cnn_best_acc.mat");
load("../data/test.mat")

% analyze the fearture map of miss classified images
% find the index of miss classified images in the test set
[preds, cnn] = predict(cnn, data_test);
select_mask = (preds ~= labels_test) & (preds == 7);
miss_idx = find(select_mask);

if 1 % viz the featmap of label 7
    h_idx = find(labels_test == 7);
    % average feature map of label 7
    ave_featmap_conv1 = zeros(120, 120, 8);
    ave_featmap_conv2 = zeros(26, 26, 16);
    ave_featmap_conv3 = zeros(9, 9, 32);

    for m = 1:size(h_idx)
        idx = h_idx(m);

        for layer = 2:4
            fig = figure(layer);
            set(fig, "visible", "off")
            ax = axes(fig);
            filter_num = cnn.layers{layer}.numFilters;
            rows = ceil(sqrt(filter_num));

            for i = 1:filter_num
                subplot(rows, rows, i);
                featmap = cnn.layers{layer}.convolvedFeatures(:, :, i, idx);

                switch layer
                    case 2
                        ave_featmap_conv1(:, :, i) = ave_featmap_conv1(:, :, i) + featmap;
                    case 3
                        ave_featmap_conv2(:, :, i) = ave_featmap_conv2(:, :, i) + featmap;
                    case 4
                        ave_featmap_conv3(:, :, i) = ave_featmap_conv3(:, :, i) + featmap;
                end

                imshow(featmap);
            end

            folder = sprintf("../temp/featmap_H_%d/", idx);

            if ~exist(folder, 'dir')
                mkdir(folder);
            end

            file_path = folder + "layer_"+layer + ".png";
            saveas(gcf, file_path);

        end

    end

    folder = "../temp/";
    ave_featmap = {ave_featmap_conv1, ave_featmap_conv2, ave_featmap_conv3};

    for m = 1:3
        ave_feat = ave_featmap{m};
        ave_feat(:, :, :) = ave_feat(:, :, :) / size(h_idx, 1);
        fig = figure(m);
        set(fig, "visible", "off")
        filter_num = size(ave_feat, 3);
        rows = ceil(sqrt(filter_num));
        % draw featmap of all the filters for a layer in a single figure
        for i = 1:filter_num
            subplot(rows, rows, i);
            imshow(ave_feat(:, :, i));
        end

        file_name = sprintf("ave_featmap_conv%d.png", m);
        saveas(gcf, folder + file_name);
    end

end

if 0 % viz the feature map of miss classified images
    % miss_idx = miss_idx(1:10);

    for m = 1:size(miss_idx)
        idx = miss_idx(m);
        img = data_test(:, :, :, idx);
        img_dim = size(img, 1);
        img = reshape(img, img_dim, img_dim, 1, []);
        layer_num = size(cnn.layers, 1);

        for layer = 2:4
            fig = figure(layer);
            set(fig, "visible", "off")
            ax = axes(fig);
            filter_num = cnn.layers{layer}.numFilters;
            rows = ceil(sqrt(filter_num));

            for i = 1:filter_num
                subplot(rows, rows, i);
                featmap = cnn.layers{layer}.convolvedFeatures(:, :, i, idx);
                imshow(featmap);
            end

            folder = sprintf("../temp/featmap_%d_G%d_P%d/", idx, ...
                labels_test(idx), preds(idx));

            if ~exist(folder, 'dir')
                mkdir(folder);
            end

            file_path = folder + "layer_"+layer + ".png";
            saveas(gcf, file_path);

        end

    end

end

if 0 % viz the feature map of a given image
    % choose a random image
    idx = 100;
    % idx = miss_idx(1:10);
    img = data_test(:, :, :, idx);
    img_dim = size(img, 1);
    img = reshape(img, img_dim, img_dim, 1, []);
    layer_num = size(cnn.layers, 1);

    for layer = 2:4
        figure(layer);
        filter_num = cnn.layers{layer}.numFilters;
        rows = ceil(sqrt(filter_num));

        for i = 1:filter_num
            subplot(rows, rows, i);
            featmap = cnn.layers{layer}.convolvedFeatures(:, :, i);
            imshow(featmap);
        end

    end

end
