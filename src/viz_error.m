close all; % show the images that are miss classified
clear all;
addpath("cnn_core")
log_path = "../logs/11-19_15-56-26/";
% load(log_path + "/results_on_test.mat")
load(log_path + "/cnn_best_acc.mat")
load("../data/test.mat") % data_test, labels_test

[preds,~] = predict(cnn, data_test);

cnt = 0;
h_idx = [];

for i = 1:length(preds)

    if preds(i) ~= labels_test(i)

        if preds(i) == 7
            fig = figure(i);
            ax = axes(fig);
            h_idx = [h_idx; i];
            imshow(data_test(:, :, :, i), "Parent", ax);
            title(ax, "Predicted: " + labelNum2Char(preds(i)) + " Actual: " + labelNum2Char(labels_test(i)) + ...
                " idx: " + num2str(i))
            cnt = cnt + 1;
            % if cnt > 10
            %     break
            % end
        end

    end

end
