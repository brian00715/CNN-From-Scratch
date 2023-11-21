clear all;
close all;
addpath("utils");
set(gca, 'Visible', 'off');

% find the latest log
% log_path = "../logs/";
% logs = dir(log_path);
% logs = logs(~ismember({logs.name}, {'.', '..'}));
% [~, idx] = max([logs.datenum]);
% log_path = log_path + logs(idx).name + "/";
% disp("Loading log from " + log_path);

log_path = "E:/ProgramDev/NUS-ME5411\logs\best\lr_max_1/";

show_figures = false;

% load data
load(log_path + 'acc_test.mat');
load(log_path + 'acc_train.mat');
load(log_path + 'loss_ar.mat');
load(log_path + 'lr_ar.mat');

% plot
figure(1);
x = 1:5;
plot(x,acc_test, 'LineWidth', 2);
hold on;
plot(x,acc_train, 'LineWidth', 2);
hold off;
grid on;
xlabel('Epoch');
ylabel('Accuracy');
legend('Test', 'Train');
title('Accuracy');
saveas(gcf, log_path + 'acc.png');

figure(2);
plot(loss_ar, 'LineWidth', 2);
grid on;
xlabel('Epoch');
ylabel('Loss');
title('Loss');
saveas(gcf, log_path + 'loss.png');

figure(3);
plot(lr_ar, 'LineWidth', 2);
grid on;
xlabel('Epoch');
ylabel('Learning Rate');
title('Learning Rate');
saveas(gcf, log_path + 'lr.png');

% draw confusion matrix
figure(4);
% load(log_path + 'results_on_testset.mat');
load("../data/test.mat") % data_test, labels_test
load(log_path + "/cnn_best_acc.mat");
[preds, ~] = predict(cnn, data_test);
miss_detect_cnt = zeros(7,1);
miss_detect_idx = [];
for i=1:size(preds,1)
    if preds(i)~=labels_test(i)
        miss_detect_idx = [miss_detect_idx;i];
        miss_detect_cnt(labels_test(i)) = miss_detect_cnt(labels_test(i))+1;
    end
end
confu_mat = confusionmat(labels_test, preds);
confusionchart(confu_mat);
title('Confusion Matrix');
saveas(gcf, log_path + 'confusion_matrix.png');
