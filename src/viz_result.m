addpath("utils")
log_path = "../logs/11-19_11-39-13/";

show_figures = false;

% load data
load(log_path + 'acc_test.mat');
load(log_path + 'acc_train.mat');
load(log_path + 'loss_ar.mat');
load(log_path + 'lr_ar.mat');

% plot
figure(1);
plot(acc_test, 'LineWidth', 2);
hold on;
plot(acc_train, 'LineWidth', 2);
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
load(log_path + 'results_on_test.mat');
confusionchart(confusionmat(preds, labels_test));
title('Confusion Matrix');
saveas(gcf, log_path + 'confusion_matrix.png');
