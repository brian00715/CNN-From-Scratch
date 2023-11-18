log_path = "logs/11-18_18-15-07/";

% load data
load(log_path + 'acc_test.mat');
load(log_path + 'acc_train.mat');
load(log_path + 'loss.mat');
load(log_path + 'lr.mat');

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

figure(2);
plot(loss, 'LineWidth', 2);
grid on;
xlabel('Epoch');
ylabel('Loss');
title('Loss');

figure(3);
plot(lr, 'LineWidth', 2);
grid on;
xlabel('Epoch');
ylabel('Learning Rate');
title('Learning Rate');
