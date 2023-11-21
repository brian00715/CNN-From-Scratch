% plot data of differnet batch size
% data: acc_test10, acc_train10, acc_test20, acc_train20,
% acc_test40, acc_train40,acc_test60, acc_train60, acc_test80, acc_train80
close all;
clear all;
colors = ["#0072BD", "#D95319", "#EDB120", "#7E2F8E", "#77AC30", "#4DBEEE", "#A2142F", "#000000", "#000000", "#000000"];

if 0 % learning rate
    data_path = "../logs\2nd_cloud\exp2_1_2/";
    lr1_acc_test = load(data_path + "/lr_max_1/acc_test.mat");
    lr1_acc_test = lr1_acc_test.acc_test;
    lr1_acc_train = load(data_path + "/lr_max_1/acc_train.mat");
    lr1_acc_train = lr1_acc_train.acc_train;
    lr2_acc_test = load(data_path + "/lr_max_2/acc_test.mat");
    lr2_acc_test = lr2_acc_test.acc_test;
    lr2_acc_train = load(data_path + "/lr_max_2/acc_train.mat");
    lr2_acc_train = lr2_acc_train.acc_train;
    lr3_acc_test = load(data_path + "/lr_max_3/acc_test.mat");
    lr3_acc_test = lr3_acc_test.acc_test;
    lr3_acc_train = load(data_path + "/lr_max_3/acc_train.mat");
    lr3_acc_train = lr3_acc_train.acc_train;
    lr4_acc_test = load(data_path + "/lr_max_4/acc_test.mat");
    lr4_acc_test = lr4_acc_test.acc_test;
    lr4_acc_train = load(data_path + "/lr_max_4/acc_train.mat");
    lr4_acc_train = lr4_acc_train.acc_train;

    lr1_loss = load(data_path + "/lr_max_1/loss_ar.mat");
    lr1_loss = lr1_loss.loss_ar;
    lr2_loss = load(data_path + "/lr_max_2/loss_ar.mat");
    lr2_loss = lr2_loss.loss_ar;
    lr3_loss = load(data_path + "/lr_max_3/loss_ar.mat");
    lr3_loss = lr3_loss.loss_ar;
    lr4_loss = load(data_path + "/lr_max_4/loss_ar.mat");
    lr4_loss = lr4_loss.loss_ar;

    figure(1)
    plot(lr1_acc_test, 'color', colors(1), 'LineWidth', 1, 'LineStyle', '-'); hold on;
    plot(lr1_acc_train, 'color', colors(1), 'LineWidth', 1, 'LineStyle', '--'); hold on;
    plot(lr2_acc_test, 'color', colors(2), 'LineWidth', 1, 'LineStyle', '-'); hold on;
    plot(lr2_acc_train, 'color', colors(2), 'LineWidth', 1, 'LineStyle', '--'); hold on;
    plot(lr3_acc_test, 'color', colors(3), 'LineWidth', 1, 'LineStyle', '-'); hold on;
    plot(lr3_acc_train, 'color', colors(3), 'LineWidth', 1, 'LineStyle', '--'); hold on;
    plot(lr4_acc_test, 'color', colors(4), 'LineWidth', 1, 'LineStyle', '-'); hold on;
    plot(lr4_acc_train, 'color', colors(4), 'LineWidth', 1, 'LineStyle', '--'); hold on;
    legend('Test-LR_m=0.5', 'Train-LR_m=0.5', 'Test-LR_m=0.1', 'Train-LR_m=0.1', 'Test-LR_m=0.01', 'Train-LR_m=0.01', 'Test-LR_m=0.001', 'Train-LR_m=0.001');
    xlabel('Epoch');
    ylabel('Accuracy');

    figure(2)
    plot(lr1_loss, 'color', colors(1), 'LineWidth', 1, 'LineStyle', '-'); hold on;
    plot(lr2_loss, 'color', colors(2), 'LineWidth', 1, 'LineStyle', '-'); hold on;
    plot(lr3_loss, 'color', colors(3), 'LineWidth', 1, 'LineStyle', '-'); hold on;
    plot(lr4_loss, 'color', colors(4), 'LineWidth', 1, 'LineStyle', '-'); hold on;
    legend('LR_m=0.5', 'LR_m=0.1', 'LR_m=0.01', 'LR_m=0.001');
    xlabel('Iteration');
    ylabel('Loss');
end

if 0 % learning rate chedule scheme
    data_path = "../logs\2nd_cloud\exp2_1_1/";
    lr1_acc_test = load(data_path + "/lr_method_1/acc_test.mat");
    lr1_acc_test = lr1_acc_test.acc_test;
    lr1_acc_train = load(data_path + "/lr_method_1/acc_train.mat");
    lr1_acc_train = lr1_acc_train.acc_train;
    lr2_acc_test = load(data_path + "/lr_method_2/acc_test.mat");
    lr2_acc_test = lr2_acc_test.acc_test;
    lr2_acc_train = load(data_path + "/lr_method_2/acc_train.mat");
    lr2_acc_train = lr2_acc_train.acc_train;
    lr3_acc_test = load(data_path + "/lr_method_3/acc_test.mat");
    lr3_acc_test = lr3_acc_test.acc_test;
    lr3_acc_train = load(data_path + "/lr_method_3/acc_train.mat");
    lr3_acc_train = lr3_acc_train.acc_train;
    lr4_acc_test = load(data_path + "/lr_method_4/acc_test.mat");
    lr4_acc_test = lr4_acc_test.acc_test;
    lr4_acc_train = load(data_path + "/lr_method_4/acc_train.mat");
    lr4_acc_train = lr4_acc_train.acc_train;

    lr1_loss = load(data_path + "/lr_method_1/loss_ar.mat");
    lr1_loss = lr1_loss.loss_ar;
    lr2_loss = load(data_path + "/lr_method_2/loss_ar.mat");
    lr2_loss = lr2_loss.loss_ar;
    lr3_loss = load(data_path + "/lr_method_3/loss_ar.mat");
    lr3_loss = lr3_loss.loss_ar;
    lr4_loss = load(data_path + "/lr_method_4/loss_ar.mat");
    lr4_loss = lr4_loss.loss_ar;

    figure(1)
    plot(lr1_acc_test, 'color', colors(1), 'LineWidth', 1, 'LineStyle', '-'); hold on;
    plot(lr1_acc_train, 'color', colors(1), 'LineWidth', 1, 'LineStyle', '--'); hold on;
    plot(lr2_acc_test, 'color', colors(2), 'LineWidth', 1, 'LineStyle', '-'); hold on;
    plot(lr2_acc_train, 'color', colors(2), 'LineWidth', 1, 'LineStyle', '--'); hold on;
    plot(lr3_acc_test, 'color', colors(3), 'LineWidth', 1, 'LineStyle', '-'); hold on;
    plot(lr3_acc_train, 'color', colors(3), 'LineWidth', 1, 'LineStyle', '--'); hold on;
    plot(lr4_acc_test, 'color', colors(4), 'LineWidth', 1, 'LineStyle', '-'); hold on;
    plot(lr4_acc_train, 'color', colors(4), 'LineWidth', 1, 'LineStyle', '--'); hold on;

    legend('Test-Fixed', 'Train-Fixed', 'Test-Cosine', 'Train-Cosine', 'Test-Cyclic Cosine', 'Train-Cyclic Cosine', 'Test-Linear', 'Train-Linear');
    xlabel('Epoch');
    ylabel('Accuracy');

    figure(2)
    lr1_smooth = smoothdata(lr1_loss, 'gaussian', 20);
    plot(lr1_smooth, 'color', colors(1), 'LineWidth', 2, 'LineStyle', '-'); hold on;
    lr2_smooth = smoothdata(lr2_loss, 'gaussian', 20);
    plot(lr2_smooth, 'color', colors(2), 'LineWidth', 2, 'LineStyle', '-'); hold on;
    lr3_smooth = smoothdata(lr3_loss, 'gaussian', 20);
    plot(lr3_smooth, 'color', colors(3), 'LineWidth', 2, 'LineStyle', '-'); hold on;
    lr4_smooth = smoothdata(lr4_loss, 'gaussian', 20);
    plot(lr4_smooth, 'color', colors(4), 'LineWidth', 2, 'LineStyle', '-'); hold on;
    h = plot(lr1_loss, 'color', colors(1), 'LineWidth', 1, 'LineStyle', '-'); h.Color(4) = 0.2; hold on;
    h = plot(lr2_loss, 'color', colors(2), 'LineWidth', 1, 'LineStyle', '-'); h.Color(4) = 0.2; hold on;
    h = plot(lr3_loss, 'color', colors(3), 'LineWidth', 1, 'LineStyle', '-'); h.Color(4) = 0.2; hold on;
    h = plot(lr4_loss, 'color', colors(4), 'LineWidth', 1, 'LineStyle', '-'); h.Color(4) = 0.2; hold on;
    legend('Fixed', 'Cosine', 'Cyclic Cosine', 'Linear');
    xlabel('Iteration');
    ylabel('Loss');

end

if 1 % batch size
    data_path = "../logs\2nd_cloud\exp2_2/";
    bt1_acc_test = load(data_path + "/batch_size_16/acc_test.mat");
    bt1_acc_test = bt1_acc_test.acc_test(1:60);
    bt1_acc_train = load(data_path + "/batch_size_16/acc_train.mat");
    bt1_acc_train = bt1_acc_train.acc_train(1:60);
    bt2_acc_test = load(data_path + "/batch_size_32/acc_test.mat");
    bt2_acc_test = bt2_acc_test.acc_test(1:60);
    bt2_acc_train = load(data_path + "/batch_size_32/acc_train.mat");
    bt2_acc_train = bt2_acc_train.acc_train(1:60);
    bt3_acc_test = load(data_path + "/batch_size_64/acc_test.mat");
    bt3_acc_test = bt3_acc_test.acc_test(1:60);
    bt3_acc_train = load(data_path + "/batch_size_64/acc_train.mat");
    bt3_acc_train = bt3_acc_train.acc_train(1:60);
    bt4_acc_test = load(data_path + "/batch_size_128/acc_test.mat");
    bt4_acc_test = bt4_acc_test.acc_test(1:60);
    bt4_acc_train = load(data_path + "/batch_size_128/acc_train.mat");
    bt4_acc_train = bt4_acc_train.acc_train(1:60);

    bt1_loss = load(data_path + "/batch_size_16/loss_ar.mat");
    bt2_loss = load(data_path + "/batch_size_32/loss_ar.mat");
    bt3_loss = load(data_path + "/batch_size_64/loss_ar.mat");
    bt4_loss = load(data_path + "/batch_size_128/loss_ar.mat");
    iter_num = floor(size(bt4_loss.loss_ar, 1) );

    bt1_loss = bt1_loss.loss_ar(1:iter_num);
    bt2_loss = bt2_loss.loss_ar(1:iter_num);
    bt3_loss = bt3_loss.loss_ar(1:iter_num);
    bt4_loss = bt4_loss.loss_ar(1:iter_num);

    figure(1)
    plot(bt1_acc_test, 'color', colors(1), 'LineWidth', 1, 'LineStyle', '-'); hold on;
    plot(bt1_acc_train, 'color', colors(1), 'LineWidth', 1, 'LineStyle', '--'); hold on;
    plot(bt2_acc_test, 'color', colors(2), 'LineWidth', 1, 'LineStyle', '-'); hold on;
    plot(bt2_acc_train, 'color', colors(2), 'LineWidth', 1, 'LineStyle', '--'); hold on;
    plot(bt3_acc_test, 'color', colors(3), 'LineWidth', 1, 'LineStyle', '-'); hold on;
    plot(bt3_acc_train, 'color', colors(3), 'LineWidth', 1, 'LineStyle', '--'); hold on;
    plot(bt4_acc_test, 'color', colors(4), 'LineWidth', 1, 'LineStyle', '-'); hold on;
    plot(bt4_acc_train, 'color', colors(4), 'LineWidth', 1, 'LineStyle', '--'); hold on;

    legend('Test-BatchSize=16', 'Train-BatchSize=16', 'Test-BatchSize=32', 'Train-BatchSize=32', 'Test-BatchSize=64', 'Train-BatchSize=64', 'Test-BatchSize=128', 'Train-BatchSize=128');
    xlabel('Epoch');
    ylabel('Accuracy');

    figure(2)
    bt1_smooth = smoothdata(bt1_loss, 'gaussian', 20);
    plot(bt1_smooth, 'color', colors(1), 'LineWidth', 2, 'LineStyle', '-'); hold on;
    bt2_smooth = smoothdata(bt2_loss, 'gaussian', 20);
    plot(bt2_smooth, 'color', colors(2), 'LineWidth', 2, 'LineStyle', '-'); hold on;
    bt3_smooth = smoothdata(bt3_loss, 'gaussian', 20);
    plot(bt3_smooth, 'color', colors(3), 'LineWidth', 2, 'LineStyle', '-'); hold on;
    bt4_smooth = smoothdata(bt4_loss, 'gaussian', 20);
    plot(bt4_smooth, 'color', colors(4), 'LineWidth', 2, 'LineStyle', '-'); hold on;
    h = plot(bt1_loss, 'color', colors(1), 'LineWidth', 1, 'LineStyle', '-'); h.Color(4) = 0.3; hold on;
    h = plot(bt2_loss, 'color', colors(2), 'LineWidth', 1, 'LineStyle', '-'); h.Color(4) = 0.3; hold on;
    h = plot(bt3_loss, 'color', colors(3), 'LineWidth', 1, 'LineStyle', '-'); h.Color(4) = 0.3; hold on;
    h = plot(bt4_loss, 'color', colors(4), 'LineWidth', 1, 'LineStyle', '-'); h.Color(4) = 0.3; hold on;
    legend('BatchSize=16', 'BatchSize=32', 'BatchSize=64', 'BatchSize=128');
    xlabel('Iteration');
    ylabel('Loss');

end

if 0 % epoches
    figure(1)
    plot(acc_test10, 'color', colors(1), 'LineWidth', 1, 'LineStyle', '-'); hold on;
    plot(acc_train10, 'color', colors(1), 'LineWidth', 1, 'LineStyle', '--'); hold on;
    plot(acc_test20, 'color', colors(2), 'LineWidth', 1, 'LineStyle', '-'); hold on;
    plot(acc_train20, 'color', colors(2), 'LineWidth', 1, 'LineStyle', '--'); hold on;
    plot(acc_test40, 'color', colors(3), 'LineWidth', 1, 'LineStyle', '-'); hold on;
    plot(acc_train40, 'color', colors(3), 'LineWidth', 1, 'LineStyle', '--'); hold on;
    plot(acc_test60, 'color', colors(4), 'LineWidth', 1, 'LineStyle', '-'); hold on;
    plot(acc_train60, 'color', colors(4), 'LineWidth', 1, 'LineStyle', '--'); hold on;
    plot(acc_test80, 'color', colors(5), 'LineWidth', 1, 'LineStyle', '-'); hold on;
    plot(acc_train80, 'color', colors(5), 'LineWidth', 1, 'LineStyle', '--'); hold on;

    legend('test10', 'train10', 'test20', 'train20', 'test40', 'train40', 'test60', 'train60', 'test80', 'train80');
    xlabel('Epoch');
    ylabel('Accuracy');
    % title('accuracy of different batch size');
    hold off
end
