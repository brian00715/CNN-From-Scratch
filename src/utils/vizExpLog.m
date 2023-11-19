% Description: used to generate the figures of training results. It's used with trainMachine.m and
%   is called after the training is done, which means the related data files have been already generated.
function vizExpLog(log_path, preds, labels_test)
    load(log_path + "acc_test.mat", "acc_test");
    load(log_path + "acc_train.mat", "acc_train");
    load(log_path + "lr_ar.mat", "lr_ar");
    load(log_path + "loss_ar.mat", "loss_ar");

    set(gca, 'Visible', 'off');
    f = figure(1);
    set(f, 'Visible', 'off');
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

    f = figure(2);
    set(f, 'Visible', 'off');
    plot(loss_ar, 'LineWidth', 2);
    grid on;
    xlabel('Epoch');
    ylabel('Loss');
    title('Loss');
    saveas(gcf, log_path + 'loss.png');

    f = figure(3);
    set(f, 'Visible', 'off');
    plot(lr_ar, 'LineWidth', 2);
    grid on;
    xlabel('Epoch');
    ylabel('Learning Rate');
    title('Learning Rate');
    saveas(gcf, log_path + 'lr.png');

    % draw confusion matrix
    f = figure(4);
    set(f, 'Visible', 'off');
    miss_detect_cnt = zeros(7, 1);
    miss_detect_idx = [];

    for i = 1:size(preds, 1)

        if preds(i) ~= labels_test(i)
            miss_detect_idx = [miss_detect_idx; i];
            miss_detect_cnt(labels_test(i)) = miss_detect_cnt(labels_test(i)) + 1;
        end

    end

    confu_mat = confusionmat(labels_test, preds);
    confusionchart(confu_mat);
    title('Confusion Matrix');
    saveas(gcf, log_path + 'confusion_matrix.png');

end
