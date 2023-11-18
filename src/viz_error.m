; % show the images that are miss classified
addpath("core")
log_path = "../logs/11-19_00-24-01"
% load(log_path + "/results_on_test.mat")

% preds, labels_test

cnt = 0;

for i = 1:length(preds)

    if preds(i) ~= labels_test(i)
        figure
        imshow(images_test(:, :, :, i))
        title("Predicted: " + labelNum2Char(preds(i)) + " Actual: " + labelNum2Char(labels_test(i)) + ...
            " idx: " + num2str(i))
        cnt = cnt + 1;

        % if cnt > 10
        %     break
        % end

    end

end
