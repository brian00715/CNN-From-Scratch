% Description: excute the a complete training process given the settings
function flag = trainMachine(cnn, dataset_options, train_options, ... ,
        data_train, labels_train, data_test, labels_test)
    numClasses = max(labels_train);
    cnn = initModelParams(cnn, data_train, numClasses);
    cnn = learn(cnn, data_train, labels_train, data_test, labels_test, train_options);

    [preds, ~] = predict(cnn, data_test);
    acc = sum(preds == labels_test) / length(preds);
    fprintf('Final accuracy: %f\n', acc);

    %% save model and parameters
    fileID = fopen(train_options.log_path + "results.txt", 'a');
    fprintf(fileID, 'Final accuracy: %f\n', acc);
    fclose(fileID);
    save(train_options.log_path + "final_cnn.mat", 'cnn');
    save(train_options.log_path + "results_on_testset.mat", 'preds', 'labels_test');
    save(train_options.log_path + "dataset_options.mat", 'dataset_options');
    save(train_options.log_path + "train_options.mat", 'train_options');

    % convert hyper-parameters to json
    json = jsonencode(options);
    fid = fopen(train_options.log_path + "train_options.json", 'w');
    fprintf(fid, json);
    json = jsonencode(dataset_options);
    fid = fopen(train_options.log_path + "dataset_optionss.json", 'w');
    fprintf(fid, json);

    % convert model structure to json
    model_json = model2json(cnn);
    fid = fopen(train_options.log_path + "model.json", 'w');
    fprintf(fid, model_json);
    flag = true;
end
