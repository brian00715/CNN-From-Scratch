% @Description: Learn the CNN model
% @Author: Kuankuan Sima
% @Email: kuankuan_sima@u.nus.edu
% @Input:
%   cnn: CNN model
%   data: training data
%   labels: training labels
%   data_t: testing data
%   labels_t: testing labels
%   options: options for learning
% @Output:
%   cnn_final: learned CNN model
function [cnn_final] = learn(cnn, data_train, labels_train, data_test, labels_test, options)
    % This file is modified based on UFLDL Deep Learning Tutorial
    % http://ufldl.stanford.edu/tutorial/
    %
    % Runs stochastic gradient descent with momentum to optimize the
    % parameters for the given objective.
    %
    % Parameters:
    %  funObj     -  function handle which accepts as input theta,
    %                data, labels and returns cost and gradient w.r.t
    %                to theta.
    %  theta      -  unrolled parameter vector
    %  data       -  stores data in m x n x numExamples tensor
    %  labels     -  corresponding labels in numExamples x 1 vector
    %  options    -  struct to store specific options for optimization
    %
    % Returns:
    %  opttheta   -  optimized parameter vector
    %
    % Options (* required)
    %  epochs*     - number of epochs through data
    %  lr*      - initial learning rate
    %  minibatch*  - size of minibatch
    %  momentum    - momentum constant, defualts to 0.9

    %% Setup
    addpath("utils");
    theta = unrollWeights(cnn);
    assert(all(isfield(options, {'epochs', 'lr', 'minibatch'})), 'Some options not defined');

    if ~isfield(options, 'momentum')
        options.momentum = 0.9;
    end

    epochs = options.epochs;
    lr = options.lr_max;
    minibatch = options.minibatch;
    m = length(labels_train); % training set size
    % Setup for momentum
    mom = 0.5;
    momIncrease = 20;
    velocity = zeros(size(theta));

    %%======================================================================
    it = 0;
    loss_ar = [];
    acc_train = [];
    acc_test = [];
    lr_ar = [];
    best_acc = 0;

    for e = 1:epochs
        % randomly permute indices of data for quick minibatch sampling
        rp = randperm(m);

        for s = 1:minibatch:(m - minibatch + 1)
            it = it + 1;
            % increase momentum after momIncrease iterations
            % if it == momIncrease
            %     mom = options.momentum;
            % end
            mom = options.momentum;

            % get next randomly selected minibatch
            mb_data = data_train(:, :, :, rp(s:s + minibatch - 1));
            mb_labels = labels_train(rp(s:s + minibatch - 1));

            cnn = forward(cnn, mb_data, options);
            [cnn, curr_loss] = calcuLoss(cnn, mb_data, mb_labels, options);

            if options.train_mode
                grad = backward(cnn, mb_data, options);
                % update weights
                % velocity = mom * velocity + lr * grad;
                % theta = theta - velocity;
                velocity = mom * velocity + (1 - mom) * grad;
                theta = theta - lr * velocity;
            end

            % update model
            if exist('theta', 'var')
                cnn = updateWeights(cnn, theta);
            end

            loss_ar = [loss_ar; curr_loss];
            % [preds,~] = predict(cnn, data);
            % preds = zeros(size(labels));
            % curr_acc = sum(preds == labels) / length(preds);
            curr_acc = 0;
            % acc_train = [acc_train; curr_acc];
            progress = 100 * it / floor(options.total_iter);
            it_len = strlength(string(options.total_iter));
            fprintf("it:%*d (%6.2f%%) loss:%.5f acc:%5.2f lr_ar:%f\n", it_len, it, progress, curr_loss, curr_acc, lr);
        end

        [preds,~] = predict(cnn, data_test);
        curr_acc_test = sum(preds == labels_test) / length(preds);
        acc_test = [acc_test; curr_acc_test];
        % [preds,~] = predict(cnn, data_train);
        % curr_acc_train = sum(preds == labels_train) / length(preds);
        curr_acc_train = 0;
        acc_train = [acc_train; curr_acc_train];
        fprintf('\nEpoch %d: acc_test:%f acc_train:%f\n', e, curr_acc_test, curr_acc_train);

        if curr_acc_test > best_acc
            best_acc = curr_acc_test;

            if options.save_best_acc_model
                save(options.log_path + 'cnn_best_acc.mat', 'cnn');
                fileID = fopen(options.log_path + "results.txt", 'w');
                fprintf(fileID, 'Best accuracy: %f\n', curr_acc_test);
                fclose(fileID);
            end

        end

        lr = lrSchedule(e, epochs, options);
        lr_ar = [lr_ar; lr];

    end

    opttheta = theta;
    cnn_final = updateWeights(cnn, opttheta);

    save(options.log_path + 'loss_ar.mat', 'loss_ar');
    save(options.log_path + 'acc_train.mat', 'acc_train');
    save(options.log_path + 'acc_test.mat', 'acc_test');
    save(options.log_path + 'lr_ar.mat', 'lr_ar');
    fprintf('Best accuracy: %f\n', best_acc);
end
