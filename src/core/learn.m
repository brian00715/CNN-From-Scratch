function [cnn_final] = learn(cnn, data, labels, data_t, labels_t, options)
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

    %%======================================================================
    %% Setup
    numClasses = max(labels);
    cnn = initModelParams(cnn, data, numClasses);
    theta = unrollWeights(cnn);
    assert(all(isfield(options, {'epochs', 'lr', 'minibatch'})), 'Some options not defined');

    if ~isfield(options, 'momentum')
        options.momentum = 0.9;
    end

    epochs = options.epochs;
    lr = options.lr_max;
    minibatch = options.minibatch;
    m = length(labels); % training set size
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
            mb_data = data(:, :, :, rp(s:s + minibatch - 1));
            mb_labels = labels(rp(s:s + minibatch - 1));

            cnn = forward(cnn, mb_data);
            [cnn, curr_loss] = calcuLoss(cnn, mb_data, mb_labels, options);
            grad = backward(cnn, mb_data, options);

            % Instructions: Add in the weighted velocity vector to the
            % gradient evaluated above scaled by the learning rate.
            % Then update the current weights theta according to the
            % sgd update rule

            % update weights
            velocity = mom * velocity + lr * grad;
            theta = theta - velocity;

            % update model
            if exist('theta', 'var')
                cnn = updateWeights(cnn, theta);
            end

            loss_ar = [loss_ar; curr_loss];
            preds = predict(cnn, data);
            curr_acc = sum(preds == labels) / length(preds);
            acc_train = [acc_train; curr_acc];
            progress = 100 * it / floor(options.total_iter);
            % fprintf('Epoch %d: curr_loss on iteration %d is %f\n',e,it,curr_loss);
            fprintf("it:%d (%.2f%%) loss_ar:%f acc:%f lr_ar:%f\n", it, progress, curr_loss, curr_acc, lr);
        end

        preds = predict(cnn, data_t);
        curr_acc = sum(preds == labels_t) / length(preds);
        acc_test = [acc_test; curr_acc];
        fprintf('Epoch %d: acc:%f\n', e, curr_acc);

        % aneal learning rate by factor of two after each epoch
        % lr = lr/1.5;
        lr_options.lr_min = options.lr_min;
        lr_options.lr_max = options.lr_max;
        lr_options.method = options.lr_method;
        lr = lrSchedule(e, epochs, lr_options);
        lr_ar = [lr_ar; lr];

    end

    opttheta = theta;
    cnn_final = updateWeights(cnn, opttheta);

    save(options.log_path + 'loss_ar.mat', 'loss_ar');
    save(options.log_path + 'acc_train.mat', 'acc_train');
    save(options.log_path + 'acc_test.mat', 'acc_test');
    save(options.log_path + 'lr_ar.mat', 'lr_ar');

end
