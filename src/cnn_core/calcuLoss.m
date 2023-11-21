% Calculate loss of the network. the loss calculation reflects to the activation of the last layer.
% @Author: Kuankuan Sima
% @Email: kuankuan_sima@u.nus.edu
% Input:
%   cnn: the cnn structure
%   x: the input data
%   y: the label of the input data
% Output: loss
function [cnn, loss] = calcuLoss(cnn, x, y, options)
    numImages = size(x, 4);
    numLayers = size(cnn.layers, 1);
    numClasses = cnn.layers{numLayers}.hiddenUnits;
    l2_penalty = options.l2_penalty;

    %% Compute loss
    if cnn.layers{numLayers}.softmax == true
        groundTruth = full(sparse(y, 1:numImages, 1, numClasses, numImages));
        loss = -1 ./ numImages * groundTruth(:)' * log(cnn.layers{end}.activations(:)); % cross-entrophy loss
        % l2 loss
        if options.use_l2

            for i = 1:numLayers - 1
                loss = loss + l2_penalty * sum(cnn.layers{i}.W(:) .^ 2); %+ l2_penalty * sum(cnn.layers{i}.b(:) .^ 2);
            end

        end

        % in fact, the gradient of cross-entrophy loss w.r.t. to the softmax layer is just like this. it's so easy that I can't believe it.
        cnn.layers{numLayers}.delta = cnn.layers{end}.activations - groundTruth;
    else
        groundTruth = full(sparse(y, 1:numImages, 1, numClasses, numImages));

        if strcmp(cnn.layers{numLayers - 1}.actiFunc, 'tanh')
            groundTruth(groundTruth == 0) = -1;
        end

        % a_nl = cnn.layers{end}.activations;
        % loss = sum(sum(-groundTruth .* log(a_nl) - (1-groundTruth) .* log(1 - a_nl))) / numImages;
        % cnn.layers{numLayers}.delta = a_nl - groundTruth;
        a_nl = cnn.layers{end}.activations;
        loss = sum(sum((a_nl - groundTruth) .^ 2) / 2) / numImages;
        cnn.layers{numLayers}.delta =- (groundTruth - a_nl) .* cnn.layers{numLayers - 1}.realGradFunc(a_nl);
    end
