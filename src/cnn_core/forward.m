% Description: Forward pass of model
% Input:
%   cnn: model
%   X: input data
% Output:
%   cnn: model with updated outputs
function cnn = forward(cnn, X, options)
    numImages = size(X, 4);
    numLayers = size(cnn.layers, 1);

    for i = 1:numLayers

        if strcmp(cnn.layers{i}.type, 'input')
            cnn.layers{i}.activations = X;
        elseif strcmp(cnn.layers{i}.type, 'Conv2D')
            filterDim = cnn.layers{i}.filterDim;
            featureDim = cnn.layers{i - 1}.outDim(1);
            convDim = featureDim - filterDim + 1;
            numFilters = cnn.layers{i}.numFilters;
            cnn.layers{i}.convolvedFeatures = zeros(convDim, convDim, numFilters, numImages);

            % convolution
            for imageNum = 1:numImages

                for filterNum = 1:numFilters
                    curFilter = cnn.layers{i - 1}.W(:, :, :, filterNum);
                    curFilter = rot90(curFilter, 2);
                    im = cnn.layers{i - 1}.activations(:, :, :, imageNum);
                    convolvedFeatures = convn(im, curFilter, 'valid'); % zero-padded convolution
                    convolvedFeatures = convolvedFeatures + cnn.layers{i - 1}.b(filterNum);
                    convolvedFeatures = cnn.layers{i - 1}.realActiFunc(convolvedFeatures);
                    cnn.layers{i}.convolvedFeatures(:, :, filterNum, imageNum) = convolvedFeatures;
                end

            end

            %pooling
            cnn.layers{i}.activations = zeros([cnn.layers{i}.outDim numImages]);
            poolDim = cnn.layers{i}.poolDim;
            mask = ones(poolDim, poolDim) / poolDim ^ 2;

            for imageNum = 1:numImages

                for filterNum = 1:numFilters
                    tmp = conv2(cnn.layers{i}.convolvedFeatures(:, :, filterNum, imageNum), mask, 'valid');
                    cnn.layers{i}.activations(:, :, filterNum, imageNum) = tmp(1:poolDim:end, 1:poolDim:end);
                end

            end

        elseif strcmp(cnn.layers{i}.type, 'Linear') || (strcmp(cnn.layers{i}.type, 'output') && cnn.layers{i}.softmax == false)
            last_acti = reshape(cnn.layers{i - 1}.activations, [], numImages);
            output = cnn.layers{i - 1}.W * last_acti + repmat(cnn.layers{i - 1}.b, 1, numImages);

            if options.train_mode == true

                if isfield(cnn.layers{i}, 'dropout')
                    mask = double(rand(size(output)) < (1 - cnn.layers{i}.dropout));
                    output = output .* mask;
                end

            end

            cnn.layers{i}.activations = cnn.layers{i - 1}.realActiFunc(output);

        elseif strcmp(cnn.layers{i}.type, 'output') && cnn.layers{i}.softmax == true
            activations = reshape(cnn.layers{i - 1}.activations, [], numImages);
            activations = cnn.layers{i - 1}.W * activations + repmat(cnn.layers{i - 1}.b, 1, numImages);

            activations = bsxfun(@minus, activations, max(activations, [], 1));
            activations = exp(activations);
            cnn.layers{i}.activations = bsxfun(@rdivide, activations, sum(activations));
        end

    end

end
