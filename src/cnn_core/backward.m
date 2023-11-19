% Description: Compute gradient of CNN
% Input:
%   cnn: CNN model
%   X: Input data
%   options: Options
% Output:
%   grad: Gradient
%   cnn: CNN model with updated gradient
function [grad, cnn] = backward(cnn, X, options)
    numImages = size(X, 4);
    % cnn.layers{1}.activations = X;
    % cnn.layers{1}.outDim = size(X);
    % cnn = forward(cnn,X);
    numLayers = size(cnn.layers, 1);
    l2_penalty = options.l2_penalty;

    %% Compute delta
    for i = numLayers - 1:-1:2

        if strcmp(cnn.layers{i}.type, 'Conv2D') % Delta of 'Conv2D'

            if strcmp(cnn.layers{i + 1}.type, 'Linear') || strcmp(cnn.layers{i + 1}.type, 'output') % Previous layer is 'Linear' or 'output'
                delta_s = cnn.layers{i}.W' * cnn.layers{i + 1}.delta;
                delta_s = reshape(delta_s, [cnn.layers{i}.outDim numImages]);
            else % Previous layer is 'Conv2D'

                numFilters = cnn.layers{i + 1}.numFilters;
                delta_s = zeros([cnn.layers{i}.outDim numImages]);

                for imageNum = 1:numImages
                    delta = zeros(cnn.layers{i}.outDim(1:3));

                    for filterNum = 1:numFilters
                        delta = delta + convn(cnn.layers{i + 1}.delta(:, :, filterNum, imageNum), ...
                            cnn.layers{i}.W(:, :, :, filterNum), 'full');
                    end

                    delta_s(:, :, :, imageNum) = delta;
                end

                % !!!real convolution, dim3 should also be fliped
                delta_s = flip(delta_s, 3);
            end

            convDim = cnn.layers{i}.outDim(1) * cnn.layers{i}.poolDim;
            numFilters = cnn.layers{i}.numFilters;
            delta_c = zeros(convDim, convDim, numFilters, numImages);

            for imageNum = 1:numImages

                for filterNum = 1:numFilters
                    delta_c(:, :, filterNum, imageNum) = (1 ./ cnn.layers{i}.poolDim ^ 2) * ...
                        kron((delta_s(:, :, filterNum, imageNum)), ones(cnn.layers{i}.poolDim));
                end

            end

            cnn.layers{i}.delta = delta_c .* cnn.layers{i - 1}.realGradFunc(cnn.layers{i}.convolvedFeatures);
        elseif strcmp(cnn.layers{i}.type, 'Linear') % Delta of 'Linear'
            cnn.layers{i}.delta = cnn.layers{i}.W' * cnn.layers{i + 1}.delta ...
                .* cnn.layers{i - 1}.realGradFunc(cnn.layers{i}.activations);
        end

    end

    %% Compute gradient
    for i = numLayers - 1:-1:1

        if strcmp(cnn.layers{i + 1}.type, 'Conv2D')
            numFilters = cnn.layers{i + 1}.numFilters;
            Wgrad = zeros(size(cnn.layers{i}.W));
            bgrad = zeros(size(cnn.layers{i}.b));
            filterDim = cnn.layers{i + 1}.filterDim;

            if strcmp(cnn.layers{i}.type, 'input')
                filterDim3 = size(X, 3);
            else
                filterDim3 = cnn.layers{i}.outDim(3);
            end

            for filterNum = 1:numFilters
                Wgrad_i = zeros(filterDim, filterDim, filterDim3);

                for imageNum = 1:numImages
                    Wgrad_i = Wgrad_i + convn(cnn.layers{i}.activations(:, :, :, imageNum), ...
                        rot90(cnn.layers{i + 1}.delta(:, :, filterNum, imageNum), 2), 'valid');
                end

                Wgrad(:, :, :, filterNum) = flip(Wgrad_i, 3) / numImages;
                b_i = cnn.layers{i + 1}.delta(:, :, filterNum, :);
                bgrad(filterNum) = sum(b_i(:)) / numImages;
            end

            cnn.layers{i}.bgrad = bgrad;

            if options.use_l2
                cnn.layers{i}.Wgrad = Wgrad + l2_penalty * cnn.layers{i}.W;
                % cnn.layers{i}.bgrad = bgrad + l2_penalty * cnn.layers{i}.b;
            else
                cnn.layers{i}.Wgrad = Wgrad;
            end

        elseif strcmp(cnn.layers{i + 1}.type, 'Linear') || strcmp(cnn.layers{i + 1}.type, 'output')

            if strcmp(cnn.layers{i}.type, 'Conv2D')
                activations = reshape(cnn.layers{i}.activations, [], numImages);
                cnn.layers{i}.Wgrad = cnn.layers{i + 1}.delta * activations' / numImages;
            else
                cnn.layers{i}.Wgrad = cnn.layers{i + 1}.delta * cnn.layers{i}.activations' / numImages;
            end

            cnn.layers{i}.bgrad = sum(cnn.layers{i + 1}.delta, 2) / numImages;

            if options.use_l2
                cnn.layers{i}.Wgrad = cnn.layers{i}.Wgrad + l2_penalty * cnn.layers{i}.W;
                % cnn.layers{i}.bgrad = cnn.layers{i}.bgrad + l2_penalty * cnn.layers{i}.b;
            end

        end

    end

    %% Unroll gradient
    grad = [];
    numLayers = size(cnn.layers, 1);

    for i = 1:numLayers - 1
        Wgrad = cnn.layers{i}.Wgrad(:);
        grad = [grad; Wgrad];
    end

    for i = 1:numLayers - 1
        bgrad = cnn.layers{i}.bgrad(:);
        grad = [grad; bgrad];
    end

end
