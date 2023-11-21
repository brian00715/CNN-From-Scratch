% Description: Unroll the weights of the CNN into a single vector
% Input:
%   cnn: the CNN to unroll
% Output:
%   params: the unrolled weights of the CNN
function params = unrollWeights(cnn)
    params = [];
    numLayers = size(cnn.layers, 1);

    for i = 1:numLayers - 1
        W = cnn.layers{i}.W(:);
        params = [params; W];
    end

    for i = 1:numLayers - 1
        b = cnn.layers{i}.b(:);
        params = [params; b];
    end
