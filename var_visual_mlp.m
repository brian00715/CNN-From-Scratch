p2 = load('params');
p2 = p2.params;


dsFolder = "p_dataset_26";
subFolder = ["0", "4", "7", "8", "A", "D", "H"];
categories = eye(7);

numImages = 254 * 7;


testFolder = "given_image_split";
testFiles = dir(fullfile(testFolder, "*.png"));
finaltestfeatures = zeros(16384, 10);
for j = 1:length(testFiles)
        testFilePath = fullfile(testFolder, testFiles(j).name);
        image = imread(testFilePath);
        image = imresize(image, [128, 128]);
        finaltestfeatures(:,j) = image(:);
end

finaltestfeatures = 255 - finaltestfeatures;

[AL, caches] = model_forward(finaltestfeatures, p2);
maxValues = max(AL);
testOut = AL == maxValues;


for i = 1:length(subFolder)
    folderPath = fullfile(dsFolder, subFolder(i));
    pngFiles = dir(fullfile(folderPath, "*.png"));
    for j = 1:length(pngFiles)
        pngFilePath = fullfile(folderPath, pngFiles(j).name);
        image = imread(pngFilePath);
        features(:,(i-1)*254 + j) = image(:);
        labels(:,(i-1)*254 + j) = categories(:, i);
    end
end


cv = cvpartition(numImages, 'HoldOut', 0.25);
idx = cv.test;
trainIdx = training(cv);
testIdx = test(cv);
trainFeatures = features(:, trainIdx);
trainLabels = labels(:, trainIdx);
testFeatures = features(:, testIdx);
testLabels = labels(:, testIdx);
testImages = size(testLabels, 2);
trainImages = size(trainLabels, 2);


[AL, caches] = model_forward(double(testFeatures), p2);
maxValues = max(AL);
preds = AL == maxValues;



figure();
% load(log_path + 'results_on_test.mat');
% miss_detect_cnt = zeros(7,1);
% miss_detect_idx = [];
% for i=1:size(preds,2)
%     if preds(:, i)~=testLabels(i)
%         miss_detect_idx = [miss_detect_idx;i];
%         miss_detect_cnt(labels_test(i)) = miss_detect_cnt(labels_test(i))+1;
%     end
% end
confu_mat = confusionmat(vec2ind(logical(testLabels)), vec2ind(preds));
confusionchart(confu_mat);
title('Confusion Matrix');


function [A, cache] = relu(Z)
    A = max(0, Z);
    cache = Z;
end

function dZ = relu_backward(dA, cache)
    Z = cache;
    dZ = dA;
    dZ(Z <=0) = 0;
end


function [A, cache] = sigmoid(Z)
    A = 1./(1+exp(-Z));
    cache = Z;
end

function dZ = sigmoid_backward(dA, cache)
    Z = cache;
    s = 1./(1+exp(-Z));
    dZ = dA .* s .* (1-s);
end

function dZ = softmax_backward(dA, cache)
    Z = cache;
    s = softmax(Z);
    dZ = dA .* s .* (1-s);
end

function [W, b] = initialize_weights(n_x, n_y)
    W = randn(n_y, n_x) * 0.01;
    b = zeros(n_y, 1);
end

function params = initialize_network(layers)
    params = {};
    for i = 1:size(layers, 2)-1
        [W, b] = initialize_weights(layers(i), layers(i+1));
        params{end+1} = {W,b};
    end
end


function [Z, cache] = linear_forward(A, W, b)
    Z = W * A + b;
    cache = {A, W, b};
end

function [A, cache] = linear_activation_forward(A_prev, W, b, activation)
    [Z, linear_cache] = linear_forward(A_prev, W, b);
    if activation == 1
        [A, activation_cache] = sigmoid(Z);
    else
        activation_cache = Z;
        A = softmax(Z);
    end
    cache = {linear_cache, activation_cache};
end

function [AL, caches] = model_forward(X, params)
    caches = {};
    A = X;
    L = size(params, 2);
    for i = 1:L-1
        A_prev = A;
        [A, cache] = linear_activation_forward(A_prev, params{i}{1}, params{i}{2}, 1);
        caches{end+1} = cache;
    end
    A_prev = A;
    [AL, cache] = linear_activation_forward(A_prev, params{L}{1}, params{L}{2}, 2);
    caches{end+1} = cache;
end

function loss = compute_loss(AL, y)
    loss = sum(sum(-((y .* log(AL)) + ((1 - y) .* log(1 - AL))))) / size(AL, 2);
end

function [dA_prev, dW, db] = linear_backward(dZ, cache)
    [A_prev, W, b] = cache{:};
    m = size(A_prev, 2);
    dW = 1 / m * dZ * A_prev' + 1e-4 / m * W;
    db = 1 / m * sum(dZ, 2);
    dA_prev = W' * dZ;
end

function [dA_prev, dW, db] = linear_activation_backward(dA, cache, activation)
    [linear_cache, activation_cache] = cache{:};
    if activation == 1
        dZ = sigmoid_backward(dA, activation_cache);
    else
        dZ = softmax_backward(dA, activation_cache);
    end
    [dA_prev, dW, db] = linear_backward(dZ, linear_cache);
end

function grads = model_backward(AL, Y, caches)
    grads = {};
    L = size(caches, 2);
    dAL = - Y ./ AL + (1 - Y) ./ (1 - AL);
    [dA, dW, db] = linear_activation_backward(dAL, caches{L}, 2);
    grads{end+1} = {dA, dW, db};
    for l = L-1:-1:1
        [dA, dW, db] = linear_activation_backward(dA, caches{l}, 1);
        grads{end+1} = {dA, dW, db};
    end
end

function params = update_parameters(params, grads, learning_rate)
    L = size(params, 2);
    for i = 1:L
        params{i}{1} = params{i}{1} - learning_rate * grads{L+1-i}{2};
        params{i}{2} = params{i}{2} - learning_rate * grads{L+1-i}{3};
    end
end
        
