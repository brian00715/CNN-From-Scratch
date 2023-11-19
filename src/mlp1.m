clc; clear;
dsFolder = "p_dataset_26";
subFolder = ["0", "4", "7", "8", "A", "D", "H"];
categories = eye(7);

layers = [128 * 128, 32 * 32, 8 * 8, 7];

numImages = 254 * 7;
imageSize = 128;
lr = 0.02;
epoches = 1000;
varFrec = 50;

features = zeros(imageSize * imageSize, numImages);
labels = zeros(7, numImages);

for i = 1:length(subFolder)
    folderPath = fullfile(dsFolder, subFolder(i));
    pngFiles = dir(fullfile(folderPath, "*.png"));

    for j = 1:length(pngFiles)
        pngFilePath = fullfile(folderPath, pngFiles(j).name);
        image = imread(pngFilePath);
        features(:, (i - 1) * 254 + j) = image(:);
        labels(:, (i - 1) * 254 + j) = categories(:, i);
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

params = initialize_network(layers);

trainLoss = [];
testLoss = [];
trainAcc = [];
testAcc = [];

for i = 1:epoches
    [AL, caches] = model_forward(trainFeatures, params);
    grads = model_backward(AL, trainLabels, caches);
    params = update_parameters(params, grads, lr);

    if mod(i, varFrec) == 0
        disp(['epoch: ', num2str(i)])
        loss = compute_loss(AL, trainLabels);
        disp(['train loss: ', num2str(loss)])
        trainLoss(end + 1) = loss;
        maxValues = max(AL);
        testOut = AL == maxValues;
        same = sum(sum(testOut == trainLabels));
        acc = (same - 5 * trainImages) / 2.0 / trainImages;
        trainAcc(end + 1) = acc;
        disp(['train acc: ', num2str(acc)])
        [AL, caches] = model_forward(testFeatures, params);
        loss = compute_loss(AL, testLabels);
        disp(['test loss: ', num2str(loss)])
        testLoss(end + 1) = loss;
        maxValues = max(AL);
        testOut = AL == maxValues;
        same = sum(sum(testOut == testLabels));
        acc = (same - 5 * testImages) / 2.0 / testImages;
        testAcc(end + 1) = acc;
        disp(['test acc: ', num2str(acc)])
    end

end

fig = figure;
x = 50:50:1000;
plot(x, trainLoss, '-*b', x, trainAcc, '-ob', x, testLoss, '-*r', x, testAcc, '-or');
axis([0, 1000, 0, 3])
set(gca, 'XTick', [0:50:1000])
set(gca, 'YTick', [0:0.2:3])
legend('trainLoss', 'trainAcc', 'testLoss', 'testAcc');
xlabel('epoch') %x轴坐标描述
imwrite(frame2im(getframe(fig)), 'results.png');

function [A, cache] = relu(Z)
    A = max(0, Z);
    cache = Z;
end

function dZ = relu_backward(dA, cache)
    Z = cache;
    dZ = dA;
    dZ(Z <= 0) = 0;
end

function [A, cache] = sigmoid(Z)
    A = 1 ./ (1 + exp(-Z));
    cache = Z;
end

function dZ = sigmoid_backward(dA, cache)
    Z = cache;
    s = 1 ./ (1 + exp(-Z));
    dZ = dA .* s .* (1 - s);
end

function dZ = softmax_backward(dA, cache)
    Z = cache;
    s = softmax(Z);
    dZ = dA .* s .* (1 - s);
end

function [W, b] = initialize_weights(n_x, n_y)
    W = randn(n_y, n_x) * 0.01;
    b = zeros(n_y, 1);
end

function params = initialize_network(layers)
    params = {};

    for i = 1:size(layers, 2) - 1
        [W, b] = initialize_weights(layers(i), layers(i + 1));
        params{end + 1} = {W, b};
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

    for i = 1:L - 1
        A_prev = A;
        [A, cache] = linear_activation_forward(A_prev, params{i}{1}, params{i}{2}, 1);
        caches{end + 1} = cache;
    end

    A_prev = A;
    [AL, cache] = linear_activation_forward(A_prev, params{L}{1}, params{L}{2}, 2);
    caches{end + 1} = cache;
end

function loss = compute_loss(AL, y)
    loss = sum(sum(- ((y .* log(AL)) + ((1 - y) .* log(1 - AL))))) / size(AL, 2);
end

function [dA_prev, dW, db] = linear_backward(dZ, cache)
    [A_prev, W, b] = cache{:};
    m = size(A_prev, 2);
    dW = 1 / m * dZ * A_prev' +1e-4 / m * W;
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
    dAL =- Y ./ AL + (1 - Y) ./ (1 - AL);
    [dA, dW, db] = linear_activation_backward(dAL, caches{L}, 2);
    grads{end + 1} = {dA, dW, db};

    for l = L - 1:-1:1
        [dA, dW, db] = linear_activation_backward(dA, caches{l}, 1);
        grads{end + 1} = {dA, dW, db};
    end

end

function params = update_parameters(params, grads, learning_rate)
    L = size(params, 2);

    for i = 1:L
        params{i}{1} = params{i}{1} - learning_rate * grads{L + 1 - i}{2};
        params{i}{2} = params{i}{2} - learning_rate * grads{L + 1 - i}{3};
    end

end
