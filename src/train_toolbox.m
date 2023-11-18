% Load the MNIST dataset
% [XTrain, YTrain, XTest, YTest] = digitTrain4DArrayData;
imageDim = 28;
images = loadMNISTImages('train-images-idx3-ubyte');
images = reshape(images, imageDim, imageDim, 1, []);
labels = loadMNISTLabels('train-labels-idx1-ubyte');
labels(labels == 0) = 10; % Remap 0 to 10

testImages = loadMNISTImages('t10k-images-idx3-ubyte');
testLabels = loadMNISTLabels('t10k-labels-idx1-ubyte');
testImages = reshape(testImages, imageDim, imageDim, 1, []);
testLabels(testLabels == 0) = 10; % Remap 0 to 10

% Move data to the GPU
XTrain = gpuArray(XTrain);
YTrain = gpuArray(YTrain);
XTest = gpuArray(XTest);
YTest = gpuArray(YTest);

% Define the architecture of the custom CNN
layers = [
          imageInputLayer([28 28 1], 'Name', 'input')

          convolution2dLayer(3, 32, 'Padding', 'same', 'Name', 'conv1')
          batchNormalizationLayer('Name', 'bn1')
          reluLayer('Name', 'relu1')

          maxPooling2dLayer(2, 'Stride', 2, 'Name', 'maxpool1')

          convolution2dLayer(3, 64, 'Padding', 'same', 'Name', 'conv2')
          batchNormalizationLayer('Name', 'bn2')
          reluLayer('Name', 'relu2')

          maxPooling2dLayer(2, 'Stride', 2, 'Name', 'maxpool2')

          fullyConnectedLayer(10, 'Name', 'Linear')
          softmaxLayer('Name', 'softmax')
          classificationLayer('Name', 'output')];

% Specify training options
options = trainingOptions('sgdm', ...
    'MaxEpochs', 10, ...
    'MiniBatchSize', 128, ...
    'Verbose', true, ...
    'Plots', 'training-progress', ...
    'ExecutionEnvironment', 'gpu'); % Specify GPU training
Ï
% Train the network
net = trainNetwork(XTrain, YTrain, layers, options);

% Make predictions on the test set
YPred = classify(net, XTest);

% Evaluate the accuracy
accuracy = sum(YPred == YTest) / numel(YTest);
fprintf('Accuracy: %.2f%%\n', accuracy * 100);
