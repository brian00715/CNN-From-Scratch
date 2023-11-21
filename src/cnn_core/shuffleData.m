function [shuffledData, shuffledLabels] = shuffleData(data, labels)
    numSamples = size(data, 3);

    randomIndices = randperm(numSamples);

    shuffledData = data(:, :, randomIndices);
    shuffledLabels = labels(:, randomIndices);
end
