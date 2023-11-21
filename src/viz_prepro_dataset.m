clear all;
load("../data/train.mat");
load("../data/test.mat");

addpath("cnn_core/");
addpath('image_core');
figure;

data = data_train;
labels = labels_train;
for i = 1:size(data, 4)
    img = data(:, :, :, i);
    label = labels(i);
    label = labelNum2Char(label);
    if label=="H"
        text(0, 0, label, 'Color', 'red', 'FontSize', 12, 'HorizontalAlignment', 'center');
        imshow(img);
        title(label);
        pause(0.1);
    end
end
