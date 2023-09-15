close all;

img_raw = imread("./data/charact2.bmp", "bmp");
img_raw = rgb2gray(img_raw);
img_size = size(img_raw);
% figure; imshow(img_raw); title('Original Image');

mask_ave_33 = ones(3, 3) / 9;
img_ave1 = imfilter(img_raw, mask_ave_33);
mask_ave_55 = ones(7, 7) / 49;
img_ave2 = imfilter(img_raw, mask_ave_55);
% figure; subplot(1, 3, 1); imshow(img_raw); title('Original Image');
% subplot(1, 3, 3); imshow(img_ave2); title('Average Masked Image (5x5)');
% subplot(1, 3, 2); imshow(img_ave1); title('Average Masked Image (3x3)');

rotating_mask_33 = [0 1 0; -1 0 1; 0 -1 0];
img_rotating1 = imfilter(img_raw, rotating_mask_33);
rotating_mask_55 = [0 0 1 0 0; 0 1 0 1 0; -1 0 0 0 1; 0 -1 0 -1 0; 0 0 -1 0 0];
img_rotating2 = imfilter(img_raw, rotating_mask_55);
% figure; subplot(1, 2, 1); imshow(img_rotating1); title('Rotating Masked Image (3x3)');
% subplot(1, 2, 2); imshow(img_rotating2); title('Rotating Masked Image (5x5)');

%% subimage comprising the middle line
img_middle_line = img_raw(ceil(img_size(1) / 2):img_size(1), :);
% figure; imshow(img_middle_line); title('HD44780A00');

%% denoising
img_filted = img_middle_line;
% img_filted = imgaussfilt(img_middle_line, 1);
% num_denoise = 5;

% for i = 1:num_denoise
%     img_filted = medfilt2(img_filted, [5 5]);
% end

% figure;
% subplot(2, 1, 1); imshow(img_middle_line); title('Original Image');
% subplot(2, 1, 2); imshow(img_filted); title('Filtered Image');

%% binary
% open operation
img_filted = imopen(img_filted, strel('disk', 8));
% erode vertically
% img_filted = imerode(img_filted, strel('line', 5, 90));
% figure; imshow(img_filted); title('Eroded Image');

% threshold = graythresh(img_filted);
threshold = otsu(img_filted);
img_bin = imbinarize(img_filted, threshold);
figure; imshow(img_bin); title('Binary Image');

%% outline
img_outline = edge(img_bin, 'canny');
figure; imshow(img_outline); title('Outline');

%% character segmentation
cc = bwconncomp(img_bin);
characterProps = regionprops(cc, 'BoundingBox');

for i = 1:length(characterProps)
    bb = characterProps(i).BoundingBox;
    character = imcrop(img_bin, bb);
    % figure;
    % imshow(character);
end
