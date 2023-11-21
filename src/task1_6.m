close all;

img_raw = imread("../data/charact2.bmp");
img_raw = rgb2gray(img_raw);
img_size = size(img_raw);
% figure; imshow(img_raw); title('Original Image');

if 0
    mask_ave_33 = ones(3, 3) / 9;
    img_ave1 = imfilter(img_raw, mask_ave_33);
    mask_ave_55 = ones(7, 7) / 49;
    img_ave2 = imfilter(img_raw, mask_ave_55);
    figure; subplot(3, 1, 1); imshow(img_raw); title('Original Image');
    subplot(3, 1, 2); imshow(img_ave1); title('Average Masked Image (3x3)');
    subplot(3, 1, 3); imshow(img_ave2); title('Average Masked Image (5x5)');
end

if 0
    rotating_mask_33 = [0 1 0;
        -1 0 1;
        0 -1 0];
    img_rotating1 = imfilter(img_raw, rotating_mask_33);
    rotating_mask_55 = [
        0 0 1 0 0;
        0 1 0 1 0;
        -1 0 0 0 1;
        0 -1 0 -1 0;
        0 0 -1 0 0];
    img_rotating2 = imfilter(img_raw, rotating_mask_55);
    figure; subplot(2, 1, 1); imshow(img_rotating1); title('Rotating Masked Image (3x3)');
    subplot(2, 1, 2); imshow(img_rotating2); title('Rotating Masked Image (5x5)');
end

%% subimage comprising the middle line
img_middle_line = img_raw(ceil(img_size(1) / 2):img_size(1), :);
% figure; imshow(img_middle_line); title('HD44780A00');

%% denoising
img_filted = img_middle_line;

if 0
    img_filted = imgaussfilt(img_middle_line, 1);
    num_denoise = 5;

    for i = 1:num_denoise
        img_filted = medfilt2(img_filted, [5 5]);
    end

    figure;
    subplot(2, 1, 1); imshow(img_middle_line); title('Original Image');
    subplot(2, 1, 2); imshow(img_filted); title('Filtered Image');
end

%% binary
% open operation
img_filted = imopen(img_filted, strel('disk', 8));
% erode vertically
% img_filted = imerode(img_filted, strel('line', 5, 90));
% figure; imshow(img_filted); title('Eroded Image');
imwrite(img_filted,"before_bin.png");

% threshold = graythresh(img_filted);
threshold = OtsuThres(img_filted);
img_bin = imbinarize(img_filted, threshold);
% figure; imshow(img_bin); title('Binary Image');
imwrite(img_bin,"after_bin.png");

%% outline
img_outline = edge(img_bin, 'canny');
% figure; imshow(img_outline); title('Outline');

%% character segmentation
width = 96;
width_th = 120;

if 0
    % crop according to the width
    col_idx = 1

    while col_idx <= size(img_outline, 2)
        right_edge = col_idx + width - 1;

        if right_edge > size(img_outline, 2)
            right_edge = size(img_outline, 2);
        end

        charac_img = img_outline(:, col_idx:right_edge);
        figure; imshow(charac_img); title('Character Image');
        col_idx = col_idx + width;
    end

end


if 0
    cc = bwconncomp(img_bin);
    characterProps = regionprops(cc, 'BoundingBox');

    cnt = 1;

    for i = 1:length(characterProps)
        bb = characterProps(i).BoundingBox;
        character = imcrop(img_bin, bb);

        if size(character, 2) > width_th
            % crop to the middle
            mid_idx = ceil(size(character, 2) / 2);
            character1 = character(:, 1:mid_idx);
            character2 = character(:, mid_idx + 1:end);
            figure; imshow(character1); %title('Character Image 1');
            imwrite(character1, sprintf('../temp/%d.png', cnt));
            cnt = cnt + 1;
            figure; imshow(character2); %title('Character Image 2');
            imwrite(character2, sprintf('../temp/%d.png', cnt));
            cnt = cnt + 1;
        else
            figure;
            imshow(character);
            imwrite(character, sprintf('../temp/%d.png', cnt));
            cnt = cnt + 1;
        end

    end
end