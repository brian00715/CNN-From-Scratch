% Task 1-6
close all;
clear all;

addpath('image_core');

%% Task 1: Display the original image on screen
img_raw = imread("../data/charact2.bmp");
img_raw = rgb2gray(img_raw);
img_size = size(img_raw);
figure; imshow(img_raw); title('Original Image');

%% Task 2: averaging mask and rotating mask

if 0 % Use costom implementation
    [h, w, c] = size(img_raw);
    h1 = ones(3, 3) / 9;
    im1r = filter2(h1, img_raw(:, :, 1));
    im1g = filter2(h1, img_raw(:, :, 2));
    im1b = filter2(h1, img_raw(:, :, 3));
    im1 = cat(3, im1r, im1g, im1b);
    im11 = average_mask(double(img_raw), double(3));
    figure(2)
    imshow(uint8(im11));
    im2 = rotating_mask(double(img_raw));
    figure(3)
    imshow(uint8(im2));
    im3 = im(h / 2:h, :, :);
    imshow(im3)
    im4 = zeros(h / 2, w);

    for i = 1:184

        for j = 1:990
            im4(i, j) = 0.2989 * im(i, j, 1) + 0.587 * im(i, j, 2) + 0.114 * im(i, j, 3);
        end

    end

    im4 = im4 > 128;
    imshow(im4);
end

if 1 % Use toolbox
    mask_ave_33 = ones(3, 3) / 9;
    img_ave1 = imfilter(img_raw, mask_ave_33);
    mask_ave_55 = ones(7, 7) / 49;
    img_ave2 = imfilter(img_raw, mask_ave_55);
    figure; subplot(3, 1, 1); imshow(img_raw); title('Original Image');
    subplot(3, 1, 2); imshow(img_ave1); title('Average Masked Image (3x3)');
    subplot(3, 1, 3); imshow(img_ave2); title('Average Masked Image (5x5)');
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

%% Task 3: Subimage comprising the middle line
img_middle_line = img_raw(ceil(img_size(1) / 2):img_size(1), :);
figure; imshow(img_middle_line); title('HD44780A00');

%% Task4: binary

%% denoising (optional )
if 0
    img_filted = imgaussfilt(img_middle_line, 3);
    % img_filted = myGaussianFilter(img_middle_line, 3, 0.1);

    % num_denoise = 5;
    % for i = 1:num_denoise
    %     img_filted = myMedianFilter(img_filted, [5 5]);
    % end

    figure;
    subplot(2, 1, 1); imshow(img_middle_line); title('Original Image');
    subplot(2, 1, 2); imshow(img_filted); title('Filtered Image');
else
    img_filted = img_middle_line;
end

%% option2: open operation
% img_filted = imopen(img_filted, strel('disk', 8));
img_filted = myOpen(img_filted, strel('disk', 8));
figure; imshow(img_filted); title('Opened Image');
% erode vertically
% img_filted = imerode(img_filted, strel('line', 5, 90));
img_filted = myErode(img_filted, strel('line', 5, 90));
figure; imshow(img_filted); title('Eroded Image');
%imwrite(img_filted,"before_bin.png");

threshold = myOtsuThres(img_filted);
img_bin = imbinarize(img_filted, threshold);
figure; imshow(img_bin); title('Binary Image');
%imwrite(img_bin,"after_bin.png");

%% Task5: outline
% img_outline = edge(img_bin, 'canny');
[x, y, z] = size(img_bin);
I = im2double(img_bin);

I_sobel = zeros(x, y);
I_sobelx = I_sobel;
I_sobely = I_sobel;

for i = 2:x - 1

    for j = 2:y - 1
        I_sobelx(i, j) = abs(I(i - 1, j + 1) - I(i - 1, j - 1) + 2 * I(i, j + 1) - 2 * I(i, j - 1) + I(i + 1, j + 1) - I(i + 1, j - 1));
        I_sobely(i, j) = abs(I(i - 1, j - 1) - I(i + 1, j - 1) + 2 * I(i - 1, j) - 2 * I(i + 1, j) + I(i - 1, j + 1) - I(i + 1, j + 1));
    end

end

I_sobel = I_sobelx + I_sobely;

for i = 1:x

    for j = 1:y

        if I_sobel(i, j) > 0.7
            I_sobel(i, j) = 1;
        else
            I_sobel(i, j) = 0;
        end

    end

end

% figure; imshowpair(I, I_sobel, 'montage');
% imwrite(I_sobel, 'after_sobel.jpg');
img_outline = I_sobel;
figure; imshow(img_outline); title('Outline');

%% Task6: character segmentation

width = 96;
width_th = 120;
%  crop according to the connected components
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
        %imwrite(character1, sprintf('../temp/%d.png', cnt));
        cnt = cnt + 1;
        figure; imshow(character2); %title('Character Image 2');
        %imwrite(character2, sprintf('../temp/%d.png', cnt));
        cnt = cnt + 1;
    else
        figure;
        imshow(character);
        %imwrite(character, sprintf('../temp/%d.png', cnt));
        cnt = cnt + 1;
    end

end

if 0 % crop according to the width
    col_idx = 1;

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

function im2 = average_mask(im1, r)
    [h, w, c] = size(im1);
    im2 = zeros(h, w, c);
    mask = double(1.0 / (r * r));
    edge = int16(r / 2);
    width = edge - 1;

    for x = 1:c

        for i = edge:h - edge

            for j = edge:w - edge
                pix = 0.0;

                for a = i - width:i + width

                    for b = j - width:j + width
                        pix = pix + im1(a, b, x) * mask;
                    end

                end

                im2(i, j, x) = uint8(pix);
            end

        end

    end

end

function llist = location_list(x, y)
    llist = [[x, x + 2, y, y + 2]; [x - 1, x + 1, y, y + 2]; [x - 2, x, y, y + 2]; ...
                                                                  [x - 2, x, y - 1, y + 1]; [x - 2, x, y - 2, y]; [x - 1, x + 1, y - 2, y]; ...
                                                                  [x, x + 2, y - 2, y]; [x, x + 2, y - 1, y + 1]; [x - 1, x + 1, y - 1, y + 1]];
end

function [value, b] = dispersion_value(im, r)
    %r
    a = 0.0;
    b = 0.0;
    % TODO: running slow here
    [i1, i2, j1, j2] = deal(r(1), r(2), r(3), r(4));

    for i = i1:i2

        for j = j1:j2
            a = a + im(i, j) * im(i, j);
            b = b + im(i, j);
        end

    end

    %b*b
    %(a-(b*b/9))
    value = (a - (b * b / 9)) / 9;
end

function im2 = rotating_mask(im1)
    [h, w, c] = size(im1);
    im2 = zeros(h, w, c);

    for x = 1:c

        for i = 3:h - 3

            for j = 3:w - 3
                dispersion_values = [];
                llist = location_list(i, j);

                for a = 1:size(llist)
                    %a
                    %llist
                    [t1, t2] = dispersion_value(im1(:, :, x), llist(a, :));
                    dispersion_values = [dispersion_values; [t1, t2]];
                end

                dispersion_values = sortrows(dispersion_values, 1);
                %dispersion_values
                im2(i, j, x) = uint8(dispersion_values(1, 2) / 9);
            end

        end

    end

end
