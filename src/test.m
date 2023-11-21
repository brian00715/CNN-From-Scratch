close all; clear all;
img_raw = imread("../data/charact2.bmp");
img_raw = rgb2gray(img_raw);
% thres = myOtsuThres(img_raw);
% img_bin = imbinarize(img_raw, thres);
% im_erode = myDilate(img_bin, strel('line', 5, 90));
% img_bin = img_raw;
% im_open = myOpen(img_raw, strel('disk', 8));
im_open = myErode(img_raw, strel('disk', 8));
% im_open = imerode(img_raw, strel('disk', 8));
% im_open = myDilate(img_raw, strel('disk', 8));
% im_open = imdilate(img_raw, strel('disk', 8));
figure; imshowpair(img_raw, im_open, 'montage');
title('Original Image (Left) vs. Eroded Image (Right)');
