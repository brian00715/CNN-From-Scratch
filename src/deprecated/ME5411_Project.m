%% part_0 图像分割
addpath("../data/")
I = imread('charact2.jpg');
[x1,x2,y1,y2] = deal(50,950,200,350);
I1 = imcrop(I,[x1,y1,abs(x1-x2),abs(y1-y2)]);
figure,imshow(I1);
imwrite(I1,'after_cut.jpg');

%% part_1 读取图片及转换为灰度图
close all;
clear;
clc;
A=imread('after_cut.jpg');
I1 = rgb2gray(A); %TODO
figure;
imshowpair(A,I1,'montage')
imwrite(I1,'after_grey.jpg');

%% part_2 均值滤波及中值滤波
close all;
clear;
clc;
I = imread("after_grey.jpg");
Mat = ones(3,3)/9;
[m,n] = size(I);
I1 = double(I);
I2 = I1;
%从去除边缘的像素开始滤波
for i=2:m-1
    for j=2:n-1
        Mat2 = I1(i-1:i+1,j-1:j+1).*Mat;
        s = sum(Mat2(:));
        I2(i,j) = s;
    end
end
I3 = uint8(I2);
I4 = I3;
I5 = I4;
%从去除边缘的像素开始滤波
for i=2:m-1
    for j=2:n-1
        Mat = I4(i-1:i+1,j-1:j+1);
        Mat2 = Mat(:);
        s=sort(Mat2); %对像素进行排序
        I5(i,j) = s(floor(9/2)+1); %选取中值
    end
end
I6 = uint8(I5);

%图像对比
figure;
subplot(311)
imshow(I);title('原始图像');
subplot(312)
imshow(I3);title('中值滤波图像');
subplot(313)
imshow(I6);title('均值滤波图像');
imwrite(I6,'after_filter.jpg');

%% part_3 二值化处理
close all;
clear;
clc;
I = imread("after_filter.jpg");
I1=imbinarize(I); %TODO

%去除面积较小的区域
CC = bwconncomp(I1,8);
S = regionprops(CC,'Area');
L = labelmatrix(CC);
I1 = ismember(L,find([S.Area]>=200));

figure;
imshowpair(I,I1,'montage')
imwrite(I1,'after_double.jpg');

%% part_4 基于sobel方法的边缘提取
close all;
clear;
clc;
I = imread('after_double.jpg');
[x,y,z]=size(I);
I=im2double(I);

I_sobel=zeros(x,y);
I_sobelx=I_sobel;
I_sobely=I_sobel;
for i=2:x-1
    for j=2:y-1
        I_sobelx(i,j)=abs(I(i-1,j+1)-I(i-1,j-1)+2*I(i,j+1)-2*I(i,j-1)+I(i+1,j+1)-I(i+1,j-1));
        I_sobely(i,j)=abs(I(i-1,j-1)-I(i+1,j-1)+2*I(i-1,j)-2*I(i+1,j)+I(i-1,j+1)-I(i+1,j+1));
    end
end
I_sobel=I_sobelx+I_sobely;
for i=1:x
    for j=1:y
        if I_sobel(i,j)>0.7
            I_sobel(i,j)=1;
        else
            I_sobel(i,j)=0;
        end
    end
end

figure
imshowpair(I,I_sobel,'montage')
imwrite(I_sobel,'after_sobel.jpg');

%% part_5 字符分割
close all;
clear;
clc;
I = imread('after_double.jpg');
[m,n] = size(I);


set1 = 1000; % RGB变化的最小值
set2 = 4000; % 切割部分RGB的最大值
set3 = 20; % 字符间距

col_A = sum(I);
A = zeros(1,n);
counter = 0;
for i = 1:n-1
    if col_A(1,i+1) - col_A(1,i) > set1 && col_A(1,i) < set2
        counter = counter + 1;
        A(1,counter) = i;
    end
end
A(:,all(A==0,1))=[];


temp = 0;
for j = A
    if j - temp > set3
        [x1,x2,y1,y2] = deal(temp,j,0,m);
        I1 = imcrop(I,[x1,y1,abs(x1-x2),abs(y1-y2)]);
        temp = j;
        figure
        imshow(I1)
    end
end

[x1,x2,y1,y2] = deal(temp,n-8,0,m);
I1 = imcrop(I,[x1,y1,abs(x1-x2),abs(y1-y2)]);
figure
imshow(I1)