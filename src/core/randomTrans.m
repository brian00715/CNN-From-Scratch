function augmentedImage = randomTrans(image)
    % 输入：
    %   image: 输入图像

    % 随机生成平移量
    translateX = randi([-10, 10]);
    translateY = randi([-10, 10]);

    % 随机生成缩放因子
    scaleFactor = 1 + rand() * 0.2; % 缩放范围在 0.8 到 1.2 之间

    % 随机生成旋转角度
    rotateAngle = randi([-15, 15]);

    % 随机决定是否进行水平翻转
    % flipHorizontal = rand() > 0.5;
    flipHorizontal = false;

    % 创建仿射变换矩阵
    tform = affine2d([
                      scaleFactor * cosd(rotateAngle), -sind(rotateAngle), 0;
                      sind(rotateAngle), scaleFactor * cosd(rotateAngle), 0;
                      translateX, translateY, 1
                      ]);

    % 应用变换
    augmentedImage = imwarp(image, tform);

    % 随机决定是否进行水平翻转
    if flipHorizontal
        augmentedImage = flip(augmentedImage, 2);
    end

end
