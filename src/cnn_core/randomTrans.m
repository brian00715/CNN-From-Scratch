function augmentedImage = randomTrans(image)

    image_dim = size(image, 1);

    % trans 20%
    trans_delta = round(image_dim * 0.2);
    translateX = randi([-trans_delta, trans_delta]);
    translateY = randi([-trans_delta, trans_delta]);

    % scale 20%
    scaleFactor = 1 + rand() * 0.2;

    % rotate 15 degree
    rotateAngle = randi([-30, 30]);

    % flipHorizontal = rand() > 0.5;
    flipHorizontal = false;

    tform = affine2d([
                      scaleFactor * cosd(rotateAngle), -sind(rotateAngle), 0;
                      sind(rotateAngle), scaleFactor * cosd(rotateAngle), 0;
                      translateX, translateY, 1
                      ]);

    augmentedImage = imwarp(image, tform);

    if flipHorizontal
        augmentedImage = flip(augmentedImage, 2);
    end

end
