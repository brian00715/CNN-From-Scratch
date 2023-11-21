function augmentedImage = randTF(image,options)
    image_dim = size(image, 1);
    trans_delta = round(image_dim * options.trans_ratio);
    tform = randomAffine2d("Scale", options.scale_ratio, "XTranslation", [-trans_delta trans_delta], ...
        "YTranslation", [-trans_delta trans_delta], "Rotation", options.rot_range);
    centerOutput = affineOutputView(size(image), tform, "BoundsStyle", "CenterOutput");
    augmentedImage = imwarp(image, tform, 'OutputView', centerOutput,"FillValues",255);
end
