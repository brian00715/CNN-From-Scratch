% sacle image while keeping the size. padding with 0
function paddedImage = scaleKeepSize(img, scale_factor)
    [h, w] = size(img);
    x_fac = scale_factor(1);
    y_fac = scale_factor(2);
    tgt_size = [round(h * y_fac), round(w * x_fac)];
    scaled_img = imresize(img, tgt_size, 'nearest');

    if x_fac + y_fac > 2
        paddedImage = padarray(scaled_img, [h - size(scaled_img, 1), w - size(scaled_img, 2)], 0, 'post');
    else
        paddedImage = zeros(h, w);
        scaled_size = size(scaled_img);
        row_st = floor((h - scaled_size(1)) / 2) + 1;
        col_st = floor((w - scaled_size(2)) / 2) + 1;
        paddedImage(row_st:row_st + scaled_size(1) - 1, col_st:col_st + scaled_size(2) - 1) = scaled_img;

    end

end
