function result = myConvolution(inputImage, filter)
    [rows, cols] = size(inputImage);
    [filterSize, ~] = size(filter);

    result = zeros(rows, cols);

    delta = floor(filterSize / 2);

    for i = 1:rows

        for j = 1:cols
            temp = 0;

            for l = 1:filterSize

                for k = 1:filterSize

                    if (i - delta + l > 0 && i - delta + l <= rows && j - delta + k > 0 && j - delta + k <= cols)
                        temp = temp + inputImage(i - delta + l, j - delta + k) * filter(l, k);
                    end

                end

            end

            result(i, j) = temp;
        end

    end

end
