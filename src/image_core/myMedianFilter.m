function outputImage = myMedianFilter(inputImage, filterSize)
    [rows, cols] = size(inputImage);
    outputImage = zeros(rows, cols);

    % Apply the median filter to the input image
    for i = 1:rows

        for j = 1:cols
            % Get the region of interest around the pixel
            roi = inputImage(max(1, i - (filterSize - 1) / 2):min(rows, i + (filterSize - 1) / 2), ...
                max(1, j - (filterSize - 1) / 2):min(cols, j + (filterSize - 1) / 2));

            % Reshape the ROI into a column vector and find the median
            outputImage(i, j) = median(roi(:));
        end

    end

end
