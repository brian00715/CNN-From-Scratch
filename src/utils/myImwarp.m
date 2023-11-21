function outputImage = myImwarp(inputImage, transformMatrix, outputSize)
    % Input:
    %   inputImage: Input image
    %   transformMatrix: 2x3 affine transformation matrix
    %   outputSize: Size of the output image [rows, cols]

    [rows, cols] = size(inputImage);

    % Generate the output image
    outputImage = zeros(outputSize);

    % Loop through each pixel in the output image
    for row = 1:outputSize(1)

        for col = 1:outputSize(2)
            % Calculate the coordinates in the input image
            inputCoord = transformMatrix * [row; col; 1];
            inputCoord = round(inputCoord ./ inputCoord(3)); % Normalize and round

            % Check if the input coordinates are within the input image range
            if inputCoord(1) >= 1 && inputCoord(1) <= rows && inputCoord(2) >= 1 && inputCoord(2) <= cols
                % Use bilinear interpolation to calculate the pixel value in the output image
                outputImage(row, col) = bilinearInterpolation(inputImage, inputCoord(1), inputCoord(2));
            end

        end

    end

end

function value = bilinearInterpolation(image, x, y)
    % Bilinear interpolation function
    x1 = floor(x);
    x2 = x1 + 1;
    y1 = floor(y);
    y2 = y1 + 1;

    if x1 < 1 || x2 > size(image, 1) || y1 < 1 || y2 > size(image, 2)
        value = 0; % Return 0 if coordinates are outside the image range
    else
        % Bilinear interpolation
        Q11 = image(x1, y1);
        Q12 = image(x1, y2);
        Q21 = image(x2, y1);
        Q22 = image(x2, y2);

        value = (Q11 * (x2 - x) * (y2 - y) + Q21 * (x - x1) * (y2 - y) + ...
            Q12 * (x2 - x) * (y - y1) + Q22 * (x - x1) * (y - y1)) / ((x2 - x1) * (y2 - y1));
    end

end
