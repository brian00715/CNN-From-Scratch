function dilatedImage = myDilate(inputImage, se)
    % Function to perform grayscale dilation on an image using a structuring element
    se = se.Neighborhood;
    % Get the size of the input image
    [rows, cols] = size(inputImage);

    % Get the size of the structuring element
    [seRows, seCols] = size(se);

    % Initialize the dilated image
    dilatedImage = zeros(rows, cols);

    % Loop through each pixel in the input image
    for i = 1:rows

        for j = 1:cols
            % Overlay the structuring element on the input image
            neighborhood = zeros(seRows, seCols);

            for m = 1:seRows

                for n = 1:seCols
                    % Calculate the coordinates in the input image
                    x = i - (seRows - 1) / 2 + m - 1;
                    y = j - (seCols - 1) / 2 + n - 1;

                    % Check if the coordinates are within the bounds of the input image
                    if x >= 1 && x <= rows && y >= 1 && y <= cols
                        % Store the intensity values in the neighborhood
                        neighborhood(m, n) = inputImage(x, y);
                    end

                end

            end

            % Compute the maximum intensity in the neighborhood
            dilatedImage(i, j) = max(neighborhood(:));
        end

    end

end
