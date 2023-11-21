% Description: This script is used to apply Gaussian filtering to an image
% Input:       im0 - the original image, must be a grayscale image
%              n - the size of the Gaussian filter
%              sigma - the standard deviation of the Gaussian filter
% Output:      im_filtered - the filtered image
function im_filtered = myGaussianFilter(im0, n, sigma)

    %------------------------ 2D Gaussian Filter --------------------------
    coreGauss2 = my2DGaussianFilter(n, sigma);

    % Apply 2D Gaussian filtering to the noisy image
    im_filtered = myConvolution(im0, coreGauss2);

    %------------------------ 1D Gaussian Filter --------------------------
    if 0
        % Horizontal Gaussian Filter
        coreGauss1_horizontal = my1DGaussianFilter(n, sigma);

        % Apply horizontal Gaussian filtering
        im_Gauss1_horizontal = myConvolution(im0, coreGauss1_horizontal);

        % Vertical Gaussian Filter
        coreGauss1_vertical = my1DGaussianFilter(n, sigma);

        % Apply vertical Gaussian filtering
        im_filtered = myConvolution(im_Gauss1_horizontal, coreGauss1_vertical);
    end

end

function coreGauss2 = my2DGaussianFilter(n, sigma)
    coreGauss2 = ones(n, n);
    delta = ceil(n / 2);

    for i = 1:n

        for j = 1:n
            coreGauss2(i, j) = exp(- (power(i - delta, 2) + power(j - delta, 2)) / (2 * sigma * sigma));
        end

    end

    % Normalize the filter
    c = 1 / coreGauss2(1, 1);
    totalWeight = 0;

    for i = 1:n

        for j = 1:n
            coreGauss2(i, j) = round(coreGauss2(i, j) * c);
            totalWeight = totalWeight + coreGauss2(i, j);
        end

    end

    coreGauss2 = coreGauss2 / totalWeight;
end

function coreGauss1 = my1DGaussianFilter(n, sigma)
    % Create a 1D Gaussian filter kernel
    coreGauss1 = zeros(1, n);
    delta = ceil(n / 2);

    for i = 1:n
        coreGauss1(i) = exp(- (power(i - delta, 2) / (2 * sigma * sigma)));
    end

    % Normalize the filter
    c = 1 / coreGauss1(1);
    totalWeight = 0;

    for i = 1:n
        coreGauss1(i) = round(coreGauss1(i) * c);
        totalWeight = totalWeight + coreGauss1(i);
    end

    coreGauss1 = coreGauss1 / totalWeight;
end
