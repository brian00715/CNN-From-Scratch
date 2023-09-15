% Author: SIMA KUANKUAN
% Input: gray image
% Return: best binary threshold
function best_threshold = otsu(img_gray)

    % Initialize variables
    num_bins = 256; % Number of grayscale levels
    histogram = imhist(img_gray, num_bins); % Compute the grayscale histogram
    total_pixels = numel(img_gray);
    sum_intensity = sum((0:(num_bins - 1))' .* histogram);

    % Initialize the best threshold and maximum variance
    best_threshold = 0;
    max_variance = 0;

    % Calculate intra-class variance and inter-class variance
    for threshold = 1:num_bins
        % Class 1
        w1 = sum(histogram(1:threshold)) / total_pixels;
        mu1 = sum((0:(threshold - 1))' .* histogram(1:threshold)) / (w1 * total_pixels);
        var1 = sum(((0:(threshold - 1)) - mu1) .^ 2 .* histogram(1:threshold)) / (w1 * total_pixels);

        % Class 2
        w2 = sum(histogram((threshold + 1):end)) / total_pixels;
        mu2 = (sum_intensity - sum((0:(threshold - 1))' .* histogram(1:threshold))) / (w2 * total_pixels);
        var2 = (sum((0:(num_bins - 1))' .* histogram) - sum((0:(threshold - 1))' .* histogram(1:threshold))) / (w2 * total_pixels);

        % Intra-class variance and inter-class variance
        intra_class_variance = w1 * var1 + w2 * var2;
        inter_class_variance = w1 * w2 * (mu1 - mu2) ^ 2;

        % Update the maximum variance and best threshold
        if inter_class_variance >= max_variance % minimize intra-class var equales to maximize inter-class var
            max_variance = inter_class_variance;
            best_threshold = threshold - 1;
        end

    end

    best_threshold = best_threshold / 256;
end
