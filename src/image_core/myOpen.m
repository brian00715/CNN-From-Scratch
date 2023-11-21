function result = myOpen(image, se)
    % image: Input image
    % se: Structuring element

    % Erode first
    erodedImage = myErode(image, se);

    % Then dilate
    result = myDilate(erodedImage, se);
end
