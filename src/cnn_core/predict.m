function [preds,cnn] = predict(cnn, X)
    local_option.train_mode = false;
    cnn = forward(cnn, X, local_option);
    [~, preds] = max(cnn.layers{end}.activations, [], 1);
    preds = preds';
