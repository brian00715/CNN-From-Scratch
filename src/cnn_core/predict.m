function preds = predict(cnn, X,options)

    % predict: Predict labels of test set
    % ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    %   preds = predict(cnn,X)
    %    ---------------------------------------------------------------------------------
    %    Arguments:
    %           cnn         - a cnn whose weights are initialized or specified
    %           X           - test data. Should be M*N*D*NUM matrix, where
    %                         a single image is of size M*N*D and NUM specifies
    %                         numbers of test data
    %    Return:
    %           preds       - predictions of test set
    %    ---------------------------------------------------------------------------------
    % cnn structure
    %   layers: layers of the cnn
    %       type:                       type of the layer, could be input layer ('input'), convolutional
    %                                   and subsampling layer ('Conv2D'), full connected layer ('Linear'),
    %                                   and output layer ('output').
    %
    %       filterDim:                  dimension of filter, convolutional and
    %                                   subsampling layer ('Conv2D') only, and real
    %                                   filter size is filterDim*filterDim*k
    %                                   where k specifies the numbers of
    %                                   feature map.
    %
    %       numFilters:                 numbers of filters, convolutional and
    %                                   subsampling layer ('Conv2D') only
    %
    %       poolDim:                    pool dimension, convolutional and
    %                                   subsampling layer ('Conv2D') only
    %
    %       hiddenUnits                 hidden units, full connected layer
    %                                   ('Linear') and output layer ('output') only
    %
    %       actiFunc:         name of activation function, could be
    %                                   'sigmoid', 'relu' and 'tanh', default
    %                                   is 'sigmoid'
    %
    %       realActiFunc:     function handle of activation function
    %
    %       realGradFunc:       function handle of the gradients of the
    %                                   activation function
    %
    %       outDim:                     output dimension
    %
    %       W:                          weights
    %
    %       b:                          bias
    %
    %       convolvedFeatures:          convolved features
    %
    %       activations:                'input' of the next layer
    %
    %       delta:                      sensitivities
    %
    %       Wgrad:                      gradients of weights
    %
    %       bgrad:                      gradients of bias
    %
    %       softmax                     if 1, implement softmax in output
    %                                   layer, output layer ('output') only
    
    local_option = options;
    local_option.train_mode = false;
    cnn = forward(cnn, X,local_option);
    [~, preds] = max(cnn.layers{end}.activations, [], 1);
    preds = preds';
