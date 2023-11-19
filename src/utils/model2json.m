function json = model2json(cnn)
    model.inputDim = cnn.layers{1}.outDim(1);
    layer_num = length(cnn.layers);
    model.structure = [];

    for i = 1:layer_num
        layer.type = cnn.layers{i}.type;

        switch layer.type
            case 'Conv2D'
                layer.filterDim = cnn.layers{i}.filterDim;
                layer.numFilters = cnn.layers{i}.numFilters;
                layer.poolDim = cnn.layers{i}.poolDim;
                layer.actiFunc = cnn.layers{i}.actiFunc;

                layer.hiddenUnits = -1;

            case 'Linear'
                layer.hiddenUnits = cnn.layers{i}.hiddenUnits;
                layer.actiFunc = cnn.layers{i}.actiFunc;

                layer.filterDim = -1;
                layer.numFilters = -1;
                layer.poolDim = -1;
            otherwise
                continue;
        end

        model.structure = [model.structure layer];
    end

    json = jsonencode(model);

end
