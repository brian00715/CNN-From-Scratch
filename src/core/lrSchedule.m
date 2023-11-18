function [lr] = lrSchedule(curr_epoch, total_epoch, options)
    % Cosine annealing learning rate schedule
    lr_max = options.lr_max;
    lr_min = options.lr_min;

    switch options.method
        case 'cosine'
            lr = lr_min + 0.5 * (lr_max - lr_min) * (1 + cos(pi * curr_epoch / total_epoch));
        case 'linear'
            lr = lr_max - (lr_max - lr_min) * curr_epoch / total_epoch;
        case 'cosine_cyclic'
            lr = lr_min + 0.5 * (lr_max - lr_min) * (1 + cos(pi * curr_epoch / total_epoch));

            if mod(curr_epoch, total_epoch / 3) == 0
                lr_max = lr_max * 0.1;
                lr_min = lr_min * 0.1;
            end

        case 'step'
            lr = lr_max * 0.1 ^ (floor(curr_epoch / 30));
        case 'exp'
            lr = lr_max * 0.95 ^ (curr_epoch);
        case 'const'
            lr = lr_max;
        otherwise
            error('Unknown learning rate schedule %s', options.lr_decay);
    end

end
