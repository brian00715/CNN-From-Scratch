% @Description: Learning rate schedule.
% @Author: Kuankuan Sima
% @Email: kuankuan_sima@u.nus.edu
function [lr] = lrSchedule(curr_epoch, total_epoch, options)
    % Cosine annealing learning rate schedule
    lr_max = options.lr_max;
    lr_min = options.lr_min;

    switch options.lr_method
        case 'cosine'
            lr = lr_min + 0.5 * (lr_max - lr_min) * (1 + cos(pi * curr_epoch / total_epoch));
        case 'linear'
            lr = lr_max - (lr_max - lr_min) * curr_epoch / total_epoch;
        case 'cosine_cyclic'
            cycle_length = options.lr_duty;
            relative_pos = mod(curr_epoch, cycle_length) / cycle_length;
            lr = lr_min + 0.5 * (lr_max - lr_min) * (1 + cos(pi * relative_pos));
        case 'step'
            lr = lr_max * 0.1 ^ (floor(curr_epoch / 30));
        case 'exp'
            lr = lr_max * exp(-0.1*curr_epoch);
        case 'const'
            lr = lr_max;
        case 'fixed'
            lr = lr_max;
        otherwise
            error('Unknown learning rate schedule %s', options.lr_decay);
    end

end
