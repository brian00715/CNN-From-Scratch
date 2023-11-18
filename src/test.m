% function lr= lrSchedule(lr_min,lr_max,curr_epoch,total_epoch,options)
% test lrSchedule

total_epoch = 50;
lr = [];
options.method = "cosine";
options.lr_min = 1e-4;
options.lr_max = 0.1;

for curr_epoch = 1:total_epoch
    curr_lr = lrSchedule(curr_epoch, total_epoch, options);
    lr = [lr curr_lr];
end

plot(lr);
xlabel('epoch');
ylabel('learning rate');
title('lrSchedule test');
