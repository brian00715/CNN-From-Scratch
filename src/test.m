% function lr= lrSchedule(lr_min,lr_max,curr_epoch,total_epoch,options)
% test lrSchedule

% total_epoch = 30;
% lr = [];
% options.lr_max = 0.01;
% options.lr = options.lr_max;
% options.lr_min = 1e-8;
% options.lr_method = 'cosine_cyclic';
% options.lr_duty = 5; % duty cycle for cosine lr
% 
% for curr_epoch = 1:total_epoch
%     curr_lr = lrSchedule(curr_epoch, total_epoch, options);
%     lr = [lr curr_lr];
% end
% 
% plot(lr);
% xlabel('epoch');
% ylabel('learning rate');
% title('lrSchedule test');

data_path = "../data/p_dataset_26/";
dataset_option.shuffle = true;
dataset_option.img_dim = 124;
dataset_option.train_ratio = 0.8;
dataset_option.save = true;
dataset_option.transform = true;
[images_train, labels_train, images_test, labels_test] = loadDataset(data_path, dataset_option);

