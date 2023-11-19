function [data_train, labels_train, data_test, labels_test] = loadDataset(data_path, options)
    addpath('cnn_core');
    labels_name = {'0', '4', '7', '8', 'A', 'D', 'H'};
    data_train = [];
    labels_train = [];
    data_test = [];
    labels_test = [];
    img_dim = options.img_dim;

    if options.load_raw
        data_path = data_path + "/p_dataset_26/";
        files = dir(data_path);

        for i = 1:length(files)
            curr_file = files(i).name;

            if sum(strcmp(curr_file, labels_name)) > 0
                fullPath = fullfile(data_path, curr_file);
                filesInFolder = dir(fullPath);
                num_files = length(filesInFolder);
                num_train = floor(num_files * options.train_ratio);

                for j = 1:length(filesInFolder)
                    filename = filesInFolder(j).name;

                    if endsWith(filename, '.png')
                        img = imread(fullfile(fullPath, filename));
                        label = labelChar2Num(curr_file);

                        if j <= num_train

                            if options.apply_rand_tf && options.rand_tf.prob > rand()
                                img = randTF(img, options.rand_tf);
                            end

                            img = imresize(img, [img_dim, img_dim]);
                            img = double(img) / 255; % normalize
                            data_train = cat(3, data_train, img);
                            labels_train = cat(2, labels_train, label);
                        else
                            img = imresize(img, [img_dim, img_dim]);
                            img = double(img) / 255; % normalize
                            data_test = cat(3, data_test, img);
                            labels_test = cat(2, labels_test, label);
                        end

                    end

                end

            end

        end

    else

        for p = 1:2

            if p == 1
                path = "../data/train/";
            else
                path = "../data/test/";
            end

            files = dir(path);

            for i = 1:length(files)
                curr_file = files(i).name;

                if sum(strcmp(curr_file, labels_name)) > 0
                    fullPath = fullfile(path, curr_file);
                    filesInFolder = dir(fullPath);

                    for j = 1:length(filesInFolder)
                        filename = filesInFolder(j).name;

                        if endsWith(filename, '.png')
                            img = imread(fullfile(fullPath, filename));

                            if options.apply_rand_tf && p == 1  && options.rand_tf.prob > rand()
                                img = randTF(img, options.rand_tf);
                            end

                            img = imresize(img, [img_dim, img_dim]);
                            img = double(img) / 255; % normalize
                            label = labelChar2Num(curr_file);

                            if p == 1
                                data_train = cat(3, data_train, img);
                                labels_train = cat(2, labels_train, label);
                            else
                                data_test = cat(3, data_test, img);
                                labels_test = cat(2, labels_test, label);
                            end

                        end

                    end

                end

            end

        end

        if options.shuffle
            [data_train, labels_train] = shuffleData(data_train, labels_train);
            [data_test, labels_test] = shuffleData(data_test, labels_test);
        end

        data_train = reshape(data_train, img_dim, img_dim, 1, []);
        labels_train = permute(labels_train, [2, 1]);
        data_test = reshape(data_test, img_dim, img_dim, 1, []);
        labels_test = permute(labels_test, [2, 1]);

        if options.save
            save('../data/train.mat', 'data_train', 'labels_train');
            save('../data/test.mat', 'data_test', 'labels_test');
        end

    end
