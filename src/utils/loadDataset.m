function [images_train, labels_train, images_test, labels_test] = loadDataset(data_path, options)
    addpath('cnn_core');
    labels_name = {'0', '4', '7', '8', 'A', 'D', 'H'};
    images_train = [];
    labels_train = [];
    images_test = [];
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
                        % apply random transformation
                        if options.transform
                            img = randomTrans(img);
                        end

                        img = imresize(img, [img_dim, img_dim]);
                        img = double(img) / 255; % normalize

                        % img = rgb2gray(img);
                        % img = imbinarize(img);
                        % img = reshape(img,1,[]);
                        label = labelChar2Num(curr_file);

                        if j <= num_train
                            images_train = cat(3, images_train, img);
                            labels_train = cat(2, labels_train, label);
                        else
                            images_test = cat(3, images_test, img);
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

                            if options.transform
                                img = randomTrans(img);
                            end

                            img = imresize(img, [img_dim, img_dim]);
                            img = double(img) / 255; % normalize
                            label = labelChar2Num(curr_file);

                            if p == 1
                                images_train = cat(3, images_train, img);
                                labels_train = cat(2, labels_train, label);
                            else
                                images_test = cat(3, images_test, img);
                                labels_test = cat(2, labels_test, label);
                            end

                        end

                    end

                end

            end

        end

        if options.shuffle
            [images_train, labels_train] = shuffleData(images_train, labels_train);
            [images_test, labels_test] = shuffleData(images_test, labels_test);
        end

        images_train = reshape(images_train, img_dim, img_dim, 1, []);
        labels_train = permute(labels_train, [2, 1]);
        images_test = reshape(images_test, img_dim, img_dim, 1, []);
        labels_test = permute(labels_test, [2, 1]);

        if options.save
            save('../data/train.mat', 'images_train', 'labels_train');
            save('../data/test.mat', 'images_test', 'labels_test');
        end

    end
