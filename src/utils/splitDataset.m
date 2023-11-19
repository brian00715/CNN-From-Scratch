data_path = '../data/p_dataset_26/';
labels_name = {'0', '4', '7', '8', 'A', 'D'};
files = dir(data_path);
train_ratio = 0.8;

for i = 1:length(files)
    curr_file = files(i).name;

    if sum(strcmp(curr_file, labels_name)) > 0
        fullPath = fullfile(data_path, curr_file);
        filesInFolder = dir(fullPath);
        num_files = length(filesInFolder);
        num_train = floor(num_files * train_ratio);

        for j = 1:length(filesInFolder)
            filename = filesInFolder(j).name;

            if endsWith(filename, '.png')
                sourcePath = fullfile(fullPath, filename);

                if j <= num_train
                    destinationPath = fullfile("../data/train/"+curr_file); 

                    if ~exist(destinationPath, 'dir')
                        mkdir(destinationPath);
                    end

                    dest_file_path = destinationPath + "/"+filename;

                    copyfile(sourcePath, destinationPath);
                else
                    destinationPath = fullfile("../data/test/"+curr_file);

                    if ~exist(destinationPath, 'dir')
                        mkdir(destinationPath);
                    end

                    dest_file_path = destinationPath + "/"+filename;

                    copyfile(sourcePath, destinationPath);
                end

            end

        end

    end

end
