clear all
close all

% Set the path for the training image folder and limit to the first 100 images
trainImgFolder = 'train-resized';
imgFiles = dir(fullfile(trainImgFolder, '*.jpg'));

%imgFiles = imgFiles(1:min(100, numel(imgFiles))); % Only take the first 100 files

% Define a temporary datastore limited to the first 100 images
imgFileNames = fullfile({imgFiles.folder}, {imgFiles.name});
imgDataTrain = imageDatastore(imgFileNames); % Create datastore with only 100 images

% Get the filenames of label files, also limiting to the first 100 files
clsFolder = 'train';
clsFiles = dir(fullfile(clsFolder, '*.cls'));

%clsFiles = clsFiles(1:min(100, numel(clsFiles))); % Only take the first 100 files

% Check if the number of image files matches the number of label files
if numel(imgFiles) ~= numel(clsFiles)
    error('The number of label files and training images do not match. Please check.');
end

% Initialize the container for storing labels
labelsTrain = strings(numel(clsFiles), 1); % Initialize as a string array

% Read and store labels for each image
for i = 1:numel(clsFiles)
    % Check if the filenames match between the images and labels
    [~, imgName, ~] = fileparts(imgFiles(i).name);
    [~, clsName, ~] = fileparts(clsFiles(i).name);
    if ~strcmp(imgName, clsName)
        error('Filenames do not match: %s and %s', imgFiles(i).name, clsFiles(i).name);
    end

    % Read labels from each .cls file
    clsFilePath = fullfile(clsFiles(i).folder, clsFiles(i).name);
    fileID = fopen(clsFilePath, 'r');
    labels = fscanf(fileID, '%d'); % Assumes each line in .cls contains an integer label
    fclose(fileID);

    % Combine multiple labels into a single string
    labelsTrain(i) = strjoin(string(labels), ',');
end

% Convert the labelsTrain string array to a categorical type for easier handling in classification tasks
imgDataTrain.Labels = categorical(labelsTrain);


