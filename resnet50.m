numClasses = 80;
net = imagePretrainedNetwork("resnet50", NumClasses=numClasses);
inputSize = net.Layers(1).InputSize; % 224x224x3
categoriesTrain=0:79; % crée un tableau contenant les nombres de 0 à 79 inclus

dataFolder = fullfile(pwd, "ms-coco"); % On suppose que la base ms-coco est dans le même dossier que ce script
labelLocationTrain = fullfile(dataFolder,"labels","train");
imageLocationTrain = fullfile(dataFolder,"images", "train-resized");

[dataTrain, encodedLabels] = prepareData(labelLocationTrain,imageLocationTrain, numClasses, inputSize, true);
numObservations = dataTrain.NumObservations;

options = trainingOptions("adam", ...
    InitialLearnRate=0.0005, ...
    MiniBatchSize=32, ...
    MaxEpochs=10, ...
    Verbose= false, ...
    Plots="training-progress");


trainedNet = trainnet(dataTrain, net,"binary-crossentropy",options);



function [data, labelsTrain] = prepareData(labelLocation, imageLocation, numClasses, inputSize,doAugmentation)

miniBatchSize = 32;

imgFiles = dir(fullfile(imageLocation, '*.jpg'));

imgFiles = imgFiles(1:min(100, numel(imgFiles))); % Only take the first 100 files
% Define a temporary datastore limited to the first 100 images
imageFilename = fullfile(imageLocation, {imgFiles.name})';

% Get the filenames of label files, also limiting to the first 100 files
clsFiles = dir(fullfile(labelLocation, '*.cls'));

clsFiles = clsFiles(1:min(100, numel(clsFiles))); % Only take the first 100 files

% Check if the number of image files matches the number of label files
if numel(imgFiles) ~= numel(clsFiles)
    error('The number of label files and training images do not match. Please check.');
end

numImages = numel(clsFiles);
% Initialize variables
labelsTrain = zeros(numImages, numClasses); % Initialize a binary label matrix for multi-labels

% Read and store labels for each image
for i = 1:numImages
    % Check if the filenames match between the images and labels
    [~, imgName, ~] = fileparts(imgFiles(i).name);
    [~, clsName, ~] = fileparts(clsFiles(i).name);
    
    if ~strcmp(imgName, clsName)
        error('Filenames do not match: %s and %s', imgFiles(i).name, clsFiles(i).name);
    end
    
    % Read labels from each .cls file
    clsFilePath = fullfile(clsFiles(i).folder, clsFiles(i).name);
    fileID = fopen(clsFilePath, 'r');
    
    if fileID == -1
        error('Cannot open label file: %s', clsFilePath);
    end
    
    labels = fscanf(fileID, '%d'); % Assumes each line in .cls contains integer labels
    fclose(fileID);

    % Set corresponding indices in the label matrix to 1
    labelsTrain(i, labels + 1) = 1; % Add 1 because MATLAB indexing is 1-based
end


% Define the image augmentation scheme.
imageAugmenter = imageDataAugmenter( ...
    RandRotation=[-45,45], ...
    RandXReflection=true);

% Store the data in a table.
dataTable = table(Size=[numImages 2], ...
    VariableTypes=["string" "double"], ...
    VariableNames=["File_Location" "Labels"]);

dataTable.File_Location = imageFilename;
dataTable.Labels = labelsTrain;

% 4. Créer l'augmentedImageDatastore avec les labels multi-classes
if doAugmentation
    data = augmentedImageDatastore(inputSize(1:2), dataTable, ...
        ColorPreprocessing="gray2rgb", ...
        DataAugmentation=imageAugmenter);
else
    data = augmentedImageDatastore(inputSize(1:2), dataTable, ...
        ColorPreprocessing="gray2rgb");
end
data.MiniBatchSize = miniBatchSize;
end



