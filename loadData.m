%% Il faut executer ce fichier avant d'exécuter mobilenetv2.m
%% Ce fichier charge les données d'entraînement et de validation
%% Si vous n'avez pas encore crée de base de validation,
%% executez le fichier prepareValidationFiles.m
%% IMPORTANT : La base MS-COCO doit être dans le même dossier que les scripts


dataFolder = fullfile(pwd, "ms-coco"); % On suppose que la base ms-coco est dans le même dossier que ce script

% On charge les données d'entraînement
numberFilesTraining = 2000; % Nombre d'images d'entraînement utilisés
labelLocationTraining = fullfile(dataFolder,"labels","train");
imageLocationTraining = fullfile(dataFolder,"images", "train-resized");
fileSaveTraining = "trainingData.mat";
varSaveTraining = "dataTableTraining";

% On charge les données de validation
numberFilesValidation = 200; % Nombre d'images de validation utilisées
labelLocationValidation = fullfile(dataFolder,"labels","validation");
imageLocationValidation = fullfile(dataFolder,"images", "validation");
fileSaveValidation = "validationData.mat";
varSaveValidation = "dataTableValidation";

loadDataTable(fileSaveTraining, varSaveTraining, labelLocationTraining, imageLocationTraining, numberFilesTraining);
loadDataTable(fileSaveValidation, varSaveValidation, labelLocationValidation, imageLocationValidation, numberFilesValidation);


function [] = loadDataTable(fileSaveName, varSaveName, labelLocation, imageLocation, numberFiles)
    imgFiles = dir(fullfile(imageLocation, '*.jpg'));
    
    imgFiles = imgFiles(1:min(numberFiles, numel(imgFiles))); % Only take the first numberFiles files
    % Define a temporary datastore limited to the first 100 images
    imageFilename = fullfile(imageLocation, {imgFiles.name})';
    
    % Get the filenames of label files, also limiting to the first 100 files
    clsFiles = dir(fullfile(labelLocation, '*.cls'));
    
    clsFiles = clsFiles(1:min(numberFiles, numel(clsFiles))); % Only take the first numberFiles files
    
    % Check if the number of image files matches the number of label files
    if numel(imgFiles) ~= numel(clsFiles)
        error('The number of label files and training images do not match. Please check.');
    end
    
    % Initialize variables
    labelsTrain = zeros(numberFiles, 80); % Initialize a binary label matrix for multi-labels
    
    % Read and store labels for each image
    for i = 1:numberFiles
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
        % Store the data in a table.
    dataTable = table(Size=[numberFiles 2], ...
        VariableTypes=["string" "double"], ...
        VariableNames=["File_Location" "Labels"]);
    
    dataTable.File_Location = imageFilename;
    dataTable.Labels = labelsTrain;
    
    save(fileSaveName, "dataTable");
end