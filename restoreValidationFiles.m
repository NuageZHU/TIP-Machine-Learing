%% Remet les fichiers et labels de validation dans le dossier initial
%% Cela permet de relancer le script prepareValidationFiles.m
%% En changeant le nombre d'images de validation voulu


% Définir les chemins des dossiers
sourceImagesFolder = 'ms-coco/images/train-resized'; 
sourceLabelsFolder = 'ms-coco/labels/train';
destinationImagesFolder = 'ms-coco/images/validation';
destinationLabelsFolder = 'ms-coco/labels/validation';

% Vérifier si les dossiers source existent
if ~exist(sourceImagesFolder, 'dir') || ~exist(sourceLabelsFolder, 'dir')
    error('Un ou plusieurs dossiers source sont introuvables.');
end

% Vérifier si les dossiers de destination contiennent des fichiers
destinationImageFiles = dir(fullfile(destinationImagesFolder, '*.jpg'));
destinationLabelFiles = dir(fullfile(destinationLabelsFolder, '*.cls'));

if isempty(destinationImageFiles) && isempty(destinationLabelFiles)
    disp('Aucun fichier à restaurer dans les dossiers de destination.');
    return;
end

% Initialiser les compteurs
restoredImagesCount = 0;
restoredLabelsCount = 0;

% Restaurer les fichiers images
for i = 1:numel(destinationImageFiles)
    % Chemins complets
    destinationImage = fullfile(destinationImagesFolder, destinationImageFiles(i).name);
    sourceImage = fullfile(sourceImagesFolder, destinationImageFiles(i).name);
    
    % Déplacer le fichier
    movefile(destinationImage, sourceImage);
    restoredImagesCount = restoredImagesCount + 1; % Incrémenter le compteur
end

% Restaurer les fichiers labels
for i = 1:numel(destinationLabelFiles)
    % Chemins complets
    destinationLabel = fullfile(destinationLabelsFolder, destinationLabelFiles(i).name);
    sourceLabel = fullfile(sourceLabelsFolder, destinationLabelFiles(i).name);
    
    % Déplacer le fichier
    movefile(destinationLabel, sourceLabel);
    restoredLabelsCount = restoredLabelsCount + 1; % Incrémenter le compteur
end

% Afficher le nombre de fichiers restaurés
disp(['Restauration terminée avec succès.']);
disp(['Nombre de fichiers images restaurés : ', num2str(restoredImagesCount)]);
disp(['Nombre de fichiers labels restaurés : ', num2str(restoredLabelsCount)]);