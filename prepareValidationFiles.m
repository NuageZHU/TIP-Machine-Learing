%% Génère la base de validation, reversible avec le script restoreValidationFiles.m
%% Il sépare les 65000 images du train-resized en 2 dossiers
%% un dossier validation est crée dans ms-coco/images/ contenant 13000 images par défaut
%% un dossier validation est aussi crée dans ms-coco/labels contenant les labels des images

nombreImageValidation = 13000; % 20% d'images de validation

% Définir les chemins des dossiers
sourceImagesFolder = 'ms-coco/images/train-resized';
sourceLabelsFolder = 'ms-coco/labels/train';
destinationImagesFolder = 'ms-coco/images/validation';
destinationLabelsFolder = 'ms-coco/labels/validation';

% Vérifier si les dossiers de destination contiennent déjà des fichiers
if exist(destinationImagesFolder, 'dir') && ~isempty(dir(fullfile(destinationImagesFolder, '*.jpg*')))
    disp('Le dossier de destination pour les images contient déjà des fichiers. Aucune action effectuée.');
    return;
end

if exist(destinationLabelsFolder, 'dir') && ~isempty(dir(fullfile(destinationLabelsFolder, '*.cls*')))
    disp('Le dossier de destination pour les labels contient déjà des fichiers. Aucune action effectuée.');
    return;
end

% Créer les dossiers de destination s'ils n'existent pas
if ~exist(destinationImagesFolder, 'dir')
    mkdir(destinationImagesFolder);
end
if ~exist(destinationLabelsFolder, 'dir')
    mkdir(destinationLabelsFolder);
end

% Obtenir les listes de fichiers images et labels
imageFiles = dir(fullfile(sourceImagesFolder, '*.jpg')); % Tous les fichiers images
labelFiles = dir(fullfile(sourceLabelsFolder, '*.cls')); % Tous les fichiers labels

% Vérifier qu'il y a au moins 5000 images et 5000 labels
if numel(imageFiles) < nombreImageValidation
    error('Le dossier source ne contient pas au moins 5000 fichiers image.');
end

% Boucle pour déplacer les fichiers correspondants
movedCount = 0;
for i = 1:numel(imageFiles)
    if movedCount >= nombreImageValidation
        break;
    end

    % Nom de l'image actuelle
    imageName = imageFiles(i).name;
    labelName = replace(imageName, '.jpg', '.cls'); % Remplace l'extension pour le label

    % Chemins complets des fichiers
    sourceImage = fullfile(sourceImagesFolder, imageName);
    sourceLabel = fullfile(sourceLabelsFolder, labelName);

    % Vérifier que le fichier label existe
    if exist(sourceLabel, 'file')
        % Déplacer l'image
        destinationImage = fullfile(destinationImagesFolder, imageName);
        movefile(sourceImage, destinationImage);

        % Déplacer le label
        destinationLabel = fullfile(destinationLabelsFolder, labelName);
        movefile(sourceLabel, destinationLabel);

        movedCount = movedCount + 1; % Incrémenter le compteur
    else
        warning('Label non trouvé pour l''image : %s', imageName);
    end
end

disp(['Déplacement terminé avec succès. ', num2str(movedCount), ' fichiers déplacés.']);