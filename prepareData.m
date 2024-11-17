function [data, labelsTrain, fileNames] = prepareData(fileData, inputSize, doAugmentation)
    load(fileData, "dataTable");
    fileNames = dataTable(:,1);
    fileNames = fileNames{:,1};
    labelsTrain = dataTable(:,2);
    labelsTrain = labelsTrain{:,1};
    miniBatchSize = 32;
    
    % Define the image augmentation scheme.
    imageAugmenter = imageDataAugmenter( ...
        RandRotation=[-45,45], ...
        RandXReflection=true);
    
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