numClasses = 80;
net = imagePretrainedNetwork("resnet50", NumClasses=numClasses);

%On remplace la couche pleinement connectée par une autre qui a 80 sorties
% Elle s'appelle Logits sur mobilenetv2, fc1000 sur resnet50
net = replaceLayer(net, "fc1000", fullyConnectedLayer(numClasses,'Name','fullyconnected_80'));
% On remplace la couche softmax par sigmoid pour pouvoir avoir plusieurs labels par image
% Elle s'appelle Logits_softmax sur mobilenetv2, fc1000_softmax sur resnet50
net = replaceLayer(net,"fc1000_softmax", sigmoidLayer("Name", "SigmoidLayer"));

%% On freeze les paramètres sauf ceux de la dernière couche pleinement connectée
learnables = net.Learnables;
factor = 0;

numLearnables = size(learnables,1);
for i = 1:numLearnables
    layerName = learnables.Layer(i);
    parameterName = learnables.Parameter(i);
    net = setLearnRateFactor(net,layerName,parameterName,factor);
end

net = setLearnRateFactor(net,"fullyconnected_80","Weights", 1);
net = setLearnRateFactor(net,"fullyconnected_80","Bias", 1);

%% On récupère les données d'entraînement et de validation
inputSize = net.Layers(1).InputSize; % 224x224x3

[dataTrain, encodedLabelsTrain, fileNamesTrain] = prepareData("trainingData.mat", inputSize, true);
[dataValidation, encodedLabelsValidation, fileNamesValidation] = prepareData("validationData.mat", inputSize, false);
[dataTest, ~, fileNamesTest] = prepareData("testData.mat", inputSize, false);

options = trainingOptions("adam", ...
    InitialLearnRate=0.001, ...
    MiniBatchSize=32, ...
    Shuffle='every-epoch', ...
    MaxEpochs=16, ...
    Verbose= false, ...
    ValidationData=dataValidation, ...
    ValidationFrequency=5, ...
    Metrics='accuracy',...
    Plots="training-progress");

%% On lance l'entraînement
%trainedNet = trainnet(dataTrain, net,"binary-crossentropy",options);

save("trainedNetwork.mat", "trainedNet");

%% Prediction

thresholdValue = 0.5;

scores = minibatchpredict(trainedNet,dataValidation);

YPred = double(scores >= thresholdValue);

%% Calcul de la précision et du score F1
% A décommenter uniquement si on prédit sur la base de validation
% (vu qu'on connait leurs labels), sur la base de test on ne peut pas savoir

[precision, FScore, recall] = Scores(encodedLabelsValidation, YPred);

%% Génère le fichier JSON avec le format attendu par le prof

generateJson(fileNamesValidation, YPred);


%% Fonctions utiles

function [precision, F1, recall] = Scores(T,Y)
    % TP: True Positive
    % FP: False Positive
    % TN: True Negative
    % FN: False Negative
    
    TP = sum(T .* Y,"all");
    FP = sum(Y,"all")-TP;
    
    TN = sum(~T .* ~Y,"all");
    FN = sum(~Y,"all")-TN;
    
    F1 = TP/(TP + 0.5*(FP+FN));
    precision = TP/(TP+FP);
    recall = TP/(TP+FN);
end


function [] = generateJson(filesName, labels)
    jsonMap = containers.Map;
    [lignes, ~] = size(filesName);

    for i=1:lignes
        [~, fileName, ~] = fileparts(filesName(i));
        labelIndices = find(labels(i,:) == 1);

        labelIndices = double(labelIndices);

        labelIndices = labelIndices - 1;
        
        % Si on a une seule valeur, on la met dans un tableau
        if length(labelIndices) == 1
            labelIndices = num2cell(labelIndices);
        end

        % Stocke le tableau d'indices dans la Map
        jsonMap(fileName) = labelIndices;
    end
    jsonStr = jsonencode(jsonMap);
    % Sauvegarder le JSON dans un fichier
    fid = fopen('predictions.json', 'w');
    if fid == -1
        error('Impossible d''ouvrir le fichier pour écrire.');
    end
    fprintf(fid, jsonStr);
    fclose(fid);
    disp('Fichier JSON créé avec succès.');
end


