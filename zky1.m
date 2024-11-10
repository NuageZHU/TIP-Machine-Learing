% dataset_preparation.m
clear all;
close all;

% 设置路径
trainImgFolder = 'C:\Users\Utilisateur\Downloads\ms-coco\images\small-test';
clsFolder = 'C:\Users\Utilisateur\Downloads\ms-coco\labels\small-label';

% 创建图像数据集
imgDataTrain = imageDatastore(trainImgFolder, 'IncludeSubfolders', true, 'LabelSource', 'none');

% 定义标签容器
labelsTrain = {};

% 加载和处理标签文件
clsFiles = dir(fullfile(clsFolder, '*.cls'));
for i = 1:numel(clsFiles)
    clsFilePath = fullfile(clsFiles(i).folder, clsFiles(i).name);
    fileID = fopen(clsFilePath, 'r');
    labels = fscanf(fileID, '%d'); % 读取标签
    fclose(fileID);
    
    % 将多标签转换为单一字符数组，便于分类
    labelsStr = strjoin(arrayfun(@num2str, labels', 'UniformOutput', false), ',');
    labelsTrain = [labelsTrain; labelsStr];
end

% 将标签转换为分类类型并添加到图像数据集中
imgDataTrain.Labels = categorical(labelsTrain);

% 保存图像数据集和标签
save('trainDataset.mat', 'imgDataTrain');
