
clc;
clear all;
close all;

%read data: reference: https://www.mathworks.com/help/matlab/ref/importdata.html
X_train = importdata('X_train.mat');
y_train = importdata('y_train.mat');
X_test = importdata('X_test.mat');
y_test = importdata('y_test.mat');

%Construct the classifier using fitcknn: https://www.mathworks.com/help/stats/fitcknn.html
Mdl = fitcknn(X_train,y_train,'NumNeighbors',7,'Distance','euclidean');

%predict the class: https://www.mathworks.com/help/stats/compactclassificationdiscriminant.predict.html
predictedClass = predict(Mdl, X_test);

%count accuracy
accuracy = sum(y_test == predictedClass)/length(predictedClass);
accuracyPercentage = 100*accuracy;
fprintf('Accuracy = %f%%\n',accuracyPercentage)






