%% Assignment 2
% Shivansh Suhane, CS383 - Machine Learning Winter 2019

%matlab setup -> clear stuff in mem
clear all;
close all;

%Save figures direcctoryy
if ~exist('figures', 'dir')  
   mkdir('figures')
   addpath('figures')
end



disp('Reading and standardizing data');
%read in the data, and then standardize it
csvData = csvread('diabetes.csv');  %csvData reads only numbers, for mixed, use importdata()
% disp(csvData);    %debug
[rows,cols] = size(csvData);
X = csvData(:,2:cols);  %features: 2->size
Y = csvData(:,1);       %class labels
Nor_X = normalize(X);
% disp(Nor_X);      %debug

disp('Seeding 0 in random generator');
rng(0);

k=6;
myKMeans(Nor_X,Y,k);    %check the mKMeans.m file for function definition