clear all;
close all;
clc;

% Number of points per cluster
N = 5e4;

% Number of clusters
k = 5;

% Cluster centers [x y]
mu = [ 0    0;
       10   0;
       0   10;
       20  20;
       -10 -5 ];
   
% Cluster std deviations
sigma = [ 1 2 2 3 1 ];

% Clusters
dataset = [];

for i = 1:k
    cluster = [ normrnd(mu(i,1),sigma(i), [N, 1]) normrnd(mu(i,2),sigma(i), [N, 1]) ];
    dataset = [ dataset; cluster ];
end

% Save dataset
datasetFile = fopen ( "./benchmark.txt", "w" );
trueFile = fopen ( "./benchmark-truelabels.txt", "w" );
fprintf ( datasetFile, "2" );
for i = 1:(N*k)
    fprintf ( datasetFile, "%f %f\n", dataset(i,1), dataset(i,2) );
    fprintf ( trueFile, "%d\n", 1+floor(i/N) );
end

fclose(datasetFile);
fclose(trueFile);