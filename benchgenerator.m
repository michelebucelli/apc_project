clear all;
close all;
clc;

% Number of points per cluster
N = 5e4;

% Number of clusters
k = 5;

% Dimension of the space
dim = 10;

% Cluster centers [x y]
mu = unifrnd ( -10, 10, k, dim );

% Cluster std deviations
sigma = unifrnd ( 2, 6, k, 1 );

% Clusters
dataset = [];

for i = 1:k
    cluster = [];
    for j = 1:dim
      cluster = [ cluster, normrnd(mu(i,j),sigma(i),N,1) ];
    end

    dataset = [ dataset; cluster ];
end

% Save dataset
datasetFile = fopen ( './benchmark.txt', 'w' );
trueFile = fopen ( './benchmark-truelabels.txt', 'w' );
fprintf ( datasetFile, '%d\n', dim );
for i = 1:(N*k)
    for j = 1:dim
      fprintf ( datasetFile, '%f ', dataset(i,j) );
    end
    fprintf ( datasetFile, '\n' );
    fprintf ( trueFile, '%d\n', 1+floor(i/N) );
end

fclose(datasetFile);
fclose(trueFile);
