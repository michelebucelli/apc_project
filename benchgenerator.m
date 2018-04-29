clear all;
close all;

N = 4e5; % Number of points per cluster
k = 5; % Number of clusters
dim = 20; % Dimension of the space
mu = unifrnd ( -10, 10, k, dim ); % Cluster centers [x1 x2 x3 ... xn]
sigma = unifrnd ( 2, 6, k, 1 ); % Cluster std deviations

fprintf ( 'Generating dataset for clustering benchmark\n' );
fprintf ( 'Number of points:   %d\n', N*k );
fprintf ( 'Point dimensions:   %d\n', dim );
fprintf ( 'Number of clusters: %d\n', k );
fprintf ( 'Progress: 0%%' );
nchar = 2;

datasetFile = fopen ( './benchmark.txt', 'w' );
trueFile = fopen ( './benchmark-truelabels.txt', 'w' );

% Prints the dimension of the points on the first line as required by the clustering
% program input format
fprintf ( datasetFile, '%d\n', dim );

for i = 1:k % For each cluster ...
   for j = 1:N % ... and each point in the cluster ...
      for l = 1:dim % ... and each dimension of that point
         fprintf ( datasetFile, '%f', normrnd( mu(i,l), sigma(i) ) );
      end
      fprintf ( trueFile, '%d\n', i );

      if ( mod( (i-1)*N+j-1, N*k/1000 ) == 0 )
         fprintf ( repmat('\b', nchar ) );
         str = sprintf ( '%.1f%%', ((i-1)*N+j-1)/(N*k)*100 );
         nchar = length(str);
         fprintf ( '%s', str );
      end
   end
end

fclose(datasetFile);
fclose(trueFile);
