#include "kmeans_sdg.h"
#include <ctime>

void kMeansSDG::solve ( void ) {
   std::default_random_engine eng(std::time(NULL));
   std::uniform_int_distribution<unsigned int> distro ( 0, dataset.size() - 1 );

   unsigned int iter = 0;

   // Count of elements in each class
   std::vector<unsigned int> counts ( k, 0 );

   // Initialize centroids with random elements in the dataset
   for ( unsigned int kk = 0; kk < k; ++kk )
      centroids[kk] = dataset[distro(eng)];

   while ( iter < 1e5 ) {
      // Pick a random entry in the dataset
      unsigned int idx = distro(eng);

      // Find the nearest centroid
      unsigned int nearestLabel = 0;
      real nearestDist = dist2 ( dataset[idx], centroids[0] );

      for ( unsigned int kk = 1; kk < k; ++kk ) {
         real d = dist2 ( dataset[idx], centroids[kk] );
         if ( d < nearestDist ) {
            nearestDist = d;
            nearestLabel = kk;
         }
      }

      if ( nearestLabel != dataset[idx].getLabel() ) {
         // Set the label of the picked point
         dataset[idx].setLabel(nearestLabel);
         // Update counts and centroids
         counts[nearestLabel] += 1;

         point diff = (dataset[idx] - centroids[nearestLabel]) / counts[nearestLabel];
         centroids[nearestLabel] += diff;
      }
      
      ++iter;
   }
}
