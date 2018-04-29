#include "kmeans.h"
#include <iostream>
#include <iomanip>
#include "timer.h"
using std::clog; using std::endl; using std::flush;

void kMeans::solve ( void ) {
   int size; MPI_Comm_size ( MPI_COMM_WORLD, &size );
   int rank; MPI_Comm_rank ( MPI_COMM_WORLD, &rank );

   iter = 0;

   int changesCount = stoppingCriterion.minLabelChanges + 1;
   real centroidDispl = stoppingCriterion.minCentroidDisplacement + 1;
   std::vector<point> oldCentroids;

   randomize();
   computeCentroids();

   while ( (stoppingCriterion.maxIter <= 0 || iter < stoppingCriterion.maxIter)
        && (stoppingCriterion.minLabelChanges <= 0 || changesCount >= stoppingCriterion.minLabelChanges)
        && (stoppingCriterion.minCentroidDisplacement <= 0 || centroidDispl >= stoppingCriterion.minCentroidDisplacement) ) {

      if ( stoppingCriterion.minCentroidDisplacement > 0 )
        oldCentroids = centroids;

      // Vector of vectors of changes to be made
      // changes[k][i] = j means that the element of index j has to be set to label k
      std::vector< std::vector<int> > changes ( k, std::vector<int> () );
      changesCount = 0;

      // Assigns each point to the group of the closest centroid. The changes to
      // be made are initially stored in the vector changes, and are applied only
      // later while broadcasting them to the other processes
      for ( unsigned int i = rank; i < dataset.size(); i += size ) {
         real nearestDist = dist2 ( dataset[i], centroids[0] );
         int nearestLabel = 0;

         // Finding the nearest of the centroids
         for ( unsigned int kk = 1; kk < k; ++kk ) {
            real d = dist2 ( dataset[i], centroids[kk] );

            if ( d < nearestDist ) {
               nearestDist = d;
               nearestLabel = kk;
            }
         }

         if ( dataset[i].getLabel() != nearestLabel )
            changes[nearestLabel].push_back(i);
      }

      // Now processes communicate the changes they have made
      // Each process broadcasts for each label kk the number of points that have been
      // assigned to that label during this iteration (i.e. changes[kk].size), then
      // broadcasts the indices of those points one by one
      for ( int i = 0; i < size; ++i ) { // For each process...
         for ( unsigned int kk = 0; kk < k; ++kk ) { // ...and for each label
            int nk = changes[kk].size();
            MPI_Bcast ( &nk, 1, MPI_INT, i, MPI_COMM_WORLD );

            for ( int j = 0; j < nk; ++j ) {
               int idx = rank == i ? changes[kk][j] : 0;
               MPI_Bcast ( &idx, 1, MPI_INT, i, MPI_COMM_WORLD );

               auto oldLabel = dataset[idx].getLabel();

               changesCount++;

               counts[oldLabel] -= 1;
               counts[kk] += 1;

               dataset[idx].setLabel ( kk );
            }
         }
      }

      // Recomputes the centroids in the current configuration
      computeCentroids();

      // Compute the max displacement of the centroids
      if ( stoppingCriterion.minCentroidDisplacement > 0 ) {
         centroidDispl = 0;
         for ( unsigned kk = 0; kk < k; kk += 1 ) {
            real displ = dist2 ( oldCentroids[kk], centroids[kk] );
            if ( displ > centroidDispl ) centroidDispl = displ;
         }
         centroidDispl = sqrt(centroidDispl);
      }

      ++iter;
   }
}
