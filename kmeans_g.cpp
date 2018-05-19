#include "kmeans_g.h"
#include <iostream>
#include <iomanip>
#include "timer.h"
using std::clog; using std::endl; using std::flush;

void kMeansG::solve ( void ) {
   int size; MPI_Comm_size ( MPI_COMM_WORLD, &size );
   int rank; MPI_Comm_rank ( MPI_COMM_WORLD, &rank );

   iter = 0;

   int changesCount = stoppingCriterion.minLabelChanges + 1;
   double centroidDispl = stoppingCriterion.minCentroidDisplacement + 1;
   std::vector<point> oldCentroids;

   randomize();
   computeCentroids();

   // Vector of vectors of changes to be made
   // changes[k][i] = j means that the element of index j has to be set to label k
   // std::vector< std::vector<int> > changes ( k, std::vector<int> () );

   while ( (stoppingCriterion.maxIter <= 0 || iter < stoppingCriterion.maxIter)
        && (stoppingCriterion.minLabelChanges <= 0 || changesCount >= stoppingCriterion.minLabelChanges)
        && (stoppingCriterion.minCentroidDisplacement <= 0 || centroidDispl >= stoppingCriterion.minCentroidDisplacement) ) {

      if ( stoppingCriterion.minCentroidDisplacement > 0 )
        oldCentroids = centroids;

      // for ( auto & v : changes ) v.clear();
      changesCount = 0;

      // Assigns each point to the group of the closest centroid. The changes to
      // be made are initially stored in the vector changes, and are applied only
      // later while broadcasting them to the other processes

      double nearestDist = 0, d = 0;
      int nearestLabel = 0;

      for ( unsigned int i = 0; i < dataset.size(); i += 1 ) {
         nearestDist = dist2 ( dataset[i], centroids[0] );
         nearestLabel = 0;

         // Finding the nearest of the centroids
         for ( unsigned int kk = 1; kk < k; ++kk ) {
            d = dist2 ( dataset[i], centroids[kk] );

            if ( d < nearestDist ) {
               nearestDist = d;
               nearestLabel = kk;
            }
         }

         int oldLabel = dataset[i].getLabel();
         if ( oldLabel != nearestLabel ) {
            counts[oldLabel] -= 1;
            counts[nearestLabel] += 1;
            dataset[i].setLabel(nearestLabel);
            changesCount++;
         }
      }

      // Recomputes the centroids in the current configuration
      computeCentroids();
      MPI_Allreduce ( MPI_IN_PLACE, &changesCount, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD );

      // Compute the max displacement of the centroids
      if ( stoppingCriterion.minCentroidDisplacement > 0 ) {
         centroidDispl = 0;
         for ( unsigned kk = 0; kk < k; kk += 1 ) {
            double displ = dist2 ( oldCentroids[kk], centroids[kk] );
            if ( displ > centroidDispl ) centroidDispl = displ;
         }
         centroidDispl = sqrt(centroidDispl);
      }

      ++iter;
   }

}
