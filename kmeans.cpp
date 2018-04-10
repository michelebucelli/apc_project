#include "kmeans.h"
#include <iostream>
using std::clog; using std::endl;

void kMeans::solve ( void ) {
   int size; MPI_Comm_size ( MPI_COMM_WORLD, &size );
   int rank; MPI_Comm_rank ( MPI_COMM_WORLD, &rank );

   randomize();

   iter = 0;

   // This variable counts how many points have changed label during each iteration
   unsigned changes = stoppingCriterion.minLabelChanges + 1;
   real centroidDispl = stoppingCriterion.minCentroidDisplacement + 1;
   std::vector<point> oldCentroids;

   computeCentroids();

   while ( (stoppingCriterion.maxIter <= 0 || iter < stoppingCriterion.maxIter)
        && (stoppingCriterion.minLabelChanges <= 0 || changes >= stoppingCriterion.minLabelChanges)
        && (stoppingCriterion.minCentroidDisplacement <= 0 || centroidDispl >= stoppingCriterion.minCentroidDisplacement) ) {

      oldCentroids = centroids;
      changes = 0;

      // Assigns each point to the group of the closest centroid
      for ( unsigned int i = rank; i < dataset.size(); i += size ) {
         real nearestDist = dist2 ( dataset[i], centroids[0] );
         int nearestLabel = 0;

         for ( unsigned int kk = 1; kk < k; ++kk ) {
            real d = dist2 ( dataset[i], centroids[kk] );

            if ( d < nearestDist ) {
               nearestDist = d;
               nearestLabel = kk;
            }
         }

         if ( dataset[i].getLabel() != nearestLabel ) changes++;
         dataset[i].setLabel(nearestLabel);
      }

      // Collects the assignments
      for ( int i = 0; i < int(dataset.size()); ++i ) {
         if ( i % size == rank ) {
            int lab = dataset[i].getLabel();
            for ( int j = 0; j < size; ++j )
               if ( j != rank ) MPI_Send ( &lab, 1, MPI_INT, j, 0, MPI_COMM_WORLD );
         }

         else {
            int lab = 0;
            MPI_Recv ( &lab, 1, MPI_INT, i % size, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE );
            dataset[i].setLabel ( lab );
         }
      }

      // Computes the centroids in the current configuration
      computeCentroids();

      // Compute the max displacement of the centroids
      centroidDispl = 0;
      for ( unsigned kk = rank; kk < k; kk += size ) {
         real displ = dist2 ( oldCentroids[kk], centroids[kk] );
         if ( displ > centroidDispl ) centroidDispl = displ;
      }
      centroidDispl = sqrt(centroidDispl);

      // Communicates centroid displacement
      MPI_Allreduce ( MPI_IN_PLACE, &centroidDispl, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD );

      // Collects all the changes and centroid displacements flags from all processes
      MPI_Allreduce ( MPI_IN_PLACE, &changes, 1, MPI_UNSIGNED, MPI_SUM, MPI_COMM_WORLD );

      ++iter;
   }
}
