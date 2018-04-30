#include "kmeans_sgd.h"
#include "timer.h"
#include <iostream>
#include <iomanip>
using std::clog; using std::endl; using std::flush;

void kMeansSGD::solve ( void ) {
   int size; MPI_Comm_size ( MPI_COMM_WORLD, &size );
   int rank; MPI_Comm_rank ( MPI_COMM_WORLD, &rank );

   // Parallelization: we draw entries in batches. Each process draws a portion
   // of the batch and performs the algorithm, Size of each batch is in the
   // member batchSize

   iter = 0;

   // Randomize initial assignments
   randomize();
   computeCentroids();

   int changesCount = stoppingCriterion.minLabelChanges + 1;
   double centroidDispl = stoppingCriterion.minCentroidDisplacement + 1;
   std::vector<point> oldCentroids;

   // Counts for how many subsequent iterations the stopping criteria are met; the
   // algorithm stops when it reaches a fixed number (see below)
   // This helps checking that actual convergence takes place
   int stopIters = 0;

   std::default_random_engine eng ( 10000 * rank );
   std::uniform_int_distribution<unsigned int> distro ( 0, dataset.size() - 1 );

   std::vector<int> oldGlobalCounts ( k, 0 );
   std::vector<int> newGlobalCounts ( k, 0 );

   while ( stopIters < 15 ) {
      // Checks if stopping criterion is satisfied at this iteration, and possibly
      // increment the counter
      if ( (stoppingCriterion.maxIter <= 0 || iter < stoppingCriterion.maxIter)
        && (stoppingCriterion.minLabelChanges <= 0 || changesCount >= stoppingCriterion.minLabelChanges)
        && (stoppingCriterion.minCentroidDisplacement <= 0 || centroidDispl >= stoppingCriterion.minCentroidDisplacement) ) stopIters = 0;
      else stopIters++;

      if ( stoppingCriterion.minCentroidDisplacement > 0 )
         oldCentroids = centroids;

      changesCount = 0;

      std::vector<point> centroidDiff  ( k, point(n) );

      MPI_Allreduce ( counts.data(), oldGlobalCounts.data(), k, MPI_INT, MPI_SUM, MPI_COMM_WORLD );

      // Determines the changes to be made on the assigned portion of the batch.
      // Those changes are initially stored in the vector changes, and only later
      // sent to the other processes and applied
      for ( int i = rank; i < batchSize; i += size ) {
         // Randomly selects a point
         unsigned int idx = distro(eng);

         // Find the nearest centroid to the selected point
         int nearestLabel = 0;
         double nearestDist = dist2 ( dataset[idx], centroids[0] );

         for ( unsigned int kk = 1; kk < k; ++kk ) {
            double d = dist2 ( dataset[idx], centroids[kk] );
            if ( d < nearestDist ) {
               nearestDist = d;
               nearestLabel = kk;
            }
         }

         // Assigns the chosen label
         int oldLabel = dataset[idx].getLabel();

         if ( oldLabel != nearestLabel ) {
            counts[oldLabel] -= 1;
            counts[nearestLabel] += 1;
            dataset[idx].setLabel(nearestLabel);
            changesCount++;

            for ( unsigned int nn = 0; nn < n; ++nn ) {
               centroidDiff[oldLabel][nn] -= dataset[idx][nn];
               centroidDiff[nearestLabel][nn] += dataset[idx][nn];
            }
         }
      }

      MPI_Allreduce ( counts.data(), newGlobalCounts.data(), k, MPI_INT, MPI_SUM, MPI_COMM_WORLD );

      for ( unsigned int kk = 0; kk < k; ++kk ) {
         mpi_point_reduce ( &centroidDiff[kk] );
         for ( unsigned int nn = 0; nn < n; ++nn )
            centroids[kk][nn] = ( centroids[kk][nn] * oldGlobalCounts[kk] + centroidDiff[kk][nn] ) / newGlobalCounts[kk];
      }

      MPI_Allreduce ( MPI_IN_PLACE, &changesCount, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD );

      // Now processes communicate the changes they have made
      // Each process broadcasts for each label kk the number of points that have been
      // assigned to that label during this iteration (i.e. changes[kk].size), then
      // broadcasts the indices of those points one by one
      /* for ( int i = 0; i < size; ++i ) { // For each process...
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

               // Since with this algorithm the amount of changes will be relatively
               // small (surely <= batchSize, very likely << batchSize, especially
               // when approaching convergence) it is much more efficient to update
               // centroids with changes rather than recomputing them
               for ( unsigned int nn = 0; nn < n; ++nn ) {
                  centroids[kk][nn] += ( dataset[idx][nn] - centroids[kk][nn] ) / counts[kk];
                  centroids[oldLabel][nn] += ( centroids[oldLabel][nn] - dataset[idx][nn] ) / counts[oldLabel];
               }

               dataset[idx].setLabel ( kk );
            }
         }
      }*/

      // Compute the max displacement of the centroids for the stopping criterion
      if ( stoppingCriterion.minCentroidDisplacement > 0 ) {
         centroidDispl = 0;
         for ( unsigned kk = 0; kk < k; ++kk ) {
            double displ = dist2 ( oldCentroids[kk], centroids[kk] );
            if ( displ > centroidDispl ) centroidDispl = displ;
         }
         centroidDispl = sqrt(centroidDispl);
      }

      ++iter;
   }
}
