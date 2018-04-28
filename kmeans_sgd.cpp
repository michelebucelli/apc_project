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

   std::default_random_engine eng ( rank );
   std::uniform_int_distribution<unsigned int> distro ( 0, dataset.size() - 1 );

   iter = 0;

   // Randomize initial assignments
   randomize();
   computeCentroids();

   int changes = stoppingCriterion.minLabelChanges + 1;
   real centroidDispl = stoppingCriterion.minCentroidDisplacement + 1;
   std::vector<point> oldCentroids;

   // Counts for how many subsequent iterations the stopping criteria are met; the
   // algorithm stops when it reaches a fixed number (see below)
   // This helps checking that actual convergence takes place
   int stopIters = 0;

   while ( stopIters < 15 ) {
      if ( (stoppingCriterion.maxIter <= 0 || iter < stoppingCriterion.maxIter)
        && (stoppingCriterion.minLabelChanges <= 0 || changes >= stoppingCriterion.minLabelChanges)
        && (stoppingCriterion.minCentroidDisplacement <= 0 || centroidDispl >= stoppingCriterion.minCentroidDisplacement) ) stopIters = 0;
      else stopIters++;

      if ( stoppingCriterion.minCentroidDisplacement > 0 )
         oldCentroids = centroids;

      changes = 0;

      std::vector<unsigned int> changed;
      std::vector<int> newLabels;

      for ( int i = rank; i < batchSize; i += size ) {
         unsigned int idx = distro(eng);

         // Find the nearest centroid
         int nearestLabel = 0;
         real nearestDist = dist2 ( dataset[idx], centroids[0] );

         for ( unsigned int kk = 1; kk < k; ++kk ) {
            real d = dist2 ( dataset[idx], centroids[kk] );
            if ( d < nearestDist ) {
               nearestDist = d;
               nearestLabel = kk;
            }
         }

         // Set the label of the picked point
         if ( nearestLabel != dataset[idx].getLabel() ) {
            changed.push_back(idx);
            newLabels.push_back(nearestLabel);
         }
      }

      for ( int i = 0; i < size; ++i ) { // For each process
         int nchanged = changed.size();
         MPI_Bcast ( &nchanged, 1, MPI_INT, i, MPI_COMM_WORLD );

         for ( int j = 0; j < nchanged; ++j ) {
              int idx = rank == i ? changed[j] : 0;   MPI_Bcast ( &idx, 1, MPI_INT, i, MPI_COMM_WORLD );
            short lab = rank == i ? newLabels[j] : 0; MPI_Bcast ( &lab, 1, MPI_SHORT, i, MPI_COMM_WORLD );

            auto oldLabel = dataset[idx].getLabel();

            if ( oldLabel != lab ) {
               changes++;

               counts[oldLabel] -= 1;
               counts[lab] += 1;

               for ( unsigned int nn = 0; nn < n; ++nn ) {
                  centroids[lab][nn] += ( dataset[idx][nn] - centroids[lab][nn] ) / counts[lab];
                  centroids[oldLabel][nn] += ( centroids[oldLabel][nn] - dataset[idx][nn] ) / counts[oldLabel];
               }
            }

            dataset[idx].setLabel ( lab );
         }
      }

      // Compute the max displacement of the centroids
      if ( stoppingCriterion.minCentroidDisplacement > 0 ) {
         centroidDispl = 0;
         for ( unsigned kk = 0; kk < k; ++kk ) {
            real displ = dist2 ( oldCentroids[kk], centroids[kk] );
            if ( displ > centroidDispl ) centroidDispl = displ;
         }
         centroidDispl = sqrt(centroidDispl);
      }

      ++iter;
   }
}
