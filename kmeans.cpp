#include "kmeans.h"
#include <iostream>
#include <iomanip>
#include "timer.h"
using std::clog; using std::endl; using std::flush;

void kMeans::solve ( void ) {
   int size; MPI_Comm_size ( MPI_COMM_WORLD, &size );
   int rank; MPI_Comm_rank ( MPI_COMM_WORLD, &rank );

   iter = 0;

   // This variable counts how many points have changed label during each iteration
   int changes = stoppingCriterion.minLabelChanges + 1;
   real centroidDispl = stoppingCriterion.minCentroidDisplacement + 1;
   std::vector<point> oldCentroids;

   randomize();
   computeCentroids();

   timer tm;

   while ( (stoppingCriterion.maxIter <= 0 || iter < stoppingCriterion.maxIter)
        && (stoppingCriterion.minLabelChanges <= 0 || changes >= stoppingCriterion.minLabelChanges)
        && (stoppingCriterion.minCentroidDisplacement <= 0 || centroidDispl >= stoppingCriterion.minCentroidDisplacement) ) {

      tm.start();
      if ( rank == 0 ) clog << std::setw(8) << iter << " " << flush;

      oldCentroids = centroids;
      changes = 0;

      std::vector<unsigned int> changed;
      std::vector<int> newLabels;

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

         if ( dataset[i].getLabel() != nearestLabel ) {
            changed.push_back(i);
            newLabels.push_back(nearestLabel);
         }
      }

      tm.stop();
      if ( rank == 0 ) clog << std::setw(8) << tm.getTime() << " " << flush;
      tm.start();

      for ( int i = 0; i < size; ++i ) { // For each process
         int nchanged = changed.size();
         MPI_Bcast ( &nchanged, 1, MPI_INT, i, MPI_COMM_WORLD );

         for ( int j = 0; j < nchanged; ++j ) {
              int idx = rank == i ? changed[j] : 0;   MPI_Bcast ( &idx, 1, MPI_INT, i, MPI_COMM_WORLD );
            short lab = rank == i ? newLabels[j] : 0; MPI_Bcast ( &lab, 1, MPI_SHORT, i, MPI_COMM_WORLD );

            auto oldLabel = dataset[idx].getLabel();

            if ( oldLabel != lab ) {
               counts[oldLabel] -= 1;
               counts[lab] += 1;
               changes++;
            }

            dataset[idx].setLabel ( lab );
         }
      }

      tm.stop();
      if ( rank == 0 ) clog << std::setw(8) << tm.getTime() << " " << flush;
      tm.start();

      // Computes the centroids in the current configuration
      computeCentroids();

      tm.stop();
      if ( rank == 0 ) clog << std::setw(8) << tm.getTime() << " " << changes << endl;
      tm.start();

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

   if ( rank == 0 ) clog << "-----------------------------------------" << endl;
}
