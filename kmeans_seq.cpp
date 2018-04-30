#include "kmeans_seq.h"
#include <iostream>
using std::clog; using std::endl;

void kMeansSeq::computeCentroids ( void ) {
   centroids = std::vector<point> ( k, point(n) );

   for ( auto & i : dataset ) {
      int lab = i.getLabel();
      centroids[lab] += i;
   }

   for ( unsigned int kk = 0; kk < k; ++kk )
      centroids[kk] = centroids[kk] / counts[kk];
}

void kMeansSeq::solve ( void ) {
   randomize();

   iter = 0;

   // This variable counts how many points have changed label during each iteration
   int changes = stoppingCriterion.minLabelChanges + 1;
   double centroidDispl = stoppingCriterion.minCentroidDisplacement + 1;
   std::vector<point> oldCentroids;

   computeCentroids();

   while ( (stoppingCriterion.maxIter <= 0 || iter < stoppingCriterion.maxIter)
        && (stoppingCriterion.minLabelChanges <= 0 || changes >= stoppingCriterion.minLabelChanges)
        && (stoppingCriterion.minCentroidDisplacement <= 0 || centroidDispl >= stoppingCriterion.minCentroidDisplacement) ) {

      oldCentroids = centroids;
      changes = 0;

      // Assigns each point to the group of the closest centroid
      for ( unsigned int i = 0; i < dataset.size(); i++ ) {
         double nearestDist = dist2 ( dataset[i], centroids[0] );
         int nearestLabel = 0;

         for ( unsigned int kk = 1; kk < k; ++kk ) {
            double d = dist2 ( dataset[i], centroids[kk] );

            if ( d < nearestDist ) {
               nearestDist = d;
               nearestLabel = kk;
            }
         }

         if ( dataset[i].getLabel() != nearestLabel ) {
            changes++;
            counts[dataset[i].getLabel()]--;
            counts[nearestLabel]++;
         }

         dataset[i].setLabel(nearestLabel);
      }

      // Computes the centroids in the current configuration
      computeCentroids();

      // Compute the max displacement of the centroids
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
