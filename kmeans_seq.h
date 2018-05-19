#ifndef _KMEANS_SEQ_H
#define _KMEANS_SEQ_H

#include "kmeans_base.h"

// The class performs classic kmeans algorithm without parallelization
// Used for timing reference
class kMeansSeq : public kMeansBase {
public:
   kMeansSeq ( unsigned int nn, kMeansDataset::const_iterator a, kMeansDataset::const_iterator b ) :
      kMeansBase ( nn, a, b ) { }

   // Randomize and compute centroids are overridden to be without parallelization
   // Function to recompute the centroids
   void computeCentroids ( void ) override;

   // Solve method
   void solve ( void ) override;
};

void kMeansSeq::computeCentroids ( void ) {
   centroids = std::vector<point> ( k, point(n) );

   for ( auto & i : dataset ) {
      int lab = i.getLabel();
      for ( unsigned int nn = 0; nn < n; ++nn )
         centroids[lab][nn] += i[nn] / counts[lab];
   }
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

      if ( stoppingCriterion.minCentroidDisplacement > 0 )
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

         int oldLabel = dataset[i].getLabel();
         if ( oldLabel != nearestLabel ) {
            changes++;
            counts[oldLabel]--;
            counts[nearestLabel]++;
            dataset[i].setLabel(nearestLabel);
         }

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

#endif
