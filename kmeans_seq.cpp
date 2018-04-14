#include "kmeans_seq.h"
#include <iostream>
using std::clog; using std::endl;

void kMeansSeq::computeCentroids ( void ) {
   centroids = std::vector<point> ( k, point(n) );

   for ( auto & i : dataset ) {
      int lab = i.getLabel();
      centroids[lab] += i;
   }

   for ( unsigned int kk = 0; kk < k; ++kk ) {
      if ( counts[kk] > 0 ) centroids[kk] = centroids[kk] / counts[kk];

      unsigned int nearestIdx = 0;
      real nearestDist = dist2 ( dataset[0], centroids[kk] );

      for ( unsigned int i = 1; i < dataset.size(); ++i ) {
         real d = dist2 ( dataset[i], centroids[kk] );
         if ( d < nearestDist ) {
            nearestIdx = i;
            nearestDist = d;
         }
      }

      centroids[kk] = dataset[nearestIdx];
   }
}

void kMeansSeq::randomize ( void ) {
   std::default_random_engine eng;
   std::uniform_int_distribution<unsigned int> dist ( 0, k - 1 );

   counts = std::vector<int> ( k, 0 );
   for ( auto & i : dataset ) {
      unsigned int lab = dist(eng);
      i.setLabel ( lab );
      counts[lab]++;
   }
}

void kMeansSeq::solve ( void ) {
   randomize();

   iter = 0;

   // This variable counts how many points have changed label during each iteration
   int changes = stoppingCriterion.minLabelChanges + 1;
   real centroidDispl = stoppingCriterion.minCentroidDisplacement + 1;
   std::vector<point> oldCentroids;

   computeCentroids();

   while ( (stoppingCriterion.maxIter <= 0 || iter < stoppingCriterion.maxIter)
        && (stoppingCriterion.minLabelChanges <= 0 || changes >= stoppingCriterion.minLabelChanges)
        && (stoppingCriterion.minCentroidDisplacement <= 0 || centroidDispl >= stoppingCriterion.minCentroidDisplacement) ) {

      oldCentroids = centroids;
      changes = 0;

      // Assigns each point to the group of the closest centroid
      for ( unsigned int i = 0; i < dataset.size(); i++ ) {
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
            real displ = dist2 ( oldCentroids[kk], centroids[kk] );
            if ( displ > centroidDispl ) centroidDispl = displ;
         }
         centroidDispl = sqrt(centroidDispl);
      }

      ++iter;
   }
}
