#ifndef _KMEANS_SGD_H
#define _KMEANS_SGD_H

#include "kmeans_base.h"

// K-Means algorithm using stochastic gradient descent

// At each iteration, we pick a random point from the dataset, we compute its
// distance from all the centroids and we assign to it the label of the closest
// centroid. After doing that, we recompute the centroids

// Iterations stop when there are no more changes in the centroids

class kMeansSGD : public kMeansBase {
private:
   int batchSize = 20;
public:
   kMeansSGD ( unsigned int nn, const std::vector<point> & pts ) : kMeansBase ( nn, pts ) { }
   kMeansSGD ( std::istream& in ) : kMeansBase(in) { };
   kMeansSGD ( std::istream& datasetIn, std::istream& trueLabelsIn ) : kMeansBase ( datasetIn, trueLabelsIn ) { };

   void solve ( void ) override;

   // Batch size get-set
   int getBatchSize ( void ) const { return batchSize; }
   void setBatchSize ( int bs ) { batchSize = bs; }
};

#endif
