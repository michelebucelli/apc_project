#ifndef _KMEANS_SDG_H
#define _KMEANS_SDG_H

#include "kmeans_base.h"

// K-Means algorithm using stochastic gradient descent

// At each iteration, we pick a random point from the dataset, we compute its
// distance from all the centroids and we assign to it the label of the closest
// centroid. After doing that, we recompute the centroids

// Iterations stop when there are no more changes in the centroids

class kMeansSDG : public kMeansBase {
public:
   kMeansSDG ( unsigned int nn, const std::vector<point> & pts ) : kMeansBase ( nn, pts ) { }
   kMeansSDG ( std::istream& in ) : kMeansBase(in) { };

   void solve ( void ) override;
};

#endif
