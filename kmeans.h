#ifndef _KMEANS_H
#define _KMEANS_H

#include "kmeans_base.h"

class kMeans : public kMeansBase {
public:
   kMeans ( unsigned int nn, const std::vector<point> & pts ) : kMeansBase ( nn, pts ) { }
   kMeans ( std::istream& in ) : kMeansBase(in) { };

   // Solve method
   void solve ( void ) override;
};

#endif
