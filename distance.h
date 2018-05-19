#ifndef _DISTANCE_H
#define _DISTANCE_H

#include "point.h"
#include <cmath>

// P-distance class
template < int p >
class dist_p {
   public:
   double dist ( const point & a, const point & b ) {
      assert ( a.getN() == b.getN() );

      double sum = 0; double x = 0; double xp = 0;
      for ( unsigned int i = 0; i < a.getN(); ++i ) {
         x = abs(a[i] - b[i]);

         xp = x;
         for ( int pp = 1; pp < p; ++pp ) xp *= x;

         sum += xp;
      }

      return sum;
   }
};

// Relevant aliases
using dist_manhattan = dist_p<1>;
using dist_euclidean = dist_p<2>;

// Minkowski distance class
template < int p >
class dist_minkowski {
   public:
   double dist ( const point & a, const point & b ) {
      assert ( a.getN() == b.getN() );

      double sum = 0; double x = 0;
      for ( unsigned int i = 0; i < a.getN(); ++i ) {
         x = abs(a[i] - b[i]);
         sum += pow(x, 1.0/p);
      }

      return pow(sum, p);
   }
};

#endif
