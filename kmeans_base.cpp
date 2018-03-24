#include "kmeans_base.h"

real dist2 ( const point& a, const point& b ) {
   assert ( a.getN() == b.getN() );

   real sum = 0;
   for ( size_t i = 0; i < a.getN(); ++i ) {
      real x = a[i] - b[i];
      sum += x*x;
   }

   return sum;
}

std::istream& operator>> ( std::istream &in, kMeansBase &km ) {
   unsigned int i = 0;
   real tmp = 0;
   in >> km.n;
   point p ( km.n );

   while ( in >> tmp ) {
      p[i] = tmp;
      if ( i == km.n - 1 ) {
         km.dataset.push_back(p);
         i = 0;
      }
      else i++;
   }

   return in;
}
