#include "kmeans_base.h"

std::ostream& operator<< ( std::ostream& out, const point &pt ) {
   out << pt.getLabel() << " ";

   unsigned int i = 0;
   for ( ; i < pt.getN() - 1; ++i )
      out << pt[i] << " ";
   out << pt[i];

   return out;
}

real dist2 ( const point& a, const point& b ) {
   assert ( a.getN() == b.getN() );

   real sum = 0;
   for ( unsigned int i = 0; i < a.getN(); ++i ) {
      real x = a[i] - b[i];
      sum += x*x;
   }

   return sum;
}

point operator+ ( const point& a, const point& b ) {
   assert ( a.getN() == b.getN() );

   point result ( a.getN() );
   for ( unsigned int i = 0; i < a.getN(); ++i )
      result[i] = a[i] + b[i];

   return result;
}

point operator- ( const point& a, const point& b ) {
   assert ( a.getN() == b.getN() );

   point result ( a.getN() );
   for ( unsigned int i = 0; i < a.getN(); ++i )
      result[i] = a[i] - b[i];

   return result;
}

point& operator+= ( point &a, const point &b ) {
   assert ( a.getN() == b.getN() );
   a = a + b;
   return a;
}

point operator/ ( const point &a, real t ) {
   point result ( a );
   for ( unsigned int i = 0; i < a.getN(); ++i )
      result[i] /= t;
   return result;
}

void mpi_point_send ( unsigned int dest, const point & pt ) {
   MPI_Send ( pt.data(), pt.getN(), MPI_DOUBLE, dest, 0, MPI_COMM_WORLD );
}

point mpi_point_recv ( unsigned int src, unsigned int n ) {
   std::vector<real> coords ( n );
   MPI_Recv ( coords.data(), n, MPI_DOUBLE, src, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE );
   return point ( n, coords );
}

void kMeansBase::setK ( unsigned int kk ) {
   k = kk;
   centroids = std::vector<point> ( kk, point(n) );
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

std::ostream& operator<< ( std::ostream &out, const kMeansBase &km ) {
   out << "dim = " << km.n << "\n" << "clusters = " << km.k << "\n";

   for ( const auto &i : km.dataset )
      out << i << "\n";

   return out;
}
