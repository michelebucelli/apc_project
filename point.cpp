#include "point.h"

std::ostream& operator<< ( std::ostream& out, const point &pt ) {
   out << pt.getLabel() << " ";

   unsigned int i = 0;
   for ( ; i < pt.getN() - 1; ++i )
      out << pt[i] << " ";
   out << pt[i];

   return out;
}

double dist2 ( const point& a, const point& b ) {
   assert ( a.getN() == b.getN() );

   double sum = 0; double x = 0;
   for ( unsigned int i = 0; i < a.getN(); ++i ) {
      x = a[i] - b[i];
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

point operator/ ( const point &a, double t ) {
   point result ( a );
   for ( unsigned int i = 0; i < a.getN(); ++i )
      result[i] /= t;
   return result;
}

void mpi_point_reduce ( point * pt ) {
   MPI_Allreduce ( MPI_IN_PLACE, pt->data(), pt->getN(), MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD );
}
