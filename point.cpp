#include "point.h"

std::ostream& operator<< ( std::ostream& out, const point &pt ) {
   out << pt.getLabel() << " ";

   unsigned int i = 0;
   for ( ; i < pt.getN() - 1; ++i )
      out << pt[i] << " ";
   out << pt[i];

   return out;
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

void mpi_point_allreduce ( point * pt ) {
   MPI_Allreduce ( MPI_IN_PLACE, pt->data(), pt->getN(), MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD );
}

void mpi_point_send ( unsigned int dest, const point & pt ) {
   int label = pt.getLabel();
   MPI_Send ( &label, 1, MPI_INT, dest, 0, MPI_COMM_WORLD );
   MPI_Send ( pt.data(), pt.getN(), MPI_DOUBLE, dest, 0, MPI_COMM_WORLD );
}

point mpi_point_recv ( unsigned int src, unsigned int n ) {
   point result ( n );
   int label = -1;
   MPI_Recv ( &label, 1, MPI_INT, src, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE );
   MPI_Recv ( result.data(), n, MPI_DOUBLE, src, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE );
   result.setLabel ( label );
   return result;
}
