#include "kmeans_base.h"

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

void kMeansBase::computeCentroids ( void ) {
   int size; MPI_Comm_size ( MPI_COMM_WORLD, &size );
   int rank; MPI_Comm_rank ( MPI_COMM_WORLD, &rank );

   // Centroids are computed in parallel
   // Each process accumulates, for each cluster, the sum of the points in that
   // cluster and their amount, over a portion of the whole dataset
   // Partial results are then gathered by process 0, who computes the average
   // and assign the result to the centroids member

   // Local sums
   std::vector<point> localSums ( k, point(n) );
   std::vector<unsigned int> localCounts ( k, 0 );

   for ( unsigned int i = rank; i < dataset.size(); i += size ) {
      localSums[dataset[i].getLabel()] += dataset[i];
      localCounts[dataset[i].getLabel()] += 1;
   }

   std::vector<unsigned int> globalCounts ( k, 0 );

   // Global counts are obtained reducing the local counts
   MPI_Reduce ( localCounts.data(), globalCounts.data(), k, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD );

   // Rank 0 : collects the partial sums from the others, then performs the average
   if ( rank == 0 ) {
      std::vector<point> globalSums = localSums;

      for ( int i = 1; i < size; ++i ) // For each other process
         for ( unsigned int kk = 0; kk < k; ++kk ) // and for each cluster
            globalSums[kk] += mpi_point_recv ( i, n );

      for ( unsigned int kk = 0; kk < k; ++kk )
         centroids[kk] = globalSums[kk] / globalCounts[kk];
   }

   // Other ranks : send their partial sums to rank 0
   else for ( const point& pt : localSums )
      mpi_point_send ( 0, pt );
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
