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

void mpi_point_reduce ( point * pt ) {
   MPI_Allreduce ( MPI_IN_PLACE, pt->data(), pt->getN(), MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD );
}

void kMeansBase::setK ( unsigned int kk ) {
   k = kk;
   centroids = std::vector<point> ( kk, point(n) );
   counts = std::vector<int> ( kk, 0 );
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

   for ( unsigned int i = rank; i < dataset.size(); i += size ) {
      unsigned int l = dataset[i].getLabel();
      localSums[l] += dataset[i];
   }

   for ( unsigned int kk = 0; kk < k; ++kk ) {
      mpi_point_reduce ( &localSums[kk] );
      centroids[kk] = localSums[kk] / counts[kk];
   }

   // We use as centroid the point in the dataset which is nearest to the average
   // (aka "medoid")

   for ( unsigned int kk = rank; kk < k; kk += size ) {
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

   // Processes communicate their computed centroids
   for ( int kk = 0; kk < int(k); ++kk ) {
      if ( kk % size == rank ) {
         for ( int j = 0; j < size; ++j )
            if ( j != rank ) mpi_point_send ( j, centroids[kk] );
      }

      else {
         centroids[kk] = mpi_point_recv ( kk % size, n );
      }
   }
}

void kMeansBase::randomize ( void ) {
   int size; MPI_Comm_size ( MPI_COMM_WORLD, &size );
   int rank; MPI_Comm_rank ( MPI_COMM_WORLD, &rank );

   if ( rank == 0 ) {
      std::default_random_engine eng;
      std::uniform_int_distribution<unsigned int> dist ( 0, k - 1 );

      counts = std::vector<int>(k,0);

      for ( auto & pt : dataset ) {
         unsigned int lab = dist(eng);
         pt.setLabel ( lab );
         counts[lab]++;

         for ( int i = 1; i < size; ++i )
            MPI_Send ( &lab, 1, MPI_UNSIGNED, i, 0, MPI_COMM_WORLD );
      }
   }

   else for ( auto & pt : dataset ) {
      int lab = 0;
      MPI_Recv ( &lab, 1, MPI_UNSIGNED, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE );
      pt.setLabel ( lab );
   }

   MPI_Bcast ( counts.data(), k, MPI_INT, 0, MPI_COMM_WORLD );
}

void kMeansBase::getTrueLabels ( std::istream& in, int offset ) {
   for ( auto & pt : dataset ) {
      int tmp; in >> tmp;
      pt.setTrueLabel ( tmp + offset );
   }
}

real kMeansBase::purity ( void ) const {
   int size; MPI_Comm_size ( MPI_COMM_WORLD, &size );
   int rank; MPI_Comm_rank ( MPI_COMM_WORLD, &rank );

   // True labels of the clusters
   std::vector<int> trueLabels ( k, -1 );

   // Counts of the labels assigned to each cluster
   // On the rows ( i.e. counts[i] ) we have the vector of the amounts of points
   // for each label assigned to the cluster i ( that is : counts[i][j] is the
   // number of points with true label j assigned to cluster i)
   std::vector<std::vector<int>> counts ( k, std::vector<int>(k,0) );

   // Iterate through the whole dataset and compute the counts
   for ( unsigned int i = rank; i < dataset.size(); i += size )
      counts[dataset[i].getLabel()][dataset[i].getTrueLabel()] += 1;

   // Communicate the results and compute the true labels
   for ( unsigned int kk = 0; kk < k; ++kk ) {
      MPI_Allreduce ( MPI_IN_PLACE, counts[kk].data(), k, MPI_INT, MPI_SUM, MPI_COMM_WORLD );

      int maxIdx = 0, maxCount = counts[kk][0];
      for ( unsigned int j = 1; j < k; ++j ) {
         if ( counts[kk][j] > maxCount ) {
            maxCount = counts[kk][j];
            maxIdx = j;
         }
      }

      trueLabels[kk] = maxIdx;
   }

   // Compute purity
   real result = 0;

   for ( unsigned int i = rank; i < dataset.size(); i += size )
      if ( dataset[i].getTrueLabel() == trueLabels[dataset[i].getLabel()] ) result += 1;

   MPI_Allreduce ( MPI_IN_PLACE, &result, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD );

   return result / dataset.size();
   return 0;
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
