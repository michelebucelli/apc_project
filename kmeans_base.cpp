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

   real sum = 0; real x = 0;
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

   // Computes local sums, then reduces

   centroids = std::vector<point> ( k, point(n) );

   for ( unsigned int i = rank; i < dataset.size(); i += size ) {
      unsigned int l = dataset[i].getLabel();
      for ( unsigned int nn = 0; nn < n; ++nn )
         centroids[l][nn] += dataset[i][nn] / counts[l];
   }

   for ( unsigned int kk = 0; kk < k; ++kk )
      mpi_point_reduce ( &centroids[kk] );
}

void kMeansBase::randomize ( void ) {
   std::default_random_engine eng(1);
   std::uniform_int_distribution<unsigned int> dist ( 0, k - 1 );

   counts = std::vector<int>(k,0);

   for ( auto & pt : dataset ) {
      unsigned int lab = dist(eng);
      pt.setLabel ( lab );
      counts[lab]++;
   }
}

void kMeansBase::readTrueLabels ( std::istream& in, int offset ) {
   for ( auto & pt : dataset ) {
      int tmp; in >> tmp;
      pt.setTrueLabel ( tmp + offset );
   }
}

void kMeansBase::setTrueLabels ( const std::vector<int>& in, int offset ) {
   for ( unsigned int i = 0; i < in.size(); ++i )
      dataset[i].setTrueLabel ( in[i] + offset );
}

real kMeansBase::purity ( void ) const {
   // True labels of the clusters
   std::vector<int> trueLabels ( k, -1 );

   // Counts of the labels assigned to each cluster
   // On the rows ( i.e. counts[i] ) we have the vector of the amounts of points
   // for each label assigned to the cluster i ( that is : counts[i][j] is the
   // number of points with true label j assigned to cluster i)
   std::vector<std::vector<int>> counts ( k, std::vector<int>(k,0) );

   // Iterate through the whole dataset and compute the counts
   for ( unsigned int i = 0; i < dataset.size(); ++i )
      counts[dataset[i].getLabel()][dataset[i].getTrueLabel()] += 1;

   // Compute the true labels
   for ( unsigned int kk = 0; kk < k; ++kk ) {
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

   for ( unsigned int i = 0; i < dataset.size(); ++i )
      if ( dataset[i].getTrueLabel() == trueLabels[dataset[i].getLabel()] ) result += 1;

   return result / dataset.size();
}

std::istream& operator>> ( std::istream &in, kMeansDataset &km ) {
   unsigned int i = 0;
   unsigned int n = 0;
   real tmp = 0;

   in >> n;
   point p ( n );

   while ( in >> tmp ) {
      p[i] = tmp;
      if ( i == n - 1 ) {
         km.push_back(p);
         i = 0;
      }
      else i++;
   }

   return in;
}

std::istream& operator>> ( std::istream &in, std::vector<int> & out ) {
   int tmp;
   while ( in >> tmp ) out.push_back(tmp);
   return in;
}

std::ostream& operator<< ( std::ostream &out, const kMeansBase &km ) {
   out << "dim = " << km.n << ";\n" << "clusters = " << km.k << ";\n";
   out << "dataset = [ ";

   unsigned int i = 0;
   for ( ; i < km.size()-1; ++i )
      out << km.dataset[i] << ";\n";

   out << km.dataset[i] << "];";

   return out;
}
