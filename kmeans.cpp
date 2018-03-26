#include "kmeans.h"
#include <iostream>
#include <ctime>
using std::clog; using std::endl;

void kMeans::computeCentroids ( void ) {
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
      unsigned int l = dataset[i].getLabel();
      localSums[l] += dataset[i];
      localCounts[l]++;
   }

   std::vector<unsigned int> globalCounts ( k, 0 );

   // Global counts are obtained reducing the local counts
   MPI_Reduce ( localCounts.data(), globalCounts.data(), k, MPI_UNSIGNED, MPI_SUM, 0, MPI_COMM_WORLD );

   // Rank 0 : collects the partial sums from the others, then performs the average
   // and send the result to all other processes
   if ( rank == 0 ) {
      std::vector<point> globalSums = localSums;

      for ( int i = 1; i < size; ++i ) // For each other process
         for ( unsigned int kk = 0; kk < k; ++kk ) // and for each cluster
            globalSums[kk] += mpi_point_recv ( i, n ); // Receive the local sum

      // Computes the centroids
      for ( unsigned int kk = 0; kk < k; ++kk )
         centroids[kk] = globalSums[kk] / globalCounts[kk];

      // Sends the centroids to all other processes
      for ( int i = 1; i < size; ++i )
         for ( unsigned int kk = 0; kk < k; ++kk )
            mpi_point_send ( i, centroids[kk] );
   }

   // Other ranks : send their partial sums to rank 0, then receive the centroids
   else {
      for ( const point& pt : localSums )
         mpi_point_send ( 0, pt );

      for ( unsigned int kk = 0; kk < k; ++kk )
         centroids[kk] = mpi_point_recv ( 0, n );
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

void kMeans::randomize ( void ) {
   int size; MPI_Comm_size ( MPI_COMM_WORLD, &size );
   int rank; MPI_Comm_rank ( MPI_COMM_WORLD, &rank );

   if ( rank == 0 ) {
      std::default_random_engine eng(std::time(NULL));
      std::uniform_int_distribution<unsigned int> dist ( 0, k - 1 );

      for ( auto & pt : dataset ) {
         unsigned int lab = dist(eng);
         pt.setLabel ( lab );

         for ( int i = 1; i < size; ++i )
            MPI_Send ( &lab, 1, MPI_UNSIGNED, i, 0, MPI_COMM_WORLD );
      }
   }

   else for ( auto & pt : dataset ) {
      int lab = 0;
      MPI_Recv ( &lab, 1, MPI_UNSIGNED, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE );
      pt.setLabel ( lab );
   }
}

void kMeans::solve ( void ) {
   int size; MPI_Comm_size ( MPI_COMM_WORLD, &size );
   int rank; MPI_Comm_rank ( MPI_COMM_WORLD, &rank );

   randomize();

   int iter = 0;

   while ( iter < 100 ) {
      computeCentroids();

      for ( unsigned int i = rank; i < dataset.size(); i += size ) {
         real nearestDist = dist2 ( dataset[i], centroids[0] );
         int nearestLabel = 0;

         for ( unsigned int kk = 1; kk < k; ++kk ) {
            real d = dist2 ( dataset[i], centroids[kk] );

            if ( d < nearestDist ) {
               nearestDist = d;
               nearestLabel = kk;
            }
         }

         dataset[i].setLabel(nearestLabel);
      }

      for ( int i = 0; i < int(dataset.size()); ++i ) {
         if ( i % size == rank ) {
            int lab = dataset[i].getLabel();
            for ( int j = 0; j < size; ++j )
               if ( j != rank ) MPI_Send ( &lab, 1, MPI_INT, j, 0, MPI_COMM_WORLD );
         }

         else {
            int lab = 0;
            MPI_Recv ( &lab, 1, MPI_INT, i % size, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE );
            dataset[i].setLabel ( lab );
         }
      }

      ++iter;
   }
}
