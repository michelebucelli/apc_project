#include "kmeans_sdg.h"

void kMeansSDG::solve ( void ) {
   int size; MPI_Comm_size ( MPI_COMM_WORLD, &size );
   int rank; MPI_Comm_rank ( MPI_COMM_WORLD, &rank );

   bool done = false;

   // Tolerance on the movement of the centroids
   // If between one iteration and the next at least one of the centroids moves
   // (in squared distance) more than this quantity, we assume there have been
   // changes, and iterate again
   real tol = 1e-12;

   std::default_random_engine eng;
   std::uniform_int_distribution<unsigned int> distro ( 0, dataset.size() );

   while ( done ) {
      auto old_centroids = centroids;

      // Only rank 0 should pick a point and recompute its label
      // Then the newly computed label shall be sent to all other ranks
      if ( rank == 0 ) {
         unsigned int idx = distro(eng);
         real minDist = dist2 ( dataset[idx], centroids[0] );
         unsigned int minDistLabel = 0;

         for ( unsigned int kk = 1; kk < k; ++kk ) {
            real d = dist2 ( dataset[idx], centroids[kk] );
            if ( d < minDist ) {
               minDist = d;
               minDistLabel = kk;
            }
         }

         dataset[idx].setLabel ( minDistLabel );

         // Now we send to other processes the new label
         for ( int i = 1; i < size; ++i ) {
            MPI_Send ( &idx, 1, MPI_UNSIGNED, i, 0, MPI_COMM_WORLD );
            MPI_Send ( &minDistLabel, 1, MPI_UNSIGNED, i, 0, MPI_COMM_WORLD );
         }
      }

      // Else, receive the information from rank 0 and update the dataset accordingly
      else {
         unsigned int idx; MPI_Recv ( &idx, 1, MPI_UNSIGNED, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE );
         unsigned int lab; MPI_Recv ( &lab, 1, MPI_UNSIGNED, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE );
         dataset[idx].setLabel ( lab );
      }

      // Finally, recompute the centroids
      computeCentroids();

      if ( rank == 0 ) {
         done = true;

         // Compute the distances between old and new centroids
         for ( unsigned int kk = 0; kk < k; ++kk ) {
            if ( dist2 ( centroids[kk], old_centroids[kk] ) > tol ) {
               done = false;
               break;
            }
         }
      }

      MPI_Bcast ( &done, 1, MPI_UNSIGNED, 0, MPI_COMM_WORLD );
   }
}
