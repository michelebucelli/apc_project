#ifndef _TIMER_H
#define _TIMER_H

#include <chrono>
#include <vector>
#include <string>
#include <sstream>
#include <iomanip>

class timer {
public:
   using clock = std::chrono::high_resolution_clock;
   using timePoint = std::chrono::time_point<clock>;

private:
   timePoint startTime;
   timePoint stopTime;

public:
   timer () : startTime ( clock::now() ), stopTime ( clock::now() ) { };
   void start ( void ) { startTime = stopTime = clock::now(); };
   void stop ( void ) { stopTime = clock::now(); };

   double getTime ( void ) const {
      auto tspan = std::chrono::duration_cast<std::chrono::nanoseconds> ( stopTime - startTime );
      return tspan.count() / 1000000.0;
   }
};

class multiTimer {
public:
   using clock = std::chrono::high_resolution_clock;
   using timePoint = std::chrono::time_point<clock>;

private:
   std::vector<timePoint> startTimes;
   std::vector<timePoint> stopTimes;
   std::vector<double> cumulates;

public:
   multiTimer ( int n ) : startTimes ( n, clock::now() ), stopTimes ( n, clock::now() ), cumulates(n,0) { };
   void start ( int n ) { startTimes[n] = stopTimes[n] = clock::now(); }
   void stop ( int n ) { stopTimes[n] = clock::now(); cumulates[n] += getTime(n); }

   double getTime ( int n ) const {
      auto tspan = std::chrono::duration_cast<std::chrono::nanoseconds> ( stopTimes[n] - startTimes[n] );
      return tspan.count() / 1000000.0;
   }

   std::string cumulatesToString ( void ) const {
      std::stringstream st;
      for ( auto i : cumulates )
         st << std::setw(10) << i << "  ";
      return st.str();
   }

   double getCumulateTime ( int n ) const { return cumulates[n]; }
};

#endif
