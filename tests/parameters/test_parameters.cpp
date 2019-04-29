#include <iostream>
#include <memory>
#include <stdlib.h>
#include <gsl/gsl_statistics.h>
#include <gsl/gsl_sort.h>


#include "korali.h"

using namespace std;
using namespace Korali::Parameter;

int main(int argc, char* argv[])
{

  size_t N = 0;
  unique_ptr<Korali::Parameter::Base> p[20];

  p[N++] = make_unique<Cauchy>( 5., 2.0 );
  p[N++] = make_unique<Cauchy>( 1.0, 0.0 );
  p[N++] = make_unique<Exponential>( 3., -2. );
  p[N++] = make_unique<Exponential>( 3, 0.0 );
  p[N++] = make_unique<Gamma>( 2., 3. );
  p[N++] = make_unique<Gaussian>( 0., 0.1 );
  p[N++] = make_unique<Gaussian>( 2., 3. );
  p[N++] = make_unique<Uniform>( -2., 5. );
  p[N++] = make_unique<Laplace>( -2., 5. );

  double x = 1.;

  double den, log_den;

  // Test that log-density is correctly implemented
  printf("\n******** log-density tests ********\n\n");
  for( size_t k=0; k<N; k++){
    den = p[k]->getDensity(x);
    log_den = p[k]->getDensityLog(x) ;

    if( gsl_isinf(log_den)==-1 ) log_den = 0.;
    else log_den = gsl_sf_exp( log_den );

    p[k]->printDetails();
    printf("\n   | Density - exp(LogDensity) |  =  %le\n", fabs(den-log_den) );
  }

  // Test the statistics of the random number generators.
  printf("\n\n******** statistical tests ********\n\n");
  unique_ptr<Korali::Parameter::Base> dist;
  size_t Ns = 1e6;
  double *samples = (double*)malloc( Ns * sizeof(double) );
  size_t seed = 0xC0FFEE;

  {// 1. Gaussian
    double mean=-1., sd=2.;
    double xstat1=mean, xstat2=sd;
    double stat1, stat2;

    dist = make_unique<Gaussian>(mean,sd,seed);
    for(size_t i=0; i<Ns; i++) samples[i] = dist->getRandomNumber();
    stat1 = gsl_stats_mean(samples, 1, Ns);
    stat2 = gsl_stats_sd_m(samples, 1, Ns, stat1);

    dist->printDetails();
    printf("\n   Error1 = %e \n", fabs(xstat1-stat1));
    printf("   Error2 = %e \n",   fabs(xstat2-stat2));
  }

  {// 2. Uniform
    double a=-1., b=5.;
    double xstat1=0.5*(a+b), xstat2=(b-a)/(2.*M_SQRT3);
    double stat1, stat2;

    dist = make_unique<Uniform>(a,b,seed);
    for(size_t i=0; i<Ns; i++) samples[i] = dist->getRandomNumber();
    stat1 = gsl_stats_mean(samples, 1, Ns);
    stat2 = gsl_stats_sd_m(samples, 1, Ns, stat1);

    dist->printDetails();
    printf("\n   Error1 = %e \n", fabs(xstat1-stat1));
    printf("   Error2 = %e \n",   fabs(xstat2-stat2));
  }

  {// 3. Exponential
    double mean=3., loc=0.;
    double xstat1=loc+mean, xstat2=loc+mean;
    double stat1, stat2;

    dist = make_unique<Exponential>(mean,loc, seed);
    for(size_t i=0; i<Ns; i++) samples[i] = dist->getRandomNumber();
    stat1 = gsl_stats_mean(samples, 1, Ns);
    stat2 = gsl_stats_sd_m(samples, 1, Ns, stat1);

    dist->printDetails();
    printf("\n   Error1 = %e \n", fabs(xstat1-stat1));
    printf("   Error2 = %e \n",   fabs(xstat2-stat2));
  }

  {// 4. Gamma
    double shape=9., scale=0.5;
    double xstat1=shape*scale, xstat2=sqrt(shape)*scale;
    double stat1, stat2;

    dist = make_unique<Gamma>(shape,scale,seed);
    for(size_t i=0; i<Ns; i++) samples[i] = dist->getRandomNumber();
    stat1 = gsl_stats_mean(samples, 1, Ns);
    stat2 = gsl_stats_sd_m(samples, 1, Ns, stat1);

    dist->printDetails();
    printf("\n   Error1 = %e \n", fabs(xstat1-stat1));
    printf("   Error2 = %e \n",   fabs(xstat2-stat2));
  }

  {// 5. Cauchy
    double scale=0.5, loc=0.;
    double xstat1 = scale * tan(M_PI*(0.25-0.5)); // 25% quantile
    double xstat2 = scale * tan(M_PI*(0.75-0.5)); // 75% quantile
    double stat1, stat2;

    dist = make_unique<Cauchy>(scale,loc,seed);
    for(size_t i=0; i<Ns; i++) samples[i] = dist->getRandomNumber();
    gsl_sort( samples, 1, Ns);
    stat1 = gsl_stats_quantile_from_sorted_data(samples, 1, Ns, 0.25);
    stat2 = gsl_stats_quantile_from_sorted_data(samples, 1, Ns, 0.75);

    dist->printDetails();
    printf("\n   Error1 = %e \n", fabs(xstat1-stat1));
    printf("   Error2 = %e \n",   fabs(xstat2-stat2));
  }

  {// 6. Laplace
    double mean=-5., scale=2.;
    double xstat1=mean, xstat2=M_SQRT2*scale;
    double stat1, stat2;

    dist = make_unique<Laplace>(mean,scale,seed);
    for(size_t i=0; i<Ns; i++) samples[i] = dist->getRandomNumber();
    stat1 = gsl_stats_mean(samples, 1, Ns);
    stat2 = gsl_stats_sd_m(samples, 1, Ns, stat1);

    dist->printDetails();
    printf("\n   Error1 = %e \n", fabs(xstat1-stat1));
    printf("   Error2 = %e \n",   fabs(xstat2-stat2));
  }

 return 0;
}
