#include <array>
#include <random>

#include <gsl/gsl_vector.h>
#include <gsl/gsl_multimin.h>

#include "korali.h"

bool verbose = false;

std::default_random_engine generator;
std::exponential_distribution<double> exponential(2.0);  //mean 0.5
std::lognormal_distribution<double> lognorm(0.0,0.2);    //mean 0.2
std::uniform_real_distribution<double> uniform(0.0,1.0);

void createSamples(double * out, int N);
int test_minimizer(const double * fj, int N, double rho, double covtol, double *xmin, double *fmin, int minimizer);

int main(int argc, char* argv[]) {

  std::array<int,9> ns = {1000, 2000, 4000, 8000, 16000, 32000, 64000, 128000, 256000};
  std::array<double,6> covs = {0.05, 0.1, 0.2, 0.4, 0.8, 1.0}; 
  std::array<double,9> rhos = {0.0, 0.001, 0.01, 0.1, 0.2, 0.4, 0.6, 0.8, 0.9};

  double xmin, fmin;

  int count1 = 0;
  int count2 = 0;
  int count3 = 0;

  double err1 = 0;
  double err2 = 0;
  double err3 = 0;

  int failed1 = 0;
  int failed2 = 0;
  int failed3 = 0;

  int run = 0;
  for(auto n : ns) {
  double *fj = new double[n];
  
  for(auto c : covs) {
  for(auto r : rhos) {

  run++;
 
  printf("\nRun N:%d, Rho %f, Cov, %f\n", n, r, c);
  printf("Finished: %.2f%%\n", 100.0*run/(ns.size()*covs.size()*rhos.size()));

  createSamples(fj, n);
  
  int it1 = test_minimizer(fj, n, r, c, &xmin, &fmin, 1);  
  count1 += it1;
  err1   += fmin;
  if(xmin<r) failed1++;

  if (verbose) printf("it1: %d, xmin: %f, fmin: %f\n", it1, xmin, fmin);

  int it2 = test_minimizer(fj, n, r, c, &xmin, &fmin, 2);  
  count2 += it2;
  err2   += fmin;
  if(xmin<r) failed2++;

  if (verbose) printf("it2: %d, xmin: %f, fmin: %f\n", it2, xmin, fmin);

   int it3 = test_minimizer(fj, n, r, c, &xmin, &fmin, 3);  
  count3 += it3;
  err3   += fmin;
  if(xmin<r) failed3++;

  if (verbose) printf("it3: %d, xmin: %f, fmin: %f\n", it3, xmin, fmin);

  }}
  
  delete [] fj;

  }

  printf("gsl_multimin_fminimizer_nmsimplex: total it1: %d, failed: %d, err: %e\n", count1, failed1, err1);
  printf("gsl_multimin_fminimizer_nmsimplex2: total it2: %d, failed: %d, err: %e\n", count2, failed2, err2);
  printf("gsl_multimin_fminimizer_nmsimplex2rand: atotal it3: %d, failed: %d, err: %e\n", count3, failed3, err3);
}


void createSamples(double *out, int N)
{
    int i = 0;
    for( ; i < N-3; i+=3)
    {
        out[i]   = exponential(generator);
        out[i+1] = lognorm(generator);
        out[i+2] = uniform(generator);
    }
    for( ; i < N; ++i) out[i] = exponential(generator);
}


int test_minimizer( const double *fj, int N, double rho, double covtol, double *xmin, double *fmin, int minimizer)
{

 int fn           = N;
 double pj        = rho;
 double objTol    = covtol;

 size_t MaxIter     = 100;    /* Max number of search iterations */
 double Tol         = 1e-12;  /* Tolerance for root finding */
 double Step        = 1e-8;   /* Search stepsize */

 const gsl_multimin_fminimizer_type *T;
 gsl_multimin_fminimizer *s = NULL;
 gsl_vector *ss, *x;
 gsl_multimin_function minex_func;

 size_t iter     = 0;
 int status;
 double size;

 Korali::Solver::fparam_t fp;
 fp.fj = fj;
 fp.fn = fn;
 fp.pj = pj;
 fp.cov = objTol;

 x = gsl_vector_alloc (1);
 gsl_vector_set (x, 0, pj);

 ss = gsl_vector_alloc (1);
 gsl_vector_set_all (ss, Step);

 minex_func.n      = 1;
 minex_func.f      = Korali::Solver::TMCMC::objLog;
 minex_func.params = &fp;

 // SELECT ONE MINIMIZER STRATEGY
 switch(minimizer)
 {
   case 1: T = gsl_multimin_fminimizer_nmsimplex; break;
   case 2: T = gsl_multimin_fminimizer_nmsimplex2; break;
   case 3: T = gsl_multimin_fminimizer_nmsimplex2rand; break;
   default: printf("select minimizer 1,2 or 3!!!"); abort();
 }
 
 s = gsl_multimin_fminimizer_alloc (T, 1);
 gsl_multimin_fminimizer_set (s, &minex_func, x, ss);

 *fmin = 0;
 *xmin = 0.0;

 do {
   iter++;
   status = gsl_multimin_fminimizer_iterate(s);
   size   = gsl_multimin_fminimizer_size (s);
   status = gsl_multimin_test_size (size, Tol);

   if (status == GSL_SUCCESS) {
     if(s->fval <= Tol) {
       if(verbose) printf ("fminsearch: Converged to minimum at\n");
     } else {
       if (verbose) printf ("fminsearch: Converged but did not find minimum.\n");
     }
    } else if (s->fval <= Tol) {
     status = GSL_SUCCESS;
     if (verbose) printf ("fminsearch: NOT converged but found minimum at\n");
   } else 
    if (verbose)  printf("fminsearch: NOT converged and did not find minimum.\n");



 } while (status == GSL_CONTINUE && iter < MaxIter);

 if (s->fval <= Tol) {
   *fmin = s->fval;
   *xmin = gsl_vector_get(s->x, 0);
 }

 gsl_vector_free(x);
 gsl_vector_free(ss);
 
 return iter;

}
