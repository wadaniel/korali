#include "gaussian.h"

#include <gsl/gsl_matrix.h>
#include <gsl/gsl_randist.h>
#include <gsl/gsl_linalg.h>

gsl_vector_view mean_view;
gsl_matrix *L;

void gaussian(std::vector<double>& pars, std::vector<double>& result)
{
 size_t nPars = pars.size();
 double work[nPars];
 gsl_vector_view work_view = gsl_vector_view_array(work, nPars);
 gsl_vector_view vals_view = gsl_vector_view_array(pars.data(), nPars);
 double res = 0.0;
 gsl_ran_multivariate_gaussian_log_pdf(&vals_view.vector, &mean_view.vector, L, &res, &work_view.vector);
 result.push_back(res);
}

void gaussian_init(size_t nPars)
{
 double* mean = (double*) calloc (nPars, sizeof(double));
 double* sigma = (double*) calloc (nPars*nPars, sizeof(double));

 for (size_t i = 0; i < nPars; i++) mean[i] = -3.0;
 for (size_t i = 0; i < nPars; i++) sigma[i*nPars+i] = 5.0;

 mean_view  = gsl_vector_view_array(mean, nPars);
 gsl_matrix_view sigma_view  = gsl_matrix_view_array(sigma, nPars,nPars);
 L = gsl_matrix_alloc(nPars,nPars);
 gsl_matrix_memcpy(L, &sigma_view.matrix);
 gsl_linalg_cholesky_decomp(L);
}
