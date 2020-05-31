#ifndef __GP_LIKELIHOOD_H__
#define __GP_LIKELIHOOD_H__

#include "gp.hpp"
#include <vector>

typedef std::vector<double> dVector;

double gp_negative_loglikelihood(libgp::GaussianProcess *gp, dVector &params);
dVector gp_negative_dloglikelihood(libgp::GaussianProcess *gp, dVector &params);

#endif
