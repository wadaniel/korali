#include "gp_likelihood.hpp"

double gp_negative_loglikelihood( libgp::GaussianProcess * gp,dVector& params ){
  Eigen::VectorXd eigenParams = Eigen::VectorXd::Map(params.data(), params.size());
  gp->covf().set_loghyper(eigenParams);
  auto q=gp->covf().get_loghyper();
  return -gp->log_likelihood();
}

dVector gp_negative_dloglikelihood( libgp::GaussianProcess * gp, dVector& params ){
  Eigen::VectorXd eigenParams = Eigen::VectorXd::Map( params.data(), params.size() );
  gp->covf().set_loghyper(eigenParams);
  Eigen::VectorXd eigenGrad = -gp->log_likelihood_gradient();
  std::vector<double> grad;
  grad.resize( eigenGrad.size() );
  Eigen::VectorXd::Map( &grad[0], eigenGrad.size() ) = eigenGrad;
  return grad;
}
