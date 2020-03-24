/*
 * Test whether the distribution parameterized by S, zeta and phi in model.cpp is correct or not
*/
#include "model.hpp"
#include "korali.hpp"
#include "load_data.hpp"

#include "test_model.hpp"

#include <math.h>




int main(int argc, char* argv[])
{
	// Initialize the distribution

	ExampleDistribution2  distrib2 = ExampleDistribution2(); // @suppress("Function cannot be resolved") // @suppress("Type cannot be resolved")

    std::function<void(korali::Sample& s)> distrib2_S = [&distrib2](korali::Sample& s) -> void { // @suppress("Symbol is not resolved") // @suppress("Type cannot be resolved")
    	distrib2.S(s); } ; // @suppress("Method cannot be resolved")
    std::function<void(korali::Sample& s)> distrib2_zeta = [&distrib2](korali::Sample& s) -> void { // @suppress("Symbol is not resolved") // @suppress("Type cannot be resolved")
    	distrib2.zeta(s); } ; // @suppress("Method cannot be resolved")
    std::function<void(korali::Sample& s)> distrib2_phi = [&distrib2](korali::Sample& s) -> void { // @suppress("Type cannot be resolved") // @suppress("Symbol is not resolved")
    	distrib2.phi(s); } ; // @suppress("Method cannot be resolved")


   // int number_tests = 13;
    // int number_points = 10;
    int nClusters = distrib2._p.nClusters; // 2
    int nDimensions = distrib2._p.nDimensions; // 2
    int d2_numberLatentVars = distrib2._p.nPoints; // one for each datapoint
    int d2_numberHyperparams = distrib2._p.nDimensions * distrib2._p.nClusters + 1;

     // Some value pairs:
    std::vector<double> mu0 = {0, 0};
    std::vector<double> mu1 = {2, 2};
    std::vector<double> mu2 = {-0.5, 10};
    double sigma = 1.0;
    std::vector<std::vector<int> > assignments;
    std::vector<std::vector<double> > hyperparams;
    std::vector<std::vector<std::vector<double> > > points;
    assignments.push_back({0});
    hyperparams.push_back({0, 0, 2, 2, sigma}); // (mu1, mu2, sigma)
    points.push_back( {
        {0, 0}
        });



    double p;
    double p_from_model_direct;
    double p_from_model_via_korali;
    double _zetaValue;
    std::vector<double> _sValues;
    std::vector<double> _phiValues;
    //korali::problem::bayesian::Latent problem;
    //korali::Experiment *e;
    auto e_ptr = new korali::Experiment();
    auto e = *e_ptr;
    delete e_ptr;
    auto eng = korali::Engine();
    korali::Sample* k_ptr;
    korali::Sample k;
    std::vector<std::vector<double> > current_points;

    for (int i = 0; i < assignments.size(); i++){
        /* reference probability */
        current_points = points[i];
        p = multivariate_gaussian_probability({mu0, mu1}, nDimensions, assignments[i], nClusters, sigma, current_points);

        reset_points(distrib2._p, current_points, assignments[i], nClusters );
        //distrib2._p.points =current_points;
        /* Update S, zeta, phi to the new set of points (needed?)*/
        distrib2_S = [&distrib2](korali::Sample& s) -> void {
    	    distrib2.S(s); } ;
        distrib2_zeta = [&distrib2](korali::Sample& s) -> void {
    	    distrib2.zeta(s); } ;
        distrib2_phi = [&distrib2](korali::Sample& s) -> void {
    	    distrib2.phi(s); } ;

        /* Use the distributions S, zeta and phi functions to calculate the probability */
        k_ptr = new korali::Sample();
        k = *k_ptr;
        k["Latent Variables"] = assignments[i];
        k["Hyperparameters"] = hyperparams[i];

        //std::function<void(korali::Sample&)> *func_ptr_S = &distrib2_S;

        distrib2_S(k);
        distrib2_zeta(k);
        distrib2_phi(k);

        double _zetaValue = k["zeta"].get<double>();
        _sValues = k["S"].get<std::vector<double> >();
        _phiValues = k["phi"].get<std::vector<double> >();

        p_from_model_direct =  std::exp( - _zetaValue
                                         + std::inner_product(std::begin(_sValues), std::end(_sValues),
                                                                                        std::begin(_phiValues), 0.0) );

        assert (std::abs(p - p_from_model_direct) < 0.001);
        delete k_ptr;



    /*
     *  Create a "Latent" korali problem and use it to calculate the loglikelihood
     */

//         e_ptr  = new korali::Experiment();
//         e = *e_ptr;
//
//
////        problem._type = "Bayesian/Latent";
////        problem._sOfLikelihoodModel = distrib2_S;
////        problem._zetaOfLikelihoodModel = distrib2_zeta;
////        problem._phiOfLikelihoodModel = distrib2_phi;
////        problem._sDimension = distrib2.sufficientStatisticsDimension;
//
//        e["Problem"]["Type"] = "Bayesian/Latent";
//        e["Problem"]["S Of Likelihood Model"] = distrib2_S;
//        e["Problem"]["Zeta Of Likelihood Model"] = distrib2_zeta;
//        e["Problem"]["Phi Of Likelihood Model"] = distrib2_phi;
//        e["Problem"]["S Dimension"] = distrib2.sufficientStatisticsDimension;
//
//        eng.run(e);
//
//        k_ptr = new korali::Sample();
//        k = *k_ptr;
//        k["Latent Variables"] = assignments[i];
//        k["Hyperparameters"] = hyperparams[i];
//
//
//        e._problem->runEvaluation("Evaluate logLikelihood", k);
//       // problem.runEvaluation("Evaluate logLikelihood", k);
//
//        p_from_model_via_korali = std::exp(k["logLikelihood"].get<double>());
//
//        assert (std::abs(p - p_from_model_via_korali) < 0.001);
//        delete &e;
//        delete k_ptr;
    }


    return 0;

}

double multivariate_gaussian_probability(std::vector<std::vector<double> > mus, int nDimensions, std::vector<int> assignments,
                                         int nClusters, double sigma, std::vector<std::vector<double> > points) {
    /* Normalized, no log, clean pretty probability for N points. */
    assert (sigma > 0.0);
    assert (nDimensions > 0);
    for (int a : assignments)
        assert ((0 <= a) && ( a < nClusters));
    for (std::vector<double> v : points)
        assert (v.size() == nDimensions);

    int N = points.size();

    double p = 1.0;
    for (int i = 0; i < N; i++){
        // get the cluster mean
        int c = assignments[i];
        std::vector<double> mu = mus[c];
        // update p
        p = p * univariate_gaussian_probability(mu, sigma, points[i]);
    }
    return p;
}

double univariate_gaussian_probability(std::vector<double> mu, double sigma, std::vector<double> point){
    assert (mu.size() == point.size());
    std::vector<double> distance(mu.size());
    std::transform(mu.begin(), mu.end(), point.begin(),
               distance.begin(), std::minus<double>());
    double squared_distance = std::inner_product(std::begin(distance), std::end(distance) , std::begin(distance), 0.0);
    return 1. / std::sqrt(2.0 * M_PI) * std::exp(- squared_distance / (2 * std::pow(sigma, 2)));
}
