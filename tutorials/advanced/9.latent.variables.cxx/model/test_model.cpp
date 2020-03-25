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
    int nClusters = distrib2._p.nClusters; // 2 // @suppress("Field cannot be resolved")
    int nDimensions = distrib2._p.nDimensions; // 2 // @suppress("Field cannot be resolved")
    int d2_numberLatentVars = distrib2._p.nPoints; // one for each datapoint // @suppress("Field cannot be resolved")
    int d2_numberHyperparams = distrib2._p.nDimensions * distrib2._p.nClusters + 1; // @suppress("Field cannot be resolved")

     // Some value pairs:
    std::vector<double> mu0 = {0, 0};				// @suppress("Type cannot be resolved") // @suppress("Method cannot be resolved") // @suppress("Symbol is not resolved") // @suppress("Function cannot be resolved")
    std::vector<double> mu1 = {2, 2};				// @suppress("Type cannot be resolved") // @suppress("Method cannot be resolved") // @suppress("Symbol is not resolved") // @suppress("Function cannot be resolved")
    std::vector<double> mu2 = {-0.5, 10};			// @suppress("Type cannot be resolved") // @suppress("Method cannot be resolved") // @suppress("Symbol is not resolved") // @suppress("Function cannot be resolved")
    double sigma = 1.0;
    std::vector<std::vector<int> > assignments;
    std::vector<std::vector<double> > hyperparams;
    std::vector<std::vector<std::vector<double> > > points;
    /* The test inputs*/
    assignments.push_back({0});
    hyperparams.push_back({0, 0, 2, 2, sigma}); // (mu1, mu2, sigma)
    points.push_back( {
        {0, 0}
        });
    assignments.push_back({0, 1});
    hyperparams.push_back({0, 0, 2, 2, sigma}); // (mu1, mu2, sigma)
    points.push_back( {
        {2,2}, {0,0}
        });
    assignments.push_back({0});
    hyperparams.push_back({0, 0, 2, 2, 0.5}); // (mu1, mu2, sigma)
    points.push_back( {
        {0,0}
        });
    assignments.push_back({1,1});
    hyperparams.push_back({0, 0, 2, 2, 0.5}); // (mu1, mu2, sigma)
    points.push_back( {
        {1.5, 2.25}, {-0.5, 2}
        });
    assignments.push_back({0,0,0, 1,1,2});
    hyperparams.push_back({-1.5, 0, 20., 5., 3.0, 3.5, 7.5}); // (mu1, mu2, sigma)
    points.push_back( {
        {1.5, -1.5}, {-0.5, 2}, {-5., 7.,},
		{22.0, 10.3}, {17.5, 30.},     {0.0, 7.2}
        });



    double p;
    double p_from_model_direct;
    double p_from_model_via_korali;
    double _zetaValue;
    std::vector<double> _sValues;		// @suppress("Type cannot be resolved") // @suppress("Method cannot be resolved") // @suppress("Symbol is not resolved") // @suppress("Function cannot be resolved")
    std::vector<double> _phiValues;		// @suppress("Type cannot be resolved") // @suppress("Method cannot be resolved") // @suppress("Symbol is not resolved") // @suppress("Function cannot be resolved")
    //korali::problem::bayesian::Latent problem;
    //korali::Experiment *e;
    auto e_ptr = new korali::Experiment();
    auto e = *e_ptr;
    delete e_ptr;
    auto eng = korali::Engine();
    korali::Sample* k_ptr;
    korali::Sample k;
    std::vector<std::vector<double> > current_points;		// @suppress("Type cannot be resolved") // @suppress("Method cannot be resolved") // @suppress("Symbol is not resolved") // @suppress("Function cannot be resolved")
    std::vector<std::vector<double> > mu_vectors; // @suppress("Symbol is not resolved")
    std::vector< double> mu_vectors_concat; // @suppress("Symbol is not resolved")

    for (int i = 0; i < assignments.size(); i++){	// @suppress("Type cannot be resolved") // @suppress("Method cannot be resolved") // @suppress("Symbol is not resolved") // @suppress("Function cannot be resolved")
        /* reference probability */
        current_points = points[i]; // @suppress("Symbol is not resolved")
        sigma = hyperparams[i][hyperparams[i].size() - 1]; // @suppress("Symbol is not resolved")
        // * extract mu vectors from hyperparameters
        nClusters = (hyperparams[i].size() - 1) / nDimensions; // @suppress("Method cannot be resolved") // @suppress("Symbol is not resolved")
        mu_vectors_concat = std::vector<double>(hyperparams[i].begin(), hyperparams[i].begin() + hyperparams[i].size() - 1); // @suppress("Symbol is not resolved") // @suppress("Method cannot be resolved") // @suppress("Function cannot be resolved")
        mu_vectors.resize(nClusters); // @suppress("Symbol is not resolved") // @suppress("Method cannot be resolved")
        for (size_t j = 0; j < mu_vectors.size(); j++) // @suppress("Type cannot be resolved") // @suppress("Symbol is not resolved") // @suppress("Method cannot be resolved")
        	mu_vectors[j] = std::vector<double>(hyperparams[i].begin() + nDimensions * j, hyperparams[i].begin() + nDimensions * (j+1)); // @suppress("Symbol is not resolved") // @suppress("Function cannot be resolved") // @suppress("Method cannot be resolved")

        p = multivariate_gaussian_probability(mu_vectors, nDimensions, assignments[i], nClusters, sigma, current_points); // @suppress("Function cannot be resolved") // @suppress("Symbol is not resolved")


        reset_points(distrib2._p, current_points, assignments[i], nClusters ); // @suppress("Function cannot be resolved") // @suppress("Field cannot be resolved") // @suppress("Symbol is not resolved")
        //distrib2._p.points =current_points;
        /* Update S, zeta, phi to the new set of points (needed?)*/
        distrib2_S = [&distrib2](korali::Sample& s) -> void {
    	    distrib2.S(s); } ;			 // @suppress("Method cannot be resolved")
        distrib2_zeta = [&distrib2](korali::Sample& s) -> void {
    	    distrib2.zeta(s); } ;		 // @suppress("Method cannot be resolved")
        distrib2_phi = [&distrib2](korali::Sample& s) -> void {
    	    distrib2.phi(s); } ;		  // @suppress("Method cannot be resolved")

        /* Use the distributions S, zeta and phi functions to calculate the probability */
        k_ptr = new korali::Sample();
        k = *k_ptr;
        k["Latent Variables"] = assignments[i];  // @suppress("Symbol is not resolved")
        k["Hyperparameters"] = hyperparams[i]; // @suppress("Symbol is not resolved")

        //std::function<void(korali::Sample&)> *func_ptr_S = &distrib2_S;

        distrib2_S(k);
        distrib2_zeta(k);
        distrib2_phi(k);

        double _zetaValue = k["zeta"].get<double>(); // @suppress("Method cannot be resolved") // @suppress("Symbol is not resolved")
        _sValues = k["S"].get<std::vector<double> >(); // @suppress("Type cannot be resolved") // @suppress("Method cannot be resolved") // @suppress("Symbol is not resolved")
        _phiValues = k["phi"].get<std::vector<double> >(); // @suppress("Type cannot be resolved") // @suppress("Method cannot be resolved") // @suppress("Symbol is not resolved")

        p_from_model_direct =  std::exp( - _zetaValue    													// @suppress("Type cannot be resolved") // @suppress("Method cannot be resolved") // @suppress("Symbol is not resolved") // @suppress("Function cannot be resolved")
                                         + std::inner_product(std::begin(_sValues), std::end(_sValues),		// @suppress("Type cannot be resolved") // @suppress("Method cannot be resolved") // @suppress("Symbol is not resolved") // @suppress("Function cannot be resolved")
                                                                                        std::begin(_phiValues), 0.0) );		// @suppress("Type cannot be resolved") // @suppress("Method cannot be resolved") // @suppress("Symbol is not resolved") // @suppress("Function cannot be resolved")

        assert (std::abs(p - p_from_model_direct) < 0.1*p); // @suppress("Function cannot be resolved")
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


