/*
 * Test whether the distribution parameterized by S, zeta and phi in model.cpp is correct or not
*/
#include "model.hpp"
#include "korali.hpp"
#include "load_data.hpp"

#include "test_model.hpp"

#include <math.h>


int test_distribution_1(){

    // initialize the distribution
    ExampleDistribution1 distrib1 = ExampleDistribution1();
    std::function<void(korali::Sample& s)> distrib1_S = [&distrib1](korali::Sample& s) -> void {   
    	distrib1.S(s); } ;  
    std::function<void(korali::Sample& s)> distrib1_zeta = [&distrib1](korali::Sample& s) -> void {   
    	distrib1.zeta(s); } ;  
    std::function<void(korali::Sample& s)> distrib1_phi = [&distrib1](korali::Sample& s) -> void {  
    	distrib1.phi(s); } ;  


    int nClusters = 1;
    int nDimensions = distrib1._p.nDimensions; // 2  
    int d2_numberLatentVars = 1;
    int d2_numberHyperparams =  1;

     // Some value pairs:
    std::vector<double> mu0 = {0, 0};				    
    std::vector<double> mu1 = {2, 2};				    
    std::vector<double> mu2 = {-0.5, 10};			    
    double sigma = 1.0;
    std::vector<std::vector<double> > hyperparams;
    std::vector<std::vector<double> > mus;
    std::vector<std::vector<std::vector<double> > > points;
    /* The test inputs*/
    hyperparams.push_back({sigma});
    mus.push_back(mu0);
    points.push_back( {
        {0, 0}
        });
    hyperparams.push_back({0.2});
    mus.push_back(mu0);
    points.push_back( {
        {0, 0}, {0.5, 0.25}, {-1, 0.01}
        });
    hyperparams.push_back({0.5});
    mus.push_back(mu2);
    points.push_back( {
        {0, 9}, {0.9, 10.5}
        });
    hyperparams.push_back({4.});
    mus.push_back(mu1);
    points.push_back( {
        {0, 9}, {0.9, 10.5}, {4.0, 4.01}, {3., 2.5}, {-0.5, 1.9}, {-0.5, 1.9}
        });


    double p;
    double p_from_model_direct;
    double _zetaValue;
    std::vector<double> _sValues;		    
    std::vector<double> _phiValues;		    
    auto e_ptr = new korali::Experiment();
    auto e = *e_ptr;
    delete e_ptr;
    auto eng = korali::Engine();
    korali::Sample* k_ptr;
    korali::Sample k;
    std::vector<std::vector<double> > current_points;		    
    std::vector<double > mu_vector; 



    for (int i = 0; i < hyperparams.size(); i++){	    
        /* reference probability */
        current_points = points[i]; 
        sigma = hyperparams[i][0]; 
        // * extract mu vectors from hyperparameters
        mu_vector = mus[i];    

        p = 1;
        for (std::vector<double> pt : current_points){
            double p_single = univariate_gaussian_probability(mu_vector, sigma, pt);  
            p *= p_single;
        }

        reset_points(distrib1._p, current_points, std::vector<int>(current_points.size(), 0), nClusters );

        /* Update S, zeta, phi to the new set of points (needed?)*/
        distrib1_S = [&distrib1](korali::Sample& s) -> void {
    	    distrib1.S(s); } ;			  
        distrib1_zeta = [&distrib1](korali::Sample& s) -> void {
    	    distrib1.zeta(s); } ;		  
        distrib1_phi = [&distrib1](korali::Sample& s) -> void {
    	    distrib1.phi(s); } ;		   

        /* Use the distributions S, zeta and phi functions to calculate the probability */
        k_ptr = new korali::Sample();
        k = *k_ptr;
        k["Latent Variables"] = mu_vector;  
        k["Hyperparameters"] = hyperparams[i];  


        distrib1_S(k);
        distrib1_zeta(k);
        distrib1_phi(k);

        double _zetaValue = k["zeta"].get<double>();  
        _sValues = k["S"].get<std::vector<double> >();  
        _phiValues = k["phi"].get<std::vector<double> >();  

        p_from_model_direct =  std::exp( - _zetaValue    													 
                                                 + std::inner_product(std::begin(_sValues), std::end(_sValues),		 
                                                                                    std::begin(_phiValues), 0.0) );		
                                                                                    
        assert (std::abs(p - p_from_model_direct) < 0.1*p);  
        delete k_ptr;

    }

    return 0;
}



/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */




int test_distribution_2(){

    // Initialize the distribution
	ExampleDistribution2  distrib2 = ExampleDistribution2();   

    std::function<void(korali::Sample& s)> distrib2_S = [&distrib2](korali::Sample& s) -> void {    
    	distrib2.S(s); } ;  
    std::function<void(korali::Sample& s)> distrib2_zeta = [&distrib2](korali::Sample& s) -> void {    
    	distrib2.zeta(s); } ;  
    std::function<void(korali::Sample& s)> distrib2_phi = [&distrib2](korali::Sample& s) -> void {   
    	distrib2.phi(s); } ;  


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
    hyperparams.push_back({-1.5, 0, 20., 5., 3.0, 3.5, 7.5}); // (mu1, mu2, mu3, sigma)
    points.push_back( {
        {1.5, -1.5}, {-0.5, 2}, {-5., 7.,},
		{22.0, 10.3}, {17.5, 30.},     {0.0, 7.2}
        });



    double p;
    double p_from_model_direct;
    double p_from_model_via_korali;
    double _zetaValue;
    std::vector<double> _sValues;		     
    std::vector<double> _phiValues;
    auto e_ptr = new korali::Experiment();
    auto e = *e_ptr;
    delete e_ptr;
    auto eng = korali::Engine();
    korali::Sample* k_ptr;
    korali::Sample k;
    std::vector<std::vector<double> > current_points;		     
    std::vector<std::vector<double> > mu_vectors;  
    std::vector< double> mu_vectors_concat;  

    for (int i = 0; i < assignments.size(); i++){	     
        /* reference probability */
        current_points = points[i];  
        sigma = hyperparams[i][hyperparams[i].size() - 1];  
        // * extract mu vectors from hyperparameters
        nClusters = (hyperparams[i].size() - 1) / nDimensions;    
        mu_vectors_concat = std::vector<double>(hyperparams[i].begin(), hyperparams[i].begin() + hyperparams[i].size() - 1);     
        mu_vectors.resize(nClusters);    
        for (size_t j = 0; j < mu_vectors.size(); j++)     
        	mu_vectors[j] = std::vector<double>(hyperparams[i].begin() + nDimensions * j, hyperparams[i].begin() + nDimensions * (j+1));     

        p = multivariate_gaussian_probability(mu_vectors, nDimensions, assignments[i], nClusters, sigma, current_points);   


        reset_points(distrib2._p, current_points, assignments[i], nClusters );
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

        distrib2_S(k);
        distrib2_zeta(k);
        distrib2_phi(k);

        double _zetaValue = k["zeta"].get<double>();    
        _sValues = k["S"].get<std::vector<double> >();     
        _phiValues = k["phi"].get<std::vector<double> >();     

        p_from_model_direct =  std::exp( - _zetaValue    													     
                                         + std::inner_product(std::begin(_sValues), std::end(_sValues),		     
                                                                                        std::begin(_phiValues), 0.0) );		     

        assert (std::abs(p - p_from_model_direct) < 0.1*p); 
        delete k_ptr;
    }


    return 0;


}



int main(int argc, char* argv[])
{
    int success1 = test_distribution_1();
	int success2 = test_distribution_2();
	assert ((success1 == 0) && (success2 == 0));

    return 0;
}


