#ifndef _UTILS_CPP_
#define _UTILS_CPP_


#include "utils.hpp"

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
        std::vector<double> mu = mus[c]; // @suppress("Symbol is not resolved") // @suppress("Type cannot be resolved")
        // update p
        p = p * univariate_gaussian_probability(mu, sigma, points[i]); // @suppress("Function cannot be resolved")
    }
    return p;
}

double univariate_gaussian_probability(std::vector<double> mu, double sigma, std::vector<double> point){ // @suppress("Type cannot be resolved") // @suppress("Method cannot be resolved") // @suppress("Function cannot be resolved") // @suppress("Symbol is not resolved")
    assert (mu.size() == point.size());							// @suppress("Symbol is not resolved") // @suppress("Type cannot be resolved") // @suppress("Method cannot be resolved") // @suppress("Function cannot be resolved")
    std::vector<double> distance(mu.size());					// @suppress("Symbol is not resolved") // @suppress("Type cannot be resolved") // @suppress("Method cannot be resolved") // @suppress("Function cannot be resolved")
    std::transform(mu.begin(), mu.end(), point.begin(),			// @suppress("Symbol is not resolved") // @suppress("Type cannot be resolved") // @suppress("Method cannot be resolved") // @suppress("Function cannot be resolved")
               distance.begin(), std::minus<double>());			// @suppress("Symbol is not resolved") // @suppress("Type cannot be resolved") // @suppress("Method cannot be resolved") // @suppress("Function cannot be resolved")
    double squared_distance = std::inner_product(std::begin(distance), std::end(distance) , std::begin(distance), 0.0);    // @suppress("Type cannot be resolved") // @suppress("Method cannot be resolved") // @suppress("Function cannot be resolved") // @suppress("Symbol is not resolved")
    double exponent = - squared_distance / (2.0 * std::pow(sigma, 2.0)); // @suppress("Function cannot be resolved")
    double denominator = (std::sqrt(2.0 * M_PI) * sigma) ; // @suppress("Symbol is not resolved") // @suppress("Function cannot be resolved")
    double probability =  std::exp(exponent) / denominator;  // @suppress("Type cannot be resolved") // @suppress("Method cannot be resolved") // @suppress("Function cannot be resolved") // @suppress("Symbol is not resolved")
    return probability;
}




// ****************** vector operations ********************

double l2_norm(std::vector<double> const& u) {
    // copy-paste from http://polaris.s.kanazawa-u.ac.jp
    double accum = 0.;
    for (double x : u) {
        accum += x * x;
    }
    return sqrt(accum);
}
double l2_norm_squared(std::vector<double> const& u) {
    // copy-paste from http://polaris.s.kanazawa-u.ac.jp
    double accum = 0.;
    for (double x : u) {
        accum += x * x;
    }
    return accum;
}


// Todo: maybe creating an in-place version of this could improve performance? (Think about that when the rest works)
std::vector<double> normalize(std::vector<double> const& u){
	double norm = l2_norm(u);
	std::vector<double> result(u.size(), 0.);
	std::transform(u.begin(), u.end(), result.begin(), [norm](auto& c){return c / norm;});
	return result;
}





#endif
