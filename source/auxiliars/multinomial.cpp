#include "auxiliars/multinomial.hpp"

Korali::Multinomial::Multinomial(size_t seed)
{
 _range = gsl_rng_alloc (gsl_rng_default);
 gsl_rng_set(_range, seed);
}

Korali::Multinomial::~Multinomial()
{
 gsl_rng_free(_range);
}

void Korali::Multinomial::getSelections(std::vector<double>& p, std::vector<unsigned int>& n, int N )
{
 gsl_ran_multinomial(_range, p.size(), N, p.data(), n.data());
}
