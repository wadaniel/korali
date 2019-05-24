#ifndef _g09_H_
#define _g09_H_

#include <vector>

// Minus Rosenbrock
void g09(std::vector<double>& pars, std::vector<double>& result);

double g1(const double *x, size_t N);
double g2(const double *x, size_t N);
double g3(const double *x, size_t N);
double g4(const double *x, size_t N);
#endif
