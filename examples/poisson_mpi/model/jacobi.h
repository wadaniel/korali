/**********************************************************************/
// A generic stencil-based 3D Jacobi solver                           //
// Course Material for HPCSE-II, Spring 2019, ETH Zurich              //
// Authors: Sergio Martin                                             //
// License: Use if you like, but give us credit.                      //
/**********************************************************************/

#include "mpi.h"
#include <vector>

void jacobi(std::vector<double> xdata, std::vector<double>& parameters, std::vector<double>& results, MPI_Comm comm);
