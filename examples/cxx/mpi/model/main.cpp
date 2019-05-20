#include "jacobi.h"
#include <mpi.h>
#include <vector>

int main(int argc, char* argv[])
{
 int myRank;
 MPI_Init(&argc, &argv);
 MPI_Comm_rank(MPI_COMM_WORLD, &myRank);

 std::vector<double> xdata;
 std::vector<double> parameters;
 std::vector<double> results;

 xdata.push_back(0.364); xdata.push_back(0.944); xdata.push_back(0.560);
 xdata.push_back(0.151); xdata.push_back(0.863); xdata.push_back(0.983);
 xdata.push_back(0.292); xdata.push_back(0.979); xdata.push_back(0.876);
 xdata.push_back(0.451); xdata.push_back(0.014); xdata.push_back(0.150);
 xdata.push_back(0.893); xdata.push_back(0.935); xdata.push_back(0.316);
 xdata.push_back(0.093); xdata.push_back(0.287); xdata.push_back(0.921);
 xdata.push_back(0.671); xdata.push_back(0.565); xdata.push_back(0.078);
 xdata.push_back(0.565); xdata.push_back(0.779); xdata.push_back(0.974);

 parameters.push_back(+3.71); // X0
 parameters.push_back(-2.34); // X1
 parameters.push_back(+1.32); // Y0
 parameters.push_back(+0.78); // Y1
 parameters.push_back(+2.25); // Z0
 parameters.push_back(+1.81); // Z1

 jacobi(xdata, parameters, results, MPI_COMM_WORLD);

 if (myRank == 0) for(size_t i = 0; i < results.size(); i++) printf("Pos (%.3f, %.3f, %.3f) - Value: %f\n", xdata[i*3 + 0], xdata[i*3 + 1], xdata[i*3 + 2], results[i]);
}
