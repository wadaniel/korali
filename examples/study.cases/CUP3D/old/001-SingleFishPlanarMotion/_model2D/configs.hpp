//  Korali environment for CubismUP-2D
//  Copyright (c) 2020 CSE-Lab, ETH Zurich, Switzerland.

std::string OPTIONS = "-bpdx 4 -bpdy 4 -levelMax 7 -levelStart 4 -Rtol 10000.0 -Ctol 100.0 -extent 2 -CFL 0.7 -poissonTol 1e-5 -poissonTolRel 0.0 -bMeanConstraint 1 -bAdaptChiGradient 0 -tdump 0 -nu 0.00004 -tend 0 -muteAll 0 -verbose 1";

std::vector<std::vector<double>> initialPositions{{{1.0, 1.0}}};
