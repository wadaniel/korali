

% file = 'results_phase_1/002/final.json';
% file = 'results_phase_2/final.json';
% file = 'results_phase_3b/s00005.json';
file = 'results_phase_3b/final.json';

data = jsondecode(fileread(file));

N = length(data.Problem.Variables); 

Ns = data.Solver.PopulationSize; 

plotmatrix_hist( reshape( data.Solver.Internal.SampleDatabase, N, Ns )' );