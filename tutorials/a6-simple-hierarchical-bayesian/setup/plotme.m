

% file = 'results_phase_1/001/s00001.json';
% file = 'results_phase_1/006/final.json';
% file = 'results_phase_2/s00001.json';
file = 'results_phase_2/final.json';
% file = 'results_phase_3b/s00005.json';
% file = 'results_phase_3b/final.json';
% file = 'results_phase_3/s00001.json';
% file = 'results_phase_3/final.json';

data = jsondecode(fileread(file));

N = length(data.Problem.Variables); 

Ns = data.Solver.PopulationSize; 

x = reshape( data.Solver.Internal.SampleDatabase, N, Ns )';
y = data.Solver.Internal.SampleFitnessDatabase;

figure(1); clf
[ h, ax, BigAx, hhist, pax ] = plotmatrix_hist( x , y );