clc; clear

file = '_korali_result/gen00000003.json';
% file = '../../tutorials/a3-bayesian-inference/_korali_result/gen00000007.json';

data = jsondecode(fileread(file));

N = length(data.Variables);
Ns = data.Solver.PopulationSize;
x = data.Solver.Internal.SampleDatabase;

% x = data.Solver.Internal.SampleDatabase;
% Ns = data.Solver.Internal.DatabaseEntryCount;
% dim = length(data.Solver.Internal.SampleDatabase)/Ns;
%x = reshape(x,dim,Ns)';

%%
fig = figure(); clf
% fig.Position = [539 130 1021 825];

[ h, ax, BigAx, hhist, pax ] = plotmatrix_hist( x );