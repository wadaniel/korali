addpath('trandn/');
%
clear

folder_name = '../setup/data/';

N = 10; %number of different data sets

Ns = 100;

y = zeros(N,Ns);
theta = zeros(N,2);

for i = 1:N
  
  theta(i,1) = normrnd(10,4);
  theta(i,2) = lognrnd(0,1);

  mu    = theta(i,1);
  sigma = theta(i,2);
    
  % evaluate the model at the nominal values
%   for k = 1:Ns
%     X = trandn( (a-mu)/sigma, (b-mu)/sigma );
%     y(i,k) = mu + sigma*X;
%   end
  y(i,:) = normrnd(mu,sigma,1,Ns);
    
end

%%
figure(1); clf
for i=1:N
  pl = histogram(y(i,:));
  hold on
end
grid on
axis tight


%%

for i=1:N
  file_name = [ folder_name '/data_set_' sprintf('%03d',i-1) '.dat'];
  dlmwrite( file_name, y(i,:)', ' ');  
end

dlmwrite( [folder_name 'theta.txt'], theta, ' ');
theta
