%   Create synthetic data using the model defined in my_model
%
clear
folder_name = './';

addpath('./logistic_model/');

FLAG = 1;

N = 10; %number of different data sets
x = repmat( 0:0.5:10, N,1 );


Nx = size(x,2);

y = zeros(N,Nx);
theta = zeros(N,3);

for i = 1:N
   
    % set nominal parameters
%     r = binornd(1,0.5);
%     theta(i,1) = r*300 + (1-r)*150;
    theta(i,1) = 300;
    theta(i,2) = normrnd(40,10);
    theta(i,3) = lognrnd(0,0.5);
    theta(i,4) = 5;
    
    % evaluate the model at the nominal values
    y(i,:) = my_model( x(i,:), theta(i,1:3), FLAG );
    
    % add error to the data
    error = normrnd( 0, theta(i,4), 1,Nx);
    y(i,:) = y(i,:) + error;
    
end
%%
pl = plot(x',y','.-');
set(pl,'LineWidth',3);
set(pl,'MarkerSize',30)
set(gca,'FontSize',20);
grid on
axis tight


%%

for i=1:N
  file_name = ['data_set_' sprintf('%03d',i) '.dat'];
  dlmwrite( file_name, [x(i,:); y(i,:)]', ' ');  
end

dlmwrite( 'theta.txt', theta, ' ');

