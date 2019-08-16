addpath('trandn/');
%
clear

folder_name = '../setup/data/';

N = 10; %number of different data sets

Ns = 100;

y = zeros(N,Ns);
theta = zeros(N,3);

for i = 1:N
  
    condition = true;
    while( condition )
      theta(i,1) = unifrnd(-5,5);
      theta(i,2) = unifrnd(15,25);
      theta(i,3) = normrnd(10,4);
      theta(i,4) = lognrnd(0,1);

      a = theta(i,1);
      b = theta(i,2);
      mu = theta(i,3);
      sigma = theta(i,4);
      
      if( a<mu && b>mu)
        condition = false;
      end
      
    end
    
    % evaluate the model at the nominal values
    for k = 1:Ns
      X = trandn( (a-mu)/sigma, (b-mu)/sigma );
      y(i,k) = mu + sigma*X;
    end
    
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
  file_name = [ folder_name '/data_set_' sprintf('%03d',i) '.dat'];
  dlmwrite( file_name, y(i,:)', ' ');  
end

dlmwrite( [folder_name 'theta.txt'], theta, ' ');
theta
