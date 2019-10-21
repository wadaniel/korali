


clear;clc

x=[1 2 3]';
y=[4 5 6 7]';
z=[8 9 10 11 12]';

nx = length(x);
ny = length(y);
nz = length(z);

[xx,yy,zz] = meshgrid(x,y,z);

c=1;

xyz = zeros(nx*ny*nz,4);

for i=1:nx
    for j=1:ny
        for k=1:nz
            
            f = z(k);
            xyz(c,:) = [ x(i) y(j) z(k) f];
            c = c + 1;
            
        end
    end
end


dlmwrite('qwe.dat',xyz);


% %%
% ff = xx;
% h = slice( xx,yy,zz, ff, 3, 6, [] );
% colorbar
% %%

%%

a=load('qwe.dat');
xx = sort(unique(a(:,1)));
yy = sort(unique(a(:,2)));
zz = sort(unique(a(:,3)));
%%
f = a(:,4);
ff = reshape(f,[nz ny nx]);
ff = permute(ff,[2 3 1]);
[xx,yy,zz] = meshgrid(xx,yy,zz);

h = slice( xx,yy,zz, ff, [], [], 12 ); 
ax = gca;
h=rotate3d(ax);
h.Enable = 'on';

colorbar
