/*
- File mvf.cpp
- Original compiled by Ernie.
- Revisions:
- 2003.01.30 Ver 0.1
- 2004.03.05 Added EggHolder, StretchedV, Rana functions
- from http://www.ft.utb.cz/people/zelinka/soma/
- 2004.04.19 http://www.mat.univie.ac.at/~neum/glopt/janka/funcs.html
- Bohachevsky1, Bohachevsky2, Easom
- Hyperellipsoid, Neumaier2, Neumaier3,
- Oddsquare, Schaffer1, Schaffer2, Zimmerman
- LennardJones
- Corana
- Xor
*/

/* 
	Credits:
	https://www.sfu.ca/~ssurjano/other.html
*/


#include <stdio.h>
#include <math.h>
#include "rnd.h"
#include "mvf.h"

// ============================================== //
double sqr( double x )
{
return x*x;
};
double dmax(int n, double *x)
{
int i;
double m = x[0];
for (i = 1; i < n; i++) {
if (x[i] > m) m = x[i];
}
return m;
}
double sgn(double x)
{
if (x < 0.0) return -1.0;
return 1.0;
}
double dist2(int n, double *x, double *y)
{
register int i;
double s, t;
s = 0.0;
for (i = 1; i < n; i++) {
t = (x[i]-y[i]);
s += t * t;
}
return s;
}
double distInf(int n, double *x, double *y)
{
register int i;
double t,m;
m = fabs(x[0] - y[0]);
for (i = 1; i < n; i++) {
t = fabs(x[i]-y[i]);
if (t > m) {
m = t;
}
}
return m;
}
double prod(int n, double *x)
{
int i;
double p;
p = 1.0;
for (i = 0; i < n; i++) {
p *= x[i];
}
return -p;
}
double mvfAckley(int n, double *x)
/*
-Dimension: n arbitrary
- Domain: | x_i | <= 30.0
- Minimum 0 at x_i = 0.0
*/
{
int i;
double t, s1, s2;
s1 = s2 = 0.0;
for (i = 0; i < n; i++) {
t = x[i];
s1 += t * t;
s2 += cos(2.0*M_PI * x[i]);
}
return -20.0 * exp(-0.2 * sqrt(s1/n)) - exp(s2 / n) +
20.0 + M_E;
}

double mvfBeale(int n, double *x)
/*
Dimension: 2
Domains: | x_i | le 4.5
Minimum: 0 at (3, 0.5)
*/
{
return pow(1.5 - x[0] + x[0]*x[1], 2) +
pow(2.25 - x[0] + x[0] * x[1]*x[1], 2) +
pow(2.625 - x[0] + x[0] * pow(x[1], 3), 2);
}
double mvfBohachevsky1(int n, double *x)
/*
- Dimension: 2
- Global minimum: 0 at (0,0)
- Domain: |x[i]| <= 50.0
*/
{
return x[0] * x[0] + 2.0 * x[1]*x[1]
- 0.3 * cos(3.0 * M_PI * x[0])
- 0.4 * cos(4.0 * M_PI * x[1])
+ 0.7;
}
double mvfBohachevsky2(int n, double *x)
/*
- Dimension: 2
- Global minimum: 0 at (0,0)
- Domain: |x[i]| <= 50.0
*/
{
return x[0] * x[0] + 2.0 * x[1]*x[1]
- 0.3 * cos(3.0 * M_PI * x[0]) * cos(4.0 * M_PI * x[1])
+ 0.3;
}
double mvfBooth(int n, double *x)
/*
Dimension: 2
Domain: |x_i| <= 10
Minimum: 0 at (1,3)
*/
{
return pow(x[0] + 2*x[1] - 7, 2) +
pow(2 * x[0] + x[1] - 5, 2);
}
double mvfBoxBetts(int n, double * x)
/*
Dimension: 3
Domain: [0.9, 1.2], [9, 11.2], [0.9, 1.2]
Minimum: 0 at (1, 10, 1)
*/
{
int i;
double x0=x[0],x1=x[1],x2=x[2] ;
double sum=0.0 ;
for (i=1; i<=10 ; i++)
sum+=pow(exp(-0.1*i*x0)-exp(-0.1*i*x1)-
(exp(-0.1*i)-exp(-1.0*i))*x2,2.0);
return sum ;
}
double mvfBranin(int n, double *x)
/*
- n = 2;
- Domain: -5.0 < x[0] <= 10.0, 0 <= x[1] <= 15
- Minimum: 0.397889 at (-3.142, 12.275),
(3.142, 2.275),
(9.425, 2.425)
*/
{
double s, x0;
x0 = x[0];
s = x[1] -
( 5.1/(4.0*M_PI*M_PI) * x0 - 5.0/M_PI) * x0 - 6.0;
return s*s + 10*(1.0 - 1.0/(8.0*M_PI)) * cos(x0) + 10.0;
}
double mvfBranin2(int n, double *x)
{
return pow(1.0-2.0*x[1]+sin(4.0*M_PI*x[1])/20.0-x[0],2)+
pow(x[1]-sin(2.0*M_PI*x[0])/2.0,2);
}
double mvfCamel3(int n, double *x)
/*
- Dimension: 2
- Domain: | x[i] | <= 5
- Minimum: 0 at (0,0)
*/
{
double x02 = x[0] * x[0];
double x04 = x02 * x02;
double x06 = x04 * x02;
return 2*x02-1.05*x04 +x06/6.0+x[0]*x[1]*x[1]*x[1];
}
double mvfCamel6(int n, double *x)
/*
Dimension: 2
*/
{
double x1=x[0],x2=x[1] ;
return 4.0*x1*x1-0.21E1*pow(x1,4.0)
+ pow(x1,6.0)/3+x1*x2-4.0*x2*x2
+ 4.0*pow(x2,4.0);
}
double mvfChichinadze(int n, double *x)
/*
- Dimension: 2
- Domain: x[0] <= 30, x[1] <= 10
- f* = 43.3159
- x* = (5.90133, 0.5)
*/
{
return x[0]*x[0]-12 *x[0]+11+10*cos(M_PI/2.0*x[0])
+ 8 * sin(5 * M_PI * x[0])
- exp ( -(x[1] - 0.5) * 0.5) / sqrt(5.0);
}
static double dis[46] = {
1.27,
1.69,1.43,
2.04,2.35,2.43,
3.09,3.18,3.26,2.85,
3.20,3.22,3.27,2.88,1.55,
2.86,2.56,2.58,2.59,3.12,3.06,
3.17,3.18,3.18,3.12,1.31,1.64,3.00,
3.21,3.18,3.18,3.17,1.70,1.36,2.95,1.32,
2.38,2.31,2.42,1.94,2.85,2.81,2.56,2.91,2.97
};
double mvfCola( int n, double * x )
{
double sum = 0.0, temp;
int i, j, t, k = 1;
double mt[20] = {0, 0, 0, 0};
for( i = 4; i < 20; i++)
mt[i] = x[i-3];
for( i = 1; i < 10; i++)
for( j = 0; j < i; j++) {
temp = 0.0;
for( t = 0; t < 2; t++ )
temp += sqr( mt[i*2+t]-mt[j*2+t] );
sum += sqr( dis[k-1] - sqrt(temp) );
k++;
}
return sum;
}
double mvfColville(int n, double *x)
/*
Dimension: 4
Domain: | x[i] <= 10
Global minimum: 0 at (1,1,1,1)
*/
{
return 100 * pow(x[0]-x[1]*x[1], 2)
+ pow(1-x[0], 2)
+ 90 * pow(x[3] -x[2]*x[2], 2)
+ pow(1-x[2], 2)
+ 10.1 * (pow(x[1] -1, 2) + pow(x[3] - 1, 2))
+ 19.8 * (x[1] - 1)*(x[3] - 1);
}
double mvfCorana(int n, double *x)
/*
- Dimension 4
- Domain: | x[i] | < 1000
*/
{
int i;
double s;
double z;
double d[] = {1, 1000, 10, 100};
s = 0.0;
for (i = 0; i < 4; i++) {
z = floor( fabs(x[i]/0.2) + 0.49999) * sgn(x[i]) * 0.2;
if (fabs(x[i] - z) < 0.05) {
s += 0.15 * sqr(z - 0.05 * sgn(z)) * d[i];
} else {
s += d[i] * x[i]*x[i];
}
};
return s;
}
double mvfEasom(int n, double *x)
/*
-Dim 2
- Global minimum -1 at (pi, pi)
- Domain: | x[i] | <= 100
*/
{
return -cos(x[0])
* cos(x[1]) *
exp(- sqr(x[0] - M_PI) -
sqr(x[1] - M_PI));
}
double mvfEggholder(int n, double *x)
/*
- Dimension: n
- Domain: | x_i | < 512
*/
{
int i;
double sum;
sum = 0.0;
for (i = 0; i < n-1; i++) {
sum += -(x[i+1] + 47.0) * sin(sqrt(fabs(x[i+1] + x[i] * 0.5 + 47.0))) +
sin(sqrt(fabs(x[i] - (x[i+1] + 47.0)))) * (-x[i]);
}
return sum;
}
double mvfExp2(int n, double *x)
/*
Dimension: 2
Domain: 0 <= x[i] <= 20
Global minimum: 0 at (1, 10)
*/
{
register int i;
double sum = 0.0;
double t;
for (i = 0; i < 10; i ++) {
t = exp(-i * x[0] / 10.0)
- 5 * exp(- i * x[1] * 10) - exp(-i / 10.0)
+ 5 * exp(-i);
sum += t * t;
}
return sum;
}
double mvfFraudensteinRoth(int n, double *x)
/*
- Dimension 2
*/
{
return sqr(-13.0 + x[0] + ((5 - x[1]) * x[1] - 2) * x[1])
+ sqr(-29.0 + x[0] + ((x[1] + + 1.0) * x[1] - 14) * x[1]);
}
double mvfGear(int n, double *x)
{
double t;
/* n == 4 always */
t = 1.0 / 6.931 - floor(x[0]) * floor(x[1]) / (floor(x[2]) * floor(x[3]));
return t * t;
}
double mvfGeneralizedRosenbrock(int n, double * x)
{
double s = 0.0;
int i;
for( i = 1; i < n; i++ )
s += 100.0*pow(x[i]-x[i-1]*x[i-1],2)
+ pow(x[i-1]-1.0,2);
return s;
}
double mvfGoldsteinPrice(int n, double * x)
/*
-Dimension: n = 2 always
- Domain: | x_i | <= 2
- Minimum value 3 at (0, -1)
*/
{
return (1.0+pow(x[0]+x[1]+1.0,2)*
(19.0-14.0*x[0]+3.0*x[0]*x[0]-14.0*x[1]
+6.0*x[0]*x[1]+3.0*x[1]*x[1]))*
(30.0+pow(2.0*x[0]-3.0*x[1],2)*
(18.0-32.0*x[0]+12.0*x[0]*x[0]+48.0*x[1]
-36.0*x[0]*x[1]+27.0*x[1]*x[1]));
}
double mvfGriewank(int n, double * x)
{
int i;
double sum=0 ;
double prod=1 ;
for (i=0 ; i<10 ; i++) {
sum+=x[i]*x[i];
prod*=cos(x[i]/sqrt((double)(i+1))) ;
}
return sum/4000.0-prod + 1 ;
}
double mvfHansen(int n, double * x)
{
return (cos(1.0)+2.0*cos(x[0]+2.0)
+3.0*cos(2.0*x[0]+3.0)+4.0*cos(3.0*x[0]+4.0)
+5.0*cos(4.0*x[0]+5.0))*(cos(2.0*x[1]+1.0)
+2.0*cos(3.0*x[1]+2.0)
+3.0*cos(4.0*x[1]+3.0)
+4.0*cos(5.0*x[1]+4.0)
+5.0*cos(6.0*x[1]+5.0));
}
double mvfHartman3(int n, double *x)
{
int i, j;
double s, t, t1;
static double a[4][3] = {
{ 3, 10, 30},
{0.1, 10, 35},
{ 3, 10, 30},
{0.1, 10, 35}
};
double c[] = { 1, 1.2, 3, 3.2 };
double p[4][3] = {
{0.3689, 0.1170, 0.2673},
{0.4699, 0.4387, 0.7470},
{0.1091, 0.8732, 0.5547},
{0.03815,0.5743, 0.8828}
};
n = 3; // forced check
s = 0.0;
for (i = 0; i < 4; i++) {
t = 0.0;
for (j = 0; j < n; j++) {
t1 = x[j]-p[i][j];
t += a[i][j] * (t1*t1);
}
s += c[i] * exp(-t);
}
return -s;
}
double mvfHartman6(int n, double *x)
/*
- Harman_6 (m = 4)
- Domain: 0 <= x_i <= 1
- Minimum value -3.32 at (0.201, 0.150, 0.477, 0.275, 0.311, 0.657)
*/
{
int i, j;
double s, t, t1;
static double a[4][6] = {
{10, 3, 17, 3.5, 1.7, 8.0},
{0.05, 10, 17, 0.1, 8, 14},
{3, 3.5, 1.7, 10, 17, 8},
{17, 8, 0.05, 10, 0.1, 14}
};
double c[] = {1, 1.2, 3, 3.2};
double p[4][6] = {
{0.1312, 0.1696, 0.5569, 0.0124, 0.8283, 0.5886},
{0.2329, 0.4135, 0.8307, 0.3736, 0.1004, 0.9991},
{0.2348, 0.1415, 0.3522, 0.2883, 0.3047, 0.6650},
{0.4047, 0.8828, 0.8732, 0.5743, 0.1091, 0.0381}
};
n = 6; // forced check
s = 0.0;
for (i = 0; i < 4; i++) {
t = 0.0;
for (j = 0; j < n; j++) {
t1 = x[j]-p[i][j];
t += a[i][j] * (t1*t1);
}
s += c[i] * exp(-t);
};
return -s;
}
double mvfHimmelblau(int n, double *x)
/*
- Dimension: 2
- Domain: | x[i] | < 5.0
- Global minimum: 0 at (3,2).
*/
{
return sqr(x[0]*x[0]+x[1]-11.0) + sqr(x[0]+x[1]*x[1]-7);
}
double mvfHolzman1(int n, double *x)
/*
Dimension: 3
Domain: 0.1 <= x[0] <= 100.0
0 <= x[1] <= 25.6
0 <= x[2] <= 5
Global minimum: 0 at (50, 25, 1.5)
*/
{
int i;
double sum = 0.0;
double ui;
for (i = 0; i < 100; i++) {
ui = 25 + pow(-50.0*log(0.01*(i+1)),2.0/3.0);
sum += -0.1 * (i+1)
+exp(1/ x[0] * pow(ui-x[1], x[2]));
}
return sum;
}
double mvfHolzman2(int n, double *x)
/*
Dimension: n
Domain: | x[i] | <= 10
Global minimum: 0 at x[i] = 0
*/
{
int i;
double sum = 0.0;
for (i = 0; i < n; i++) {
sum += i * pow(x[i] , 4);
}
return sum;
}
double mvfHosaki(int n, double *x)
/*
-n = 2
*/
{
return (1 + x[0] *
(-8 + x[0] *
(7 + x[0] *
(-7.0/3.0 + x[0] *1.0/4.0
)))) *
x[1]*x[1] * exp(-x[1]);
}
double mvfHyperellipsoid(int n, double *x)
/*
-n = 30
-Domain: |x| <= 1.0
*/
{
int i;
double s = 0.0;
for (i = 0; i < n; i++) {
s += i * i + x[i]*x[i];
}
return s;
}
double mvfKatsuuras(int n, double *x)
/*
= Dimension: n (10)
- Domain: | x[i] | <= 1000
- Global minimum 1.0 at 0 vector.
*/
{
int i, k;
int d = 32;
double prod;
double s;
double pow2;
prod = 1.0;
for (i = 0; i < n; i++) {
s = 0.0;
for (k = 1; k <= d; k++) {
pow2 = pow(2, k);
s += round(pow2 * x[i]) / pow2;
}
prod *= 1.0 + (i+1) * s;
}
return prod;
}
double mvfKowalik(int n, double *x)
/*
-n = 4
-Domain: | x_i | < 5.0
-Global minimum is 0.0 at x_i = 0.00
*/
{
static double b[] = {
4.0, 2.0, 1.0, 1/2.0,1/4.0,1/6.0,1/8.0,
1/10.0,1/12.0,1/14.0, 1/16.0 };
static double a[] = {
0.1957, 0.1947, 0.1735, 0.1600, 0.0844,
0.0627, 0.0456, 0.0342, 0.0323, 0.0235,
0.0246 };
int i;
double sum;
double bb, t;
sum = 0.0;
for (i = 0; i < 11; i++) {
bb = b[i] * b[i];
t = a[i]-(x[0]*(bb+b[i]*x[1])/(bb+b[i]*x[2]+x[3]));
sum += t * t;
}
return sum;
}
double afox10[30][10] = {
{ 9.681,0.667,4.783, 9.095, 3.517, 9.325, 6.544, 0.211, 5.122, 2.020},
{9.400, 2.041, 3.788, 7.931, 2.882, 2.672, 3.568, 1.284, 7.033, 7.374},
{8.025, 9.152, 5.114, 7.621, 4.564, 4.711, 2.996, 6.126, 0.734, 4.982},
{2.196, 0.415, 5.649, 6.979, 9.510, 9.166, 6.304, 6.054, 9.377, 1.426},
{8.074, 8.777, 3.467, 1.863, 6.708, 6.349, 4.534, 0.276, 7.633, 1.567},
{7.650, 5.658, 0.720, 2.764, 3.278, 5.283, 7.474, 6.274, 1.409, 8.208},
{1.256, 3.605, 8.623, 6.905, 0.584, 8.133, 6.071, 6.888, 4.187, 5.448},
{8.314, 2.261, 4.224, 1.781, 4.124, 0.932, 8.129, 8.658, 1.208, 5.762},
{0.226, 8.858, 1.420, 0.945, 1.622, 4.698, 6.228, 9.096, 0.972, 7.637},
{7.305, 2.228, 1.242, 5.928, 9.133, 1.826, 4.060, 5.204, 8.713, 8.247},
{0.652, 7.027, 0.508, 4.876, 8.807, 4.632, 5.808, 6.937, 3.291, 7.016},
{2.699, 3.516, 5.874, 4.119, 4.461, 7.496, 8.817, 0.690, 6.593, 9.789},
{8.327, 3.897, 2.017, 9.570, 9.825, 1.150, 1.395, 3.885, 6.354, 0.109},
{2.132, 7.006, 7.136, 2.641, 1.882, 5.943, 7.273, 7.691, 2.880, 0.564},
{4.707, 5.579, 4.080, 0.581, 9.698, 8.542, 8.077, 8.515, 9.231, 4.670},
{8.304, 7.559, 8.567, 0.322, 7.128, 8.392, 1.472, 8.524, 2.277, 7.826},
{8.632, 4.409, 4.832, 5.768, 7.050, 6.715, 1.711, 4.323, 4.405, 4.591},
{4.887, 9.112, 0.170, 8.967, 9.693, 9.867, 7.508, 7.770, 8.382, 6.740},
{2.440, 6.686, 4.299, 1.007, 7.008, 1.427, 9.398, 8.480, 9.950, 1.675},
{6.306, 8.583, 6.084, 1.138, 4.350, 3.134, 7.853, 6.061, 7.457, 2.258},
{0.652, .343, 1.370, 0.821, 1.310, 1.063, 0.689, 8.819, 8.833, 9.070},
{5.558, 1.272, 5.756, 9.857, 2.279, 2.764, 1.284, 1.677, 1.244, 1.234},
{3.352, 7.549, 9.817, 9.437, 8.687, 4.167, 2.570, 6.540, 0.228, 0.027},
{8.798, 0.880, 2.370, 0.168, 1.701, 3.680, 1.231, 2.390, 2.499, 0.064},
{1.460, 8.057, 1.336, 7.217, 7.914, 3.615, 9.981, 9.198, 5.292, 1.224},
{0.432, 8.645, 8.774, 0.249, 8.081, 7.461, 4.416, 0.652, 4.002, 4.644},
{0.679, 2.800, 5.523, 3.049, 2.968, 7.225, 6.730, 4.199, 9.614, 9.229},
{4.263, 1.074, 7.286, 5.599, 8.291, 5.200, 9.214, 8.272, 4.398, 4.506},
{9.496, 4.830, 3.150, 8.270, 5.079, 1.231, 5.731, 9.494, 1.883, 9.732},
{ 4.138, 2.562, 2.532, 9.661, 5.611, 5.500, 6.886, 2.341, 9.699, 6.500}
};
double cfox10[] = {
0.806,0.517,1.5,0.908,0.965,
0.669,0.524,0.902,0.531,0.876,
0.462,0.491,0.463,0.714,0.352,
0.869,0.813,0.811,0.828,0.964,
0.789,0.360,0.369,0.992,0.332,
0.817,0.632,0.883,0.608,0.326};
double mvfLangerman(int n, double *x)
{
int i;
double Sum, dist;
cfox10[2] = 1.5;
Sum = 0;
for ( i = 0; i < 5; i++ ){
dist = dist2(n, x, afox10[i]);
Sum -= cfox10[i] * (exp(-dist) / M_PI) * cos(M_PI * dist);
}
return Sum;
}
double mvfLennardJones(int n, double *x)
{
int i,j, i3, j3;
double z, dz, dzp, E;
int natoms;
natoms = n / 3;
for (i = 0; i < natoms; i++) {
i3 = i * 3;
for (j = i + 1; j < natoms; j++) {
dz = 0.0;
z = x[i3] - x[j3=j*3];
dz += z*z;
z = x[i3+1] - x[j3+1];
dz += z*z;
z = x[i3+2] - x[j3+2];
dz += z*z;
if (dz < 1.0e-6) { /* one pair of atoms too near ? */
return 1.0e30;
}
else {
dzp = 1.0 / (dz * dz * dz);
E += (dzp - 2.0) * dzp;
}
}
}
return E;
}
double mvfLeon(int n, double *x)
/*
See mvfRosenbrock for information.
*/
{
double a = x[1]-x[0]*x[0]*x[0];
double b = 1.0-x[0] ;
return 100.0*a*a + b*b ;
}
double mvfLevy(int n, double * x)
/*
- Global minimum
- for n=4, fmin = -21.502356 at (1,1,1,-9.752356 )
- for n=5,6,7, fmin = -11.504403 at (1,\dots,1,-4.754402 )
*/
{
int i;
double sum=0.0;
for (i=0 ; i<=n-2 ; i++)
sum+=pow(x[i]-1,2.0)*(1+pow(sin(3*M_PI*x[i+1]),2.0));
return pow(sin(3*M_PI*x[0]),2.0) + sum +
(x[n-1]-1)*(1+pow(sin(2*M_PI*x[n-1]),2.0));
}
double mvfMatyas(int n, double *x)
/*
Dimension: 2
Domain: |x[i] <= 10
Global minimum: 0 at x[i] = 0
*/
{
return 0.26 * (x[0]*x[0] + x[1]*x[1]) - 0.48 * x[0] * x[1];
}
double mvfMaxmod(int n, double *x)
/*
Domain: |x[i] <= 10
Global minimum: 0 at x[i] = 0
*/
{
int i;
double t = x[0];
double u;
for (i = 1; i < n; i++) {
u = fabs(x[i]);
if (u < t) t = u;
}
return u;
}
double mvfMcCormick(int n, double * x)
/*
Dimension: 2
Domain: -1.5 <= x[0] <= 4
-3 <= x[1] <= 4
Global minimum: -1.9133 at (-0.547, -1.54719).
*/
{
return sin(x[0]+x[1])
+pow(x[0]-x[1],2.0)-1.5*x[0]+2.5*x[1]+1.0 ;
}
double mvfMichalewitz(int n, double *x)
{
double u;
int i;
u=0;
for (i=0;i<n;i++) {
u = u + sin(x[i])
* pow(sin((i+1)*x[i]*x[i]/M_PI), 2.0 * 10.0);
}
return(-u);
}
double mvfMultimod(int n, double *x)
{
int i;
double t, s, p;
s = p = fabs(x[0]);
for (i = 1; i < n; i++) {
t = fabs(x[1]);
s += t;
p *= t;
}
return s + p;
}
/* Neumaier test functions */
double PERM_BETA = 50;
double PERM0_BETA = 10; /* Non-negative */
void setNeumaierPerm( double beta)
{
PERM_BETA = beta;
}
void setNeumaierPerm0(double beta)
{
PERM0_BETA = beta;
}
double mvfNeumaierPerm(int n, double x[])
/*
Global minimum x[i] = (i+1) with f* = 0.
Domain : x[i] in [-n,n] for i=0,...,n-1.
suggested beta values: (n,beta) = (4,50), (4,0.5), (10,10^9), (10,10^7)
*/
{
int i, k;
double sum = 0.0;
for (k = 0; k < n; k++) {
for (i = 0; i < n; i++) {
sum += (pow(i+1, k+1) + PERM_BETA) * sqr(pow(x[i]/(i+1), (k+1)) -1);
}
}
return sum;
}
double mvfNeumaierPerm0(int n, double x[])
/*
Domain: x_i in [-1,1] for i=0,...,n-1.
Global minimu f* = 0 at x[i] = 1/(i+1)
Suggested test values (n, beta) = (4,10) (10,100)
*/
{
int i, k;
double sum = 0.0;
for (k = 0; k < n; k++) {
for (i = 0; i < n; i++) {
sum += ((i+1)+PERM0_BETA) * sqr(pow(x[i+1], k)- pow(1/(i+1), k));
}
}
return sum;
}
double _POWERSUM_B[20] = {8,18,44,114};
double mvfNeumaierPowersum(int n, double x[])
/*
Global minimum f* = 0.
Domain: for default values _POWERSUM_B, x[i] in [0,4].
and x* = (1,2,2,3).
*/
{
int i, k;
double s1, sum;
sum = 0.0;
for (k = 0; k < n; k++) {
s1 = 0.0;
for (i = 0; i < n; i++) {
s1 += pow(x[i], k+1);
}
sum += sqr(s1 - _POWERSUM_B[k]);
}
return sum;
}
double mvfNeumaierTrid(int n, double x[])
/*
Suitable bounds for a bound constrained version would be [-n^2,n^2] for each component.
The solution is x_i=i(n+1-i) with f(x)=-n(n+4)(n-1)/6.
*/
{
int i;
double s1, s2;
s1 = 0.0;
for (i = 0; i < n; i++) {
s1 += sqr(x[i] - 1);
if (i) s2 += x[i] * x[i-1];
}
return s1 - s2;
}
double mvfOddsquare(int n, double *x)
/*
- Dimension: n = 5, 10
- Domain: | x[i] | <= 5 *pi
*/
{
double c = 0.2;
double a[] = {1, 1.3, 0.8, -0.4, -1.3, 1.6, -0.2, -0.6, 0.5, 1.4};
double d1 = dist2(n, x, a);
double d2 = sqr(distInf(n, x, a));
return -exp(-cos(d2) / (2*M_PI)) * cos( M_PI * d2) * (1.0 + c * (d1) / (d1 + 0.01));
}
double mvfPaviani(int n, double *x)
{
double a,b,sum=0.0, mul=1.0 ;
int i;
for (i=0 ; i<n ; i++) {
a=log(x[i]-2.0) ; b=log(10.0-x[i]) ;
sum+= a*a + b*b ;
mul*= x[i] ;
}
return sum - pow(mul,0.2) ;
}
double mvfPlateau(int n, double *x)
/*
Dimension 5
Bounds: |x[i]| < 5.12 for i = 0 to 4
Reference:
L. Ingber, "Simulated Annealing, Practice Vs. Theory",
Journal Math. Computing and Modelling, V18,N11,
D1993, 29-57
*/
{
int i;
double sum = 0.0;
for (i = 0; i < 5; i++) {
sum += floor(x[i]);
}
return 30.0 + sum;
}
double mvfPowell(int n, double *x)
{
int j;
double sum;
sum = 0.0;
for (j = 0; j < n/4; j++) {
sum += sqr(x[4*j-3] + 10 * x[4*j-2])
+ 5 * sqr(x[4*j-1] - x[4*j])
+ pow(x[4*j-2] - 2 * x[4*j-1], 4)
+ 10 * pow(x[4*j - 3] - x[4*j], 4);
}
return sum;
}
double mvfQuarticNoiseU(int n, double *x)
/*
-Domain: | x_i | < 1.28
-Global minimum is 0.0 at x_i = 0.00
*/
{
int i;
double sum, t;
sum = 0.0;
for (i = 0; i < n; i++) {
t = x[i];
t = t*t;
sum += t*t + rndUniform(0.0, 1.0);
}
return sum;
}
double mvfQuarticNoiseZ(int n, double *x)
/*
-Test optimization function. Range is | x_i | < 1.28
-Global minimum is 0.0 at x_i = 0.00
*/
{
int i;
double sum, t;
sum = 0.0;
for (i = 0; i < n; i++) {
t = x[i];
t = t*t;
sum += (i*t*t + rndZ());
}
return sum;
}
double mvfRana(int n, double *x)
/*
- Domain = | x_i | < 500
*/
{
int i;
double t1, t2, sum;
sum = 0.0;
for (i = 0; i < n-1; i++) {
t1 = sqrt(fabs(x[i+1] + x[i] + 1.0));
t2 = sqrt(fabs(x[i+1] - x[i] + 1.0));
sum += (x[i+1] + 1.0) * cos(t2) * sin(t1) + cos(t1) * sin(t2) * x[i];
}
return sum;
}
double mvfRastrigin(int n, double *x)
/*
- Domain = | x_i | < 5.12
- Global minimum is 0.0 at x_i = 0.00
*/
{
int i;
double sum, t;
if (n > 20) {
n = 20;
}
sum = 0.0;
for (i = 0; i < n; i++) {
t = x[i];
sum += t*t - cos(2.0 * M_PI * x[i]);
}
return sum + n*10;
}
double mvfRastrigin2(int n, double *x)
/*
- n = 2 always
- Domain = | x_i | < 5.12
- Global minimum is 0.0 at x_i = 0.00
*/
{
double t;
t = x[0];
return 2.0 + t*t - cos(18 * t) - cos(18 * x[1]);
}
double mvfRosenbrock(int n, double * x)
{
double a = x[1]-x[0]*x[0] ;
double b = 1.0-x[0] ;
return 100.0*a*a + b*b ;
}
double mvfSchaffer1(int n, double *x)
/*
- Dimension 2
- Global minimum 0 at (0)
- Domain: |x[i]| <= 100
*/
{
return 0.5 + (sqr(sin(sqrt(sqr(x[0]) + sqr(x[1])))) - 0.5) / sqr(1.0 + 0.001 * sqr(x[0]) + sqr(x[1]));
}
double mvfSchaffer2(int n, double *x)
/*
- Dimension 2
- Global minimum = 0 at (0).
*/
{
return pow(x[0]*x[0] + x[1]*x[1], 0.25)
* (50.0 * pow(x[0]*x[0] + x[1]*x[1], 0.1) + 1.0);
}
double mvfSchwefel1_2(int n, double *x)
/*
-n = 100 for testing
-Domain: | x_i | < 10.00
-Global minimum is 0.0 at x_i = 0.00
*/
{
int i, j;
double sum, sum1;
sum = 0.0;
for (i = 0; i < n; i++) {
sum1 = 0.0;
for (j = 0; j < i; j++) {
sum1 += x[i];
}
sum += sum1 * sum1;
}
return sum;
}
double mvfSchwefel2_21(int n, double *x)
/*
-n = 100 for testing
- Domain: | x_i | < 10.000
-Global minimum is 0.0 at x_i = 0.00
*/
{
int i;
double s,t;
s = fabs(x[0]);
for (i = 1; i < n; i++) {
t = fabs(x[i]);
if (t > s) {
s = t;
}
}
return s;
}
double mvfSchwefel2_22(int n, double *x)
/*
-n = 30 for testing
-Domain: | x_i | < 10.00
-Global minimum is 0.0 at x_i = 0.00
*/
{
int i;
double sum = 0.0;
double prod = 1.0;
for (i = 0; i < n; i++) {
sum += fabs(x[i]);
prod *= fabs(x[i]);
}
return sum + prod;
}
double mvfSchwefel2_26(int n, double *x)
/*
- Domain is | x_i | < 500
- Global minimum at fmin = -122569.5 at x_i = 420.9687
*/
{
int i;
double sum;
sum = 0.0;
for (i = 0; i < n; i++) {
sum += x[i] * sin(sqrt(x[i]));
}
return - sum;
}
double mvfShekel2(int n, double *x)
/*
- n = 2 always
- Domain: | x_i | <= 65.536
- Minimum 1 at x = (-32, -32)
*/
{
int j;
double s, t0, t1;
static double a[2][25] = {
{-32,-16,0,16,32,-32,-16,0,16,32,-32,-16,0,
16,32,-32,-16,0,16,32,-32,-16,0,16,32},
{-32,-32,-32,-32,-32,-16,-16,-16,-16,-16,0,
0,0,0,0,16,16,16,16,16,32,32,32,32,32}
};
s = 0.0;
for (j = 0; j < 25; j++) {
t0 = x[0] - a[0][j];
t1 = x[1] - a[1][j];
t0 = (t0 * t0 * t0);
t0 *= t0 * t0;
t1 = (t1 * t1 * t1);
t1 = t1 * t1;
s += 1.0/ ((double) j + t0 + t1);
}
return 1.0 / (1.0/500.0 + s);
}
static double afox4[10][4]={ { 4,4,4,4 } ,
{ 1,1,1,1 } ,
{ 8,8,8,8 } ,
{ 6,6,6,6 } ,
{ 3,7,3,7 } ,
{ 2,9,2,9 } ,
{ 5,5,3,3 } ,
{ 8,1,8,1 } ,
{ 6,2,6,2 } ,
{7,3.6,7,3.6} };
static double cfox4[10]= {.1,.2,.2,.4,.4,.6,.3,.7,.5,.5 };
double mvfShekelSub4(int m, double *x)
{
double R=0.0, S;
int i,j;
for(i=0;i<m;i++) {
S=0;
for (j=0;j<4;j++) S+=pow(x[j]-afox4[i][j],2);
R-=1/(S+cfox4[i]);
}
return R;
}
double mvfShekel4_5(int n, double * x)
/*
- Domain 0 <= x_j <= 10
- Global minimum fmin = -10.1532,
- at X = (4.00004, 4.00013, 4.00004, 4.00013)
- Number of local optimizers = 5
*/
{
return mvfShekelSub4(5, x);
}
double mvfShekel4_7(int n, double * x)
/*
- Domain 0 <= x_j <= 10
-Global minimum fmin = -10/4029 at
- (4.00057,4.00069,3.99949,3.99961) \
- Number of local minimum = 7
*/
{
return mvfShekelSub4(7, x);
}
double mvfShekel4_10(int n, double * x)
/*
- Domain 0 <= x_j <= 10
- Global minimum fmin = -10.5364 at
- (4.00075,4.00059,3.99966,3.99951)
- Number of local minimum = 10
*/
{
return mvfShekelSub4(10, x);
}
double mvfShekel10(int n, double *x)
{
int i;
double sum;
cfox10[2] = 0.10;
sum = 0.0;
n = 10; /* force checked */
for (i = 0; i < 30; i++) {
sum += 1.0/(dist2(n, x, afox10[i]) + cfox10[i]);
}
return sum;
}
double mvfShubert(int n, double * x)
/*
- Domain |x| <= 10.0
- Number of local minimum = 400
- Global minimum fmin = -24.062499 at the ff. points
- (-6.774576, -6.774576), ..., (5.791794, 5.791794)
*/
{
return -sin(2.0*x[0]+1.0)
-2.0*sin(3.0*x[0]+2.0)
-3.0*sin(4.0*x[0]+3.0)
-4.0*sin(5.0*x[0]+4.0)
-5.0*sin(6.0*x[0]+5.0)
-sin(2.0*x[1]+1.0)
-2.0*sin(3.0*x[1]+2.0)
-3.0*sin(4.0*x[1]+3.0)
-4.0*sin(5.0*x[1]+4.0)
-5.0*sin(6.0*x[1]+5.0);
}
double mvfShubert2(int n, double *x)
{
int i;
double s1, s2;
s1 = s2 = 0.0;
for (i = 0; i < 5; i++) {
s1 += (i+1) * cos((i+2)* x[0] + i+1);
s2 += (i+1) * cos((i+2)* x[1] + i+1);
}
return s1 + s2;
}
double mvfShubert3(int n, double *x)
/*
- Domain |x| <= 10.0
- Number of local minimum = 400
- Global minimum fmin = -24.062499 at the ff. points
- (-6.774576, -6.774576), ..., (5.791794, 5.791794)
*/
{
int i, j;
double s;
n = 2; // force checked
s = 0.0;
for (i = 0; i < n; i++) {
for (j = 0; j < 5; j++) {
s += (j+1)*sin((j+2)*x[i] + (j + 1));
}
}
return -s;
}
double mvfSphere(int n, double *x)
/*
- Domain: | x[i] | < 10
- Global minimum: 0 at x[i] = 0
*/
{
int i;
double sum;
sum = 0.0;
for (i = 0; i < n; i++) {
sum += x[i]*x[i];
}
return sum;
}
double mvfSphere2(int n, double *x)
/*
- Domain: | x[i] | < 10
- Global minimum: 0 at x[i] = 0
*/
{
int i, j;
double s1, s2;
s1 = 0.0;
s2 = 0.0;
for (i = 0; i < n; i++) {
s2 += x[i];
s1 += s2 * s2;
}
return s1;
}
double mvfStep(int n, double *x)
/*
- n In: number of variables, 30
- x In: point of evaluation
- Domain: | x_i | < 100.0
- Global minimum is 0.0 at x_i = 0.00
*/
{
int i;
double y;
double sum;
sum = 0.0;
for (i = 0; i < n; i++) {
y = floor(x[i] + 0.5);
sum += y*y;
}
return sum;
}
double mvfStretchedV(int n, double *x)
/*
- Dimension:
- Domain: | x_i | <= 10.0
*/
{
int i = 0;
double sum;
double t;
sum = 0.0;
for (i = 0; i < n - 1; i++) {
t = x[i+1]*x[i+1] + x[i]*x[i];
sum += pow(t, 0.25) * pow(sin(50.0 * pow(t, 0.1) + 1.0), 2);
}
return sum;
}
double mvfSumSquares(int n, double *x)
/*
Dimension: n
Domain: | x[i] | <= 10
Global minimum: 0 at x[i] = 0
*/
{
int i;
double sum = 0;
for (i = 0; i < n; i++) {
sum += i * x[i]*x[i];
}
return sum;
}
double mvfTrecanni(int n, double *x)
/*
Dimension: 2
Domain: | x[i] | <= 5
Global minimum: 0 at (0,0) and (-2,0)
*/
{
return ((x[0] + 4) * x[0] + 4) * x[0] * x[0];
}
double mvfTrefethen4(int n, double *x)
/*
- Dimension: 2
- Domain: -6.5 < x[0] < 6.5,
- -4.5 < x[0] < 4.5
-
- Global minimum: -3.30686865 at
- (-0.0244031, 0.2106124)
*/
{
return exp(sin(50.0 * x[0])) +
sin(60.0 * exp(x[1])) +
sin(70.0 * sin(x[0])) +
sin(sin(80*x[1])) -
sin(10.0 * (x[0] + x[1])) +
1.0/4.0 * (x[0]*x[0] + x[1] * x[1]);
}
double mvfXor(int n, double* x)
/*
- Dimension 9
-
*/
{
return 1/ sqr(1 + exp(- x[6] / (1+exp(-x[0]-x[1]-x[4])) - x[7] / (1+exp(-x[2]-x[3]-x[5])) - x[8]))
+ 1/ sqr(1 + exp(-x[6]/(1 + exp(-x[4])) - x[7]/(1+exp(-x[5])) - x[8]))
+ 1/ sqr(1 - 1/(1+exp(x[6]/(1 + exp(-x[0]-x[4]))) - x[7]/(1 + exp(-x[3] - x[5])) - x[8]))
+ 1/ sqr(1 - 1/(1+exp(x[6]/(1 + exp(-x[1]-x[4]))) - x[7]/(1 + exp(-x[3] - x[5])) - x[8]));
}
double mvfWatson(int n, double *x)
{
int i, j;
double a;
double s;
double t, s1, s2;
s += x[0] * x[0];
for (i = 0; i < 30; i++) {
a = i / 29.0;
s1 = 0.0;
s2 = 0.0;
for (j = 0; j < 5; j++) {
s1 += (j+1)* pow(a, j) * x[j+1];
}
for (j = 0; j < 6; j ++) {
s2 += pow(a, j) * x[j];
}
t = (s1 - s2 * s2 - 1);
s += t * t;
};
return s;
}
double mvfZettl(int n, double *x)
{
double t = (x[0] * x[0] + x[1] * x[1] - 2 * x[0]);
return t * t + 0.25 * x[0];
}
double _Zh1(double *x)
{
return 9.0 - x[0] - x[1];
}
double _Zh2(double *x)
{
return sqr(x[0] - 3.0) + sqr(x[1] - 2.0) - 16.0;
}
double _Zh3(double *x)
{
return x[0] * x[1] - 14.0;
}
double _Zp(double x)
{
return 100.0 * (1 + x);
}
double mvfZimmerman(int n, double *x)
/*
- Dimension: 2
- Global minimum at
- Domain: 0 <= x[i] <= 100.0
*/
{
double px[] = {
_Zh1(x),
_Zp(_Zh2(x)) * sgn(_Zh2(x)),
_Zp(_Zh3(x)) * sgn(_Zh3(x)),
_Zp(-x[0]) * sgn(x[0]),
_Zp(-x[1]) * sgn(x[1])
};
return dmax(5, px);
}
