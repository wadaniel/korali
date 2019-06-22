/* 
	Credits:
	https://www.sfu.ca/~ssurjano/other.html
*/

#ifndef _MVF_H
#define _MVF_H

double sqr (double x );
double dmax (int n ,double * x );
double sgn (double x );
double dist2 (int n ,double * x ,double * y );
double distInf (int n ,double * x ,double * y );
double prod (int n ,double * x );

double mvfAckley (int n ,double * x );
double mvfBeale (int n ,double * x );
double mvfBohachevsky1 (int n ,double * x );
double mvfBohachevsky2 (int n ,double * x );
double mvfBooth (int n ,double * x );
double mvfBoxBetts (int n ,double * x );
double mvfBranin (int n ,double * x );
double mvfBranin2 (int n ,double * x );
double mvfCamel3 (int n ,double * x );
double mvfCamel6 (int n ,double * x );
double mvfChichinadze (int n ,double * x );
double mvfCola (int n ,double * x );
double mvfColville (int n ,double * x );
double mvfCorana (int n ,double * x );
double mvfEasom (int n ,double * x );
double mvfEggholder (int n ,double * x );
double mvfExp2 (int n ,double * x );
double mvfFraudensteinRoth (int n ,double * x );
double mvfGear (int n ,double * x);
double mvfGeneralizedRosenbrock (int n ,double * x );
double mvfGoldsteinPrice (int n ,double * x );
double mvfGriewank (int n ,double * x );
double mvfHansen (int n ,double * x );
double mvfHartman3 (int n ,double * x );
double mvfHartman6 (int n ,double * x );
double mvfHimmelblau (int n ,double * x );
double mvfHolzman1 (int n ,double * x );
double mvfHolzman2 (int n ,double * x );
double mvfHosaki (int n ,double * x );
double mvfHyperellipsoid (int n ,double * x );
double mvfKatsuuras (int n ,double * x );
double mvfKowalik (int n ,double * x );
double mvfLangerman (int n ,double * x );
double mvfLennardJones (int n ,double * x );
double mvfLeon(int n, double *x);
double mvfLevy (int n ,double * x );
double mvfMatyas (int n ,double * x );
double mvfMaxmod(int n, double *x);
double mvfMcCormick (int n ,double * x );
double mvfMichalewitz (int n ,double * x );
double mvfMultimod(int n, double *x);
void setNeumaierPerm (double beta );
void setNeumaierPerm0 (double beta );
double mvfNeumaierPerm (int n, double *x );
double mvfNeumaierPerm0 (int n, double *x );
double mvfNeumaierPowersum (int n, double *x );
double mvfNeumaierTrid (int n, double *x );
double mvfOddsquare (int n ,double * x );
double mvfPaviani (int n ,double * x );
double mvfPlateau (int n ,double * x );
double mvfPowell (int n ,double * x );
double mvfQuarticNoiseU (int n ,double * x );
double mvfQuarticNoiseZ (int n ,double * x );
double mvfRana (int n ,double * x );
double mvfRastrigin (int n ,double * x );
double mvfRastrigin2 (int n ,double * x );
double mvfRosenbrock (int n ,double * x );
double mvfSchaffer1 (int n ,double * x );
double mvfSchaffer2 (int n ,double * x );
double mvfSchwefel1_2 (int n ,double * x );
double mvfSchwefel2_21 (int n ,double * x );
double mvfSchwefel2_22 (int n ,double * x );
double mvfSchwefel2_26 (int n ,double * x );
double mvfShekel2 (int n ,double * x );
double mvfShekelSub4 (int m ,double * x );
double mvfShekel4_5 (int n ,double * x );
double mvfShekel4_7 (int n ,double * x );
double mvfShekel4_10 (int n ,double * x );
double mvfShekel10 (int n ,double * x );
double mvfShubert (int n ,double * x );
double mvfShubert2 (int n ,double * x );
double mvfShubert3 (int n ,double * x );
double mvfSphere (int n ,double * x );
double mvfSphere2(int n, double *x );
double mvfStep (int n ,double * x );
double mvfStretchedV (int n ,double * x );
double mvfSumSquares (int n ,double * x );
double mvfTrecanni (int n ,double * x );
double mvfTrefethen4 (int n ,double * x );
double mvfWatson(int n, double *x);
double mvfXor (int n, double *x);
double mvfZettl(int n, double *x);
double _Zh1 (double * x );
double _Zh2 (double * x );
double _Zh3 (double * x );
double _Zp (double x );
double mvfZimmerman (int n ,double * x );
#endif
