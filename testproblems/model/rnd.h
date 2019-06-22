/*
	Credits:
	https://www.sfu.ca/~ssurjano/other.html
*/

#ifndef _RND_H
#define _RND_H
void rndSeed(unsigned long seed);
double rnd(void);
int rnd0(int N);
int rndInt(int a, int b);
double rndUniform(double a, double b);
double rndZ(void);
double rndNormal(double mu, double std);
#endif
