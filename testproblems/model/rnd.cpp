/*
	Credits:
	https://www.sfu.ca/~ssurjano/other.html
*/

/*
File rnd.c
Remarks.
This file needs updating with the latest versions of
the Mersenne Twister.
*/
#include <math.h>
#include <signal.h>
#include <string.h>
#include <time.h>
#include <sys/time.h>
#include <unistd.h>
#include <stdlib.h>
#include <stdio.h>
#include "rnd.h"
/* ********** Start of Mersenne Twister **********************
This section contains code for the Mersenne Twister
random number generator. Please refer to the web site
http://www.math.keio.ac.jp/~matumoto/emt.html
for the latest versions and more information. The authors of
Mersenne Twister have permitted the use of MT for commercial
purposes.
http://www.math.keio.ac.jp/~matumoto/eartistic.html
*/
/* Period parameters */
#define _N 624
#define _M 397
#define MATRIX_A 0x9908b0df /* constant vector a */
#define UPPER_MASK 0x80000000 /* most significant w-r bits */
#define LOWER_MASK 0x7fffffff /* least significant r bits */
/* Tempering parameters */
#define TEMPERING_MASK_B 0x9d2c5680
#define TEMPERING_MASK_C 0xefc60000
#define TEMPERING_SHIFT_U(y) (y >> 11)
#define TEMPERING_SHIFT_S(y) (y << 7)
#define TEMPERING_SHIFT_T(y) (y << 15)
#define TEMPERING_SHIFT_L(y) (y >> 18)
static unsigned long mt[_N]; /* the array for the state vector */
static int mti=_N+1; /* mti==N+1 means mt[N] is not initialized */
/* Initializing the array with a seed */
void rndSeed(unsigned long seed)
{
  int i;
  for (i=0; i<_N; i++)
  {
    mt[i] = seed & 0xffff0000;
    seed = 69069 * seed + 1;
    mt[i] |= (seed & 0xffff0000) >> 16;
    seed = 69069 * seed + 1;
  }
  mti = _N;
}
void lsgenrand(unsigned long seed_array[])
/* the length of seed_array[] must be at least N */
{
  int i;
  for (i=0; i<_N; i++)
    mt[i] = seed_array[i];
  mti=_N;
}
double rnd(void)
{
  unsigned long y;
  static unsigned long mag01[2]=
  {
    0x0, MATRIX_A
  };
  /* mag01[x] = x * MATRIX_A for x=0,1 */
  if (mti >= _N)   /* generate N words at one time */
  {
    int kk;
    if (mti == _N+1) /* if sgenrand() has not been called, */
      rndSeed(4357); /* a default initial seed is used */
    for (kk=0; kk<_N-_M; kk++)
    {
      y = (mt[kk]&UPPER_MASK)|(mt[kk+1]&LOWER_MASK);
      mt[kk] = mt[kk+_M] ^ (y >> 1) ^ mag01[y & 0x1];
    }
    for (; kk<_N-1; kk++)
    {
      y = (mt[kk]&UPPER_MASK)|(mt[kk+1]&LOWER_MASK);
      mt[kk] = mt[kk+(_M-_N)] ^ (y >> 1) ^ mag01[y & 0x1];
    }
    y = (mt[_N-1]&UPPER_MASK)|(mt[0]&LOWER_MASK);
    mt[_N-1] = mt[_M-1] ^ (y >> 1) ^ mag01[y & 0x1];
    mti = 0;
  }
  y = mt[mti++];
  y ^= TEMPERING_SHIFT_U(y);
  y ^= TEMPERING_SHIFT_S(y) & TEMPERING_MASK_B;
  y ^= TEMPERING_SHIFT_T(y) & TEMPERING_MASK_C;
  y ^= TEMPERING_SHIFT_L(y);
  return ( ((double)y + 1.0) * 2.3283064359965952e-10 );
  /* reals: (0,1)-interval */
  /* return y; */ /* for integer generation */
}
/* ****** End of Mersenne Twister */
int rnd0(int N)
/* returns a random integer from 0 to N-1 */
{
  return (int) (rnd() * N);
}
int rndInt(int a, int b)
{
  return (int) (rnd() * (b - a + 1)) + a;
}
double rndUniform(double a, double b)
{
  return a + rnd() * (b - a);
}
double rndZ(void)
{
  /* Unoptimized version */
  return sqrt(-2.0*log(rnd())) * sin(2*M_PI*rnd());
}
double rndNormal(double mu, double std)
{
  return rndZ() * std + mu;
}
