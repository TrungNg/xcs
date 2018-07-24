#ifndef __CRAND_H__
#define __CRAND_H__

void setSeed(long newSeed);
long getSeed();
double drand();
int irand(int n); // returns a number from 0 to n-1 inclusive

#endif