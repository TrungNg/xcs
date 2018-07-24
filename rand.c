#include "rand.h"

long seed = 143907; // a number between 1 and _M-1 */

const long _M = 2147483647; //Constant for the random number generator (modulus of PMMLCG = 2^31 -1).
const long _A = 16807; //Constant for the random number generator (default = 16807).
const long _Q = 127773; // constant for the random number generator (=_M/_A).
const long _R = 2836; // constant for the random number generator (=_M mod _A).

void setSeed(long s){ // sets a random seed in order to randomize the pseudo random generator.
	seed=s;
}

long getSeed(){
	return seed;
}

double drand(){ // returns a floating-point random number generated according to uniform distribution from [0,1]
	long hi   = seed / _Q;
	long lo   = seed % _Q;
	long test = _A*lo - _R*hi;
	if (test>0){
	    seed = test;
	}
	else{
	    seed = test+_M;
	}
	return (double)(seed)/_M;
}

int irand(int n){ // returns a random number generated according to uniform distribution from [0,n-1]
	int num = (int)(drand()*(float)n);
	while(num == n){
		num = (int)(drand()*(float)n);
	}
	return num;
}