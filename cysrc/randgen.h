
const long _M = 2147483647; //Constant for the random number generator (modulus of PMMLCG = 2^31 -1).
const long _A = 16807; //Constant for the random number generator (default = 16807).

void setSeed(long newSeed);
long getSeed();
double drand();

int irand(int n); // returns a number from 0 to n-1 inclusive
