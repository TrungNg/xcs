"""
Name:        crandom.py
Authors:     Bao Trung
Contact:     baotrung@ecs.vuw.ac.nz
Created:     July, 2018
Description:
---------------------------------------------------------------------------------------------------------------------------------------------------------
"""

#Import Required Modules------------------------------------
#-----------------------------------------------------------

cdef extern from "rand.h":
    void setSeed(long newSeed)
    long getSeed()
    double drand()
    int irand(int n)

def seed(new_seed):
    """ set random seed. """
    setSeed(new_seed)

def choice( choices ):
    """ select an item from choices. """
    return choices[ irand(len(choices)) ]

def sample( choices, k ):
    """ select k samples without replacing from choices. """
    selected = []
    if k > len(choices):
        raise ValueError("k " + str(k) + " is larger than the number of choices " + str(len(choices)))
    choices_cp = choices[:]
    for i in range(k):
        selected_item = choice(choices_cp)
        selected.append( selected_item )
        choices_cp.remove( selected_item )
    return selected

def random():
    """ random(0,1). """
    return drand()

cpdef int randint(int a, int b):
    """ select an integer randomly. """
    return a + irand(b-a+1)

########################################################################################################################
# seed = 143907 # a number between 1 and _M-1 */
#
# _M = 2147483647 #Constant for the random number generator (modulus of PMMLCG = 2^31 -1).
# _A = 16807 #Constant for the random number generator (default = 16807).
# _Q = 127773 # constant for the random number generator (=_M/_A).
# _R = 2836 # constant for the random number generator (=_M mod _A).
#
# def seed( new_seed ):
#     """ """
#     global seed
#     seed = new_seed
#
# def random():
#     global seed
#     hi   = int( seed / _Q )
#     lo   = int( seed % _Q )
#     test = _A*lo - _R*hi
#     if (test>0):
#         seed = test
#     else:
#         seed = test+_M
#     return float((seed)/_M)
#
# def irand(n):
#     num = int(random()*n)
#     while(num == n):
#         num = int(random()*n)
#         return num
#     return num
#
# def sample( choices, k ):
#     """ select k samples without replacing from choices. """
#     selected = []
#     if k > len(choices):
#         raise ValueError("k " + str(k) + " is larger than the number of choices " + str(len(choices)))
#     choices_cp = choices[:]
#     for i in range(k):
#         selected_item = choice(choices_cp)
#         selected.append( selected_item )
#         choices_cp.remove( selected_item )
#     return selected
#
# def choice( choices ):
#     """ select an item from choices. """
#     return choices[ irand(len(choices)) ]
#
# def randint(a, b):
#     """ select an integer randomly. """
#     return a + irand(b-a+1)