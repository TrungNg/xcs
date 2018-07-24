"""
Name:        crandom.pxd
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
    choices_cp = choices
    if k > len(choices):
        return -1
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