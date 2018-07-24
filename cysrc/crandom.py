"""
Name:        crandom.py
Authors:     Bao Trung
Contact:     baotrung@ecs.vuw.ac.nz
Created:     July, 2018
Description:
---------------------------------------------------------------------------------------------------------------------------------------------------------

---------------------------------------------------------------------------------------------------------------------------------------------------------
"""
#Import Required Modules---------------------

#--------------------------------------------


cdef extern from "randgen.h":
    cdef void setSeed(long newSeed);
    cdef long getSeed();
    cdef double drand();

    cdef int irand(int n);