"""
Name:        xcs_run.py
Authors:     Bao Trung - Written at Victoria University of Wellington, New Zealand
Contact:     baotrung@ecs.vuw.ac.nz
Created:     July, 2017
Description:
---------------------------------------------------------------------------------------------------------------------------------------------------------
XCS: Michigan-style Learning Classifier System - A LCS for Reinforcement Learning.  This XCS follows the version descibed in "An Algorithmic Description of XCS" published by Martin Butz and Stewart Wilson (2002).

---------------------------------------------------------------------------------------------------------------------------------------------------------
"""

#Import Required Modules------------------------------------
from xcs_timer import Timer
from xcs_config_parser import ConfigParser
from xcs_environment_interactor import EnvironmentInteractor
from xcs_algorithm import XCS
from xcs_constants import cons
#-----------------------------------------------------------

helpstr = """Failed attempt to run XCS.  Please ensure that a configuration file giving all run parameters has been specified."""

#Specify the name and file path for the configuration file.
configurationFile = "XCS_Configuration_File.txt"

#Obtain all run parameters from the configuration file and store them in the 'Constants' module.
ConfigParser(configurationFile)

#Initialize the 'Timer' module which tracks the run time of algorithm and it's different components.
timer = Timer()
cons.referenceTimer(timer)

#Initialize the 'Environment' module which manages the data presented to the algorithm.  While e-LCS learns iteratively (one inistance at a time
env = EnvironmentInteractor()
cons.referenceEnv(env) #Passes the environment to 'Constants' (cons) so that it can be easily accessed from anywhere within the code.
cons.parseIterations() #Identify the maximum number of learning iterations as well as evaluation checkpoints.

#Run the e-LCS algorithm.
XCS()
