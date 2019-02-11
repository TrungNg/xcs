"""
Name:        xcs_run.py
Authors:     Bao Trung
Contact:     baotrung@ecs.vuw.ac.nz
Created:     July, 2017
Description:
---------------------------------------------------------------------------------------------------------------------------------------------------------
XCS: Michigan-style Learning Classifier System - A LCS for Reinforcement Learning.  This XCS follows the version descibed in "An Algorithmic Description of XCS" published by Martin Butz and Stewart Wilson (2002).

---------------------------------------------------------------------------------------------------------------------------------------------------------
"""

#Import Required Modules------------------------------------
#import crandom as random
import random
import datetime
from xcs_timer import Timer
from xcs_config_parser import ConfigParser
from xcs_offline_environment import Offline_Environment
from xcs_algorithm import XCS
from xcs_constants import cons
from xcs_online_environment import Online_Environment
from sys import argv
from statistics import mean
#-----------------------------------------------------------
#Function to parse arguments--------------------------------
def getOptions( argv ):
    """ Get arguments by command line and assign them to Constant object. """
    opts = {}  # Empty dictionary to store key-value pairs.
    while argv:  # While there are arguments left to parse...
        if argv[0][0] == '-':  # Found a "-name value" pair.
            opts[argv[0][1:]] = argv[1]  # Add key and value to the dictionary.
        argv = argv[1:]  # Reduce the argument list by copying it starting from index 1.

    # Assign options to paramters
    if 'seed' in opts:
        cons.random_seed = int( opts['seed'] )
    if 'N' in opts:
        cons.N = int( opts['N'] )
    if 'problemType' in opts:
        cons.problem_name = str( opts['problemType'] )
    if 'problemSize' in opts:
        sizes = opts['problemSize'].split('.')
        cons.problem_sizes = [ 0 ] * 3
        for i in range( len( sizes ) ):
            cons.problem_sizes[ i ] = int( sizes[ i ] )
    if 'crossover' in opts:
        cons.chi = float( opts['crossover'] )
    if 'mutation' in opts:
        cons.mu = float( opts['mutation'] )
    if 'beta' in opts:
        cons.beta = float( opts['beta'] )
    if 'ActionsetSub' in opts:
        cons.do_actionset_subsumption = bool(int( opts['ActionsetSub'] ))
    if 'GASub' in opts:
        cons.do_ga_subsumption = bool(int( opts['GASub'] ))

helpstr = """Failed attempt to run e-LCS.  Please ensure that a configuration file giving all run parameters has been specified."""

#Specify the name and file path for the configuration file.
config_file = "XCS_Configuration_File.txt"

#Obtain all run parameters from the configuration file and store them in the 'Constants' module.
ConfigParser( config_file )

#Initialize the 'Timer' module which tracks the run time of algorithm and it's different components.
timer = Timer()
cons.referenceTimer( timer )
getOptions( argv )

#Set random seed if specified.-----------------------------------------------
if cons.use_seed:
    random.seed(cons.random_seed)
else:
    random.seed(datetime.now())

#Initialize the 'Environment' module which manages the data presented to the algorithm.  While e-LCS learns iteratively (one inistance at a time
if cons.online_data_generator:
    env = Online_Environment( cons.problem_name, cons.problem_sizes )
else:
    env = Offline_Environment()
cons.referenceEnv( env ) #Passes the environment to 'Constants' (cons) so that it can be easily accessed from anywhere within the code.

#Run the e-LCS algorithm.
if not cons.online_data_generator and cons.kfold_cv == True:
    total_instances = env.format_data.numb_train_instances
    env.format_data.splitFolds( cons.kfold )
    accurate_numbs = [0.0] * cons.kfold
    for i in range( cons.kfold ):
        env.format_data.selectTrainTestSets(i)
        cons.parseIterations()  # Identify the maximum number of learning iterations as well as evaluation checkpoints.
        accuracy = XCS(str(i)).run()[0]
        accurate_numbs[i] = accuracy * env.format_data.numb_test_instances
    print("AVERAGE ACCURACY After " + str( cons.kfold ) + "-FOLD CROSS VALIDATION is " + str( sum(accurate_numbs) / total_instances ))
else:
    if not cons.online_data_generator:
        env.format_data.splitData2()
    cons.parseIterations()  # Identify the maximum number of learning iterations as well as evaluation checkpoints.
    XCS().run()