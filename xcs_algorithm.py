"""
Name:        xcs_algorithm.py
Authors:     Bao Trung
Contact:     baotrung@ecs.vuw.ac.nz
Created:     July, 2017
Description:
---------------------------------------------------------------------------------------------------------------------------------------------------------
XCS: Michigan-style Learning Classifier System - A LCS for Reinforcement Learning.  This XCS follows the version descibed in "An Algorithmic Description of XCS" published by Martin Butz and Stewart Wilson (2002).

---------------------------------------------------------------------------------------------------------------------------------------------------------
"""

#Import Required Modules-------------------------------
from xcs_constants import *
from xcs_classifierset import ClassifierSet
from xcs_prediction import *
from xcs_environment_interactor import *
from xcs_logger import *
import copy
import random
import math
#------------------------------------------------------

class XCS:
    def __init__(self):
        """ Initializes the XCS algorithm """
        print("XCS: Initializing Algorithm...")
        #Global Parameters-------------------------------------------------------------------------------------
        self.population = None          # The rule population (the 'solution/model' evolved by XCS)
        self.previous_state = None              # states received from environment from last time step

        #-------------------------------------------------------
        # NORMAL XCS - Run XCS from scratch on given data
        #-------------------------------------------------------
        self.logger = Logger( cons.out_filename )
        # Instantiate Population---------
        self.population = ClassifierSet()
        self.explore_iter = 0
        self.trial_count = 0
        self.steps_to_goal = [ 0 for _ in range( cons.env.number_of_states ) ]

        #Run the XCS algorithm----------------------------------
        self.run_XCS()


    def run_XCS(self):
        """ Runs the initialized XCS algorithm. """

        print("Learning Checkpoints: " +str( cons.learning_checkpoints ) )
        print("Maximum Iterations: " +str( cons.max_learning_iterations ) )
        print("Beginning XCS learning iterations.")
        print("------------------------------------------------------------------------------------------------------------------------------------------------------")

        #-------------------------------------------------------
        # MAJOR LEARNING LOOP
        #-------------------------------------------------------
        while self.explore_iter < cons.max_learning_iterations:
            #-------------------------------------------------------
            # GET NEW INSTANCE AND RUN A LEARNING ITERATION
            #-------------------------------------------------------
            self.runIteration( self.explore_iter )

            #-------------------------------------------------------
            # CHECKPOINT - COMPLETE EVALUTATION OF POPULATION - strategy different for discrete vs continuous phenotypes
            #-------------------------------------------------------
            if ( self.explore_iter + 1 ) in cons.learning_checkpoints:
                cons.timer.startTimeEvaluation()
                print("------------------------------------------------------------------------------------------------------------------------------------------------------")
                print("Running Population Evaluation after " + str( self.explore_iter + 1 )+ " iterations.")

                self.population.runPopAveEval()
                self.population.runAttGeneralitySum( True )
                cons.env.startEvaluationMode()                      #Preserves learning position in Environment
                train_evaluation = self.doPopEvaluation()
                cons.env.stopEvaluationMode()                       #Returns to learning position in Environment
                cons.timer.stopTimeEvaluation()
                cons.timer.returnGlobalTimer()

                #Write output files----------------------------------------------------------------------------------------------------------
                self.logger.writePopStats( train_evaluation, self.explore_iter + 1, self.population, self.steps_to_goal )
                self.logger.writePop( self.explore_iter + 1, self.population )
                #----------------------------------------------------------------------------------------------------------------------------

                print("Continue Learning...")
                print("------------------------------------------------------------------------------------------------------------------------------------------------------")

            self.explore_iter += 1       # Increment current learning iteration

        # Once XCS has reached the last learning iteration, close the logging file
        self.logger.closeLogFile()
        print("XCS Run Complete")


    def runIteration(self, explore_iter):
        """ Run a single XCS learning iteration. """
        state = cons.env.getCurrentCondition()
        #-----------------------------------------------------------------------------------------------------------------------------------------
        # FORM A MATCH SET - includes covering
        #-----------------------------------------------------------------------------------------------------------------------------------------
        self.population.makeMatchSet( state, explore_iter )
        if self.population.match_set == []:
            return
        #-----------------------------------------------------------------------------------------------------------------------------------------
        # MAKE A PREDICTION - utilized here for tracking estimated learning progress.  Typically used in the explore phase of many LCS algorithms.
        #-----------------------------------------------------------------------------------------------------------------------------------------
        cons.timer.startTimeEvaluation()
        prediction = Prediction( self.population )
        selected_action = prediction.getDecision()
        action_prediction = prediction.getActionPrediction()
        cons.timer.stopTimeEvaluation()
        #-----------------------------------------------------------------------------------------------------------------------------------------
        # FORM AN ACTION SET, EXECUTE THE ACTION AND GET RESPONSE FROM ENVIRONMENT
        #-----------------------------------------------------------------------------------------------------------------------------------------
        self.population.makeActionSet( selected_action )
        response = cons.env.affectEnvironment( selected_action )
        self.steps_to_goal[ self.trial_count%cons.tracking_frequency ] += 1
        #-----------------------------------------------------------------------------------------------------------------------------------------
        # CHECK PREVIOUS ACTION SET AND CURRENT ACTION SET
        #-----------------------------------------------------------------------------------------------------------------------------------------
        if self.population.prev_action_set != []:
            self.population.updateSets( self.prev_response[ 0 ], action_prediction, self.population.prev_action_set )
            #-----------------------------------------------------------------------------------------------------------------------------------------
            # RUN THE GENETIC ALGORITHM - Discover new offspring rules from a selected pair of parents
            #-----------------------------------------------------------------------------------------------------------------------------------------
            self.population.runGA( explore_iter, self.population.prev_action_set, self.previous_state )
            if cons.do_subsumption:
                cons.timer.startTimeSubsumption()
                self.population.doActionSetSubsumption( self.population.prev_action_set )
                cons.timer.stopTimeSubsumption()
        #-----------------------------------------------------------------------------------------------------------------------------------------
        # STOP TRIAL, UPDATE CURRENT ACTION SET AND MOVE TO NEW TRIAL/GAME
        #-----------------------------------------------------------------------------------------------------------------------------------------
        if response[1] == "EOP" or self.steps_to_goal[ self.trial_count%cons.tracking_frequency ] >= cons.max_learning_steps:
            self.trial_count += 1
            self.population.updateSets( response[0], 0, self.population.action_set )
            self.population.runGA( explore_iter, self.population.action_set, state )
            cons.env.newGame()  # Start new trial
            if self.steps_to_goal[ self.trial_count%cons.tracking_frequency ] >= cons.max_learning_steps:
                self.steps_to_goal[ self.trial_count%cons.tracking_frequency ] = 0
            #-------------------------------------------------------------------------------------------------------------------------------
            # EVALUATIONS OF ALGORITHM
            #-------------------------------------------------------------------------------------------------------------------------------
            cons.timer.startTimeEvaluation()
            if ( self.trial_count%cons.tracking_frequency ) == 0:
                #-------------------------------------------------------------
                # TRACK LEARNING SPEED - including exploration, random steps
                # Report learning progress to standard out and tracking file.
                self.population.runPopAveEval()
                not_solved = sum( x == 0 for x in self.steps_to_goal )
                average_steps_to_goal = sum( self.steps_to_goal ) / float( cons.env.number_of_states - not_solved )
                self.logger.track( self.population, average_steps_to_goal, self.trial_count, cons.tracking_frequency, float( cons.env.number_of_states - not_solved)/ cons.env.number_of_states )
                # reset steps counter
                self.steps_to_goal = [ 0 for _ in range( cons.env.number_of_states ) ]
            cons.timer.stopTimeEvaluation()
            if cons.do_subsumption:
                #-------------------------------------------------------------
                # ACTION SET SUBSUMPTION - if enabled
                #-------------------------------------------------------------
                cons.timer.startTimeSubsumption()
                self.population.doActionSetSubsumption( self.population.action_set )
                cons.timer.stopTimeSubsumption()
        #------------ Store response and state, clean sets -----------#
        self.prev_response = response
        self.previous_state = state  # store current state for next time step
        self.population.clearSets( response[ 1 ] == "EOP" ) #Clears the match and action sets for the next learning iteration


    def doPopEvaluation(self):
        """ Performs a complete evaluation of the current rule population.  The population is unchanged throughout this evaluation. Works on both training and testing data. """
        noMatch = 0                     # How often does the population fail to have a classifier that matches an instance in the data.
        tie = 0                         # How often can the algorithm not make a decision between classes due to a tie.
        is_training_mode = False
        cons.env.resetInitialStateRef() # Go to the first instance in dataset
        #----------------------------------------------
        instances = cons.env.number_of_states
        #----------------------------------------------------------------------------------------------
        steps_to_goal = [ 0 for inst in range( instances ) ]
        not_solved = 0
        #----------------------------------------------
        for inst in range( instances ):
            loop_more = True
            while loop_more:
                #-----------------------------------------------------------------------------
                steps_to_goal[ inst ] += 1
                state = cons.env.getCurrentCondition()
                self.population.makeEvalMatchSet( state )
                prediction = Prediction( self.population, True )
                action_selection = prediction.getDecision()
                #-----------------------------------------------------------------------------

                if action_selection == None:
                    noMatch += 1
                elif action_selection == 'Tie':
                    tie += 1
                else: #Instances which failed to be covered are excluded from the accuracy calculation
                    _, reponse = cons.env.affectEnvironment( action_selection )

                if reponse == 'EOP' or steps_to_goal[inst] >= cons.max_learning_steps:
                    loop_more = False
                    cons.env.newGame( is_training_mode )
                    # if number of steps agent has made reaches max learning steps, the problem is considered not solved
                    if steps_to_goal[ inst ] >= cons.max_learning_steps:
                        steps_to_goal[ inst ] = 0
                        not_solved += 1
                self.population.clearSets( reponse == 'EOP' )
        #----------------------------------------------------------------------------------------------
        #Calculate Learn Speed--------------------------------------------
        if not_solved < instances:
            average_steps_to_goal = float( sum( steps_to_goal ) / ( instances - not_solved ) )
        else:
            average_steps_to_goal = cons.max_learning_steps

        print("-----------------------------------------------")
        print("Evaluation Results:-------------")
        print("Average Steps to Goal: " + str( average_steps_to_goal ) )
        print("Instance Coverage: "+ str( float( instances - not_solved ) / instances ) )
        return [ average_steps_to_goal, float( instances - not_solved ) / instances ]

