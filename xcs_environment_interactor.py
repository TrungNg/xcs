"""
Name:        xcs_environment_interactor.py
Authors:     Bao Trung
Contact:     baotrung@ecs.vuw.ac.nz
Created:     July, 2017
Description:
---------------------------------------------------------------------------------------------------------------------------------------------------------
XCS: Michigan-style Learning Classifier System - A LCS for Reinforcement Learning.  This XCS follows the version descibed in "An Algorithmic Description of XCS" published by Martin Butz and Stewart Wilson (2002).

---------------------------------------------------------------------------------------------------------------------------------------------------------
"""

#Import Required Modules--------------------------------------
from environment_controller import *
from xcs_constants import *
import math
#-------------------------------------------------------------

class EnvironmentInteractor:
    def __init__(self):
        #Initialize global variables-------------------------------------------------
        self.initial_state_ref = 0
        self.store_initial_state_ref = 0
        self.store_current_state = []
        if 'maze' in cons.environment_name.lower():
            self.env = Maze( cons.environment_name[4] )
        elif 'woods' in cons.environment_name.lower():
            self.env = Woods( cons.environment_name[5] )
        if cons.convertBinary:
            self.conversion = BinaryConversion( self.env )
        else:
            self.conversion = Conversion()

        print("----------------------------------------------------------------------------")
        print("XCS Code Demo:")
        print("----------------------------------------------------------------------------")
        print("Environment " + cons.environment_name + ": Processing environment... " )
        self.action_list, _ = self.env.listOfActions()          # Stores all possible discrete phenotype states/classes
        self.formatted_states = self.formatStates()             # Store all possible positions along with their conditions
        self.number_of_attributes = len( self.formatted_states[ 0 ][ 0 ] )
        initial_position = self.formatted_states[ self.initial_state_ref ][ 1 ]
        self._moveAgentToPosition( initial_position )


    def shuffleStates(self):
        """ Shuffle property formatted_states """
        random.shuffle( self.formatted_states )


    def formatStates(self):
        """ Get the data into a format convenient for the algorithm to interact with. Specifically each instance is stored in a list as follows; [Attribute States, Phenotype, InstanceID] """
        observation_position_list, self.number_of_states = self.env.listAllObservationsAndPositions()
        formatted = []
        for i in range( self.number_of_states ):
            condition = self.conversion.convertObservationToCondition( observation_position_list[i][0] )
            formatted.append( [ condition, observation_position_list[i][1] ] )
        random.shuffle( formatted ) #One time randomization of the order the of the instances in the data, so that if the data was ordered by phenotype, this potential learning bias (based on instance ordering) is eliminated.
        return formatted


    def getCurrentCondition(self):
        """ Returns the current observed condition of the agent. """
        return self.conversion.convertObservationToCondition( self.env.detector() )


    def getInitialState(self):
        """ Returns the initial state of the current trial/game. """
        return self.formatted_states[ self.initial_state_ref ][ 0 ]


    def newGame(self, training_mode = True):
        """  Shifts the agent to a new initial state/position for a new trial/game. """
        if self.initial_state_ref < ( self.number_of_states - 1 ):
            self.initial_state_ref += 1
        else:  #Once learning has completed an epoch (i.e. a cycle of iterations though the entire training dataset) it starts back at the first instance in the data)
            if cons.shuffle_every_iteration and training_mode:
                self.shuffleStates()
            self.initial_state_ref = 0
        current_condition, current_position = self.formatted_states[ self.initial_state_ref ]
        self._moveAgentToPosition( current_position )
        return current_condition


    def affectEnvironment(self, action):
        """ Execute the action on environment and get response from environment. """
        return self.env.affector( *action )


    def resetInitialStateRef(self):
        """ Resets the environment back to the first instance in the current data set. """
        self.store_initial_state_ref = 0
        current_position = self.formatted_states[ self.initial_state_ref ][ 1 ]
        self._moveAgentToPosition( current_position )


    def startEvaluationMode(self):
        """ Turns on evaluation mode.  Saves the state we left off in the environment. """
        self.store_current_position = self.env.getAgentPosition()
        self.store_initial_state_ref = self.initial_state_ref
        self._moveAgentToPosition( self.formatted_states[ self.initial_state_ref ][ 1 ] )


    def stopEvaluationMode(self):
        """ Turns off evaluation mode.  Re-establishes place in environment."""
        self._moveAgentToPosition( self.store_current_position )
        self.initial_state_ref = self.store_initial_state_ref
        self._moveAgentToPosition( self.store_current_position )


    def _moveAgentToPosition(self, position):
        """ Convert state received from algorithm to coordinates in environment and move agent to the coordinates. """
        self.env.assignPosition( *position )


class Conversion:
    def convertObservationToCondition(self, observation):
        return observation


class BinaryConversion( Conversion ):
    def __init__(self, env ):
        all_states, number_of_states = env.listAllObservationsAndPositions()
        object_list = []
        for i in range( number_of_states ):
            for obj in all_states[ i ][ 0 ]:
                if obj not in object_list:
                    object_list.append( obj )
        self.bits_per_object = math.ceil( math.log( len( object_list ), 2) )


    def convertObservationToCondition(self, observation):
        condition_str = ''
        condition = []
        for obj in observation:
            binary_str = bin( obj )
            string_array = binary_str.split( 'b' )
            binary = string_array[ 1 ]
            while len( binary ) < self.bits_per_object:
                binary= "0" + binary
            condition_str = condition_str + binary

        for i in range( len( condition_str ) ):
            condition.append( condition_str[i] )
        return condition
