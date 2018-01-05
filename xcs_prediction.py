"""
Name:        xcs_prediction.py
Authors:     Bao Trung
Contact:     baotrung@ecs.vuw.ac.nz
Created:     July, 2017
Description:
---------------------------------------------------------------------------------------------------------------------------------------------------------
XCS: Michigan-style Learning Classifier System - A LCS for Reinforcement Learning.  This XCS follows the version descibed in "An Algorithmic Description of XCS" published by Martin Butz and Stewart Wilson (2002).

---------------------------------------------------------------------------------------------------------------------------------------------------------
"""

#Import Required Modules--------------
from xcs_constants import *
import random
#-------------------------------------

class Prediction:
    def __init__(self, population, is_testing = False):
        """ Constructs the voting array and determines the prediction decision. """
        self.decision = None
        self.prediction = {}
        denominator = {}
        tiebreak_numerosity = {}
        tiebreak_timestamp = {}

        for action in cons.env.action_list:
            self.prediction[ action ] = 0.0
            denominator[ action ] = 0.0
            tiebreak_numerosity[ action ] = 0
            tiebreak_timestamp[ action ] = 0

        for ref in population.match_set:
            cl = population.pop_set[ref]
            self.prediction[cl.action] += cl.prediction * cl.fitness# * cl.numerosity
            denominator[cl.action] += cl.fitness# * cl.numerosity
            tiebreak_numerosity[cl.action] += cl.numerosity
            tiebreak_timestamp[cl.action] += cl.init_timestamp

        for action in cons.env.action_list:
            if denominator[ action ] != 0:
                self.prediction[ action ] /= denominator[ action ]
            #else:
            #    self.prediction[eachClass] = 0

        max_prediction = 0.0
        self.best_set = [] #Prediction is set up to handle best class ties for problems with more than 2 classes
        for action in cons.env.action_list:
            if tiebreak_numerosity[ action ] != 0 and self.prediction[ action ] >= max_prediction:
                max_prediction = self.prediction[ action ]

        for action in cons.env.action_list:
            if tiebreak_numerosity[ action ] != 0 and self.prediction[ action ] == max_prediction: #Tie for best class
                self.best_set.append( action )
        self.possible_actions = [ k for k,v in self.prediction.items() if v is not None ]
        self.possible_actions.sort()
        if random.random() >= cons.exploration or is_testing == True:
            # select by exploitation
            #---------------------------
            self.is_exploit = True
            if max_prediction == 0.0:
                self.decision = None
            #-----------------------------------------------------------------------
            elif len( self.best_set ) > 1: #Randomly choose between the best tied classes
                max_numerosity = 0
                new_best_action = []
                for action in self.best_set:
                    if tiebreak_numerosity[ action ] >= max_numerosity:
                        max_numerosity = tiebreak_numerosity[ action ]

                for action in self.best_set:
                    if tiebreak_numerosity[ action ] == max_numerosity:
                        new_best_action.append( action )
                #-----------------------------------------------------------------------
                if len( new_best_action ) > 1:  #still a tie
                    latest_created = 0
                    newest_best_action = []
                    for action in new_best_action:
                        if tiebreak_timestamp[ action ] >= latest_created:
                            latest_created = tiebreak_timestamp[ action ]

                    for action in new_best_action:
                        if tiebreak_timestamp[ action ] == latest_created:
                            newest_best_action.append( action )
                    #-----------------------------------------------------------------------
                    if len(newest_best_action) > 1: # Prediction is completely tied - XCS has no useful information for making a prediction
                        self.decision = 'Tie'
                    else:
                        self.decision = newest_best_action[0]
                else:
                    self.decision = new_best_action[0]
            #----------------------------------------------------------------------
            else: #One best class determined by fitness vote
                self.decision = self.best_set[0]
        else:
            # select by exploration
            self.decision = random.choice( self.possible_actions )
            self.is_exploit = False


    def getPredictionArray(self):
        return self.prediction


    def getActionPrediction(self):
        return self.prediction[ self.decision ]


    def getHighestPredictionAction(self):
        return self.best_set


    def getDecision(self):
        """ Returns prediction decision. """
        if self.decision == None or self.decision == 'Tie':
            self.decision = random.choice( self.possible_actions )
        return self.decision

