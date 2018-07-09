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
import random

from xcs_constants import *


#-------------------------------------
class Prediction:
    def __init__(self, population):
        """ Constructs the voting array and determines the prediction decision. """
        self.decision = None
        self.prediction = {}
        denominator = {}
        self.tiebreak_numerosity = {}
        self.tiebreak_timestamp = {}

        for action in cons.env.format_data.action_list:
            self.prediction[ action ] = 0.0
            denominator[ action ] = 0.0
            self.tiebreak_numerosity[ action ] = 0
            self.tiebreak_timestamp[ action ] = 0

        for ref in population.match_set:
            cl = population.pop_set[ref]
            self.prediction[cl.action] += cl.prediction * cl.fitness
            denominator[cl.action] += cl.fitness
            self.tiebreak_numerosity[cl.action] += cl.numerosity
            self.tiebreak_timestamp[cl.action] += cl.init_timestamp

        for action in cons.env.format_data.action_list:
            if denominator[ action ] != 0:
                self.prediction[ action ] /= denominator[ action ]
            #else:
            #    self.prediction[eachClass] = 0

        max_prediction = 0.0
        self.best_set = [] #Prediction is set up to handle best class ties for problems with more than 2 classes
        for action in cons.env.format_data.action_list:
            if self.tiebreak_numerosity[ action ] != 0 and self.prediction[ action ] >= max_prediction:
                max_prediction = self.prediction[ action ]

        for action in cons.env.format_data.action_list:
            if self.tiebreak_numerosity[ action ] != 0 and self.prediction[ action ] == max_prediction: #Tie for best class
                self.best_set.append( action )
        self.possible_actions = [ k for k,_ in self.prediction.items() if self.tiebreak_numerosity[k] != 0 ]
        self.possible_actions.sort()


    def getFitnessSum(self,population,low,high):
        """ Get the fitness sum of rules in the rule-set. For continuous phenotype prediction. """
        fitness_sum = 0
        for ref in population.match_set:
            cl = population.pop_set[ref]
            if cl.action[0] <= low and cl.action[1] >= high: #if classifier range subsumes segment range.
                fitness_sum += cl.fitness
        return fitness_sum

    def getPredictionArray(self):
        return self.prediction

    def getPredictedPayoff(self):
        return self.prediction[ self.decision ]

    def decide(self, exploring=True):
        """ Returns prediction decision. """
        if exploring:
            self.decision = random.choice( self.possible_actions )
        elif len( self.best_set ) == 1:
            self.decision = self.best_set[0]
        else:
            max_numerosity = 0
            new_best_action = []
            for action in self.best_set:
                if self.tiebreak_numerosity[action] >= max_numerosity:
                    max_numerosity = self.tiebreak_numerosity[action]
            for action in self.best_set:
                if self.tiebreak_numerosity[action] == max_numerosity:
                    new_best_action.append(action)
            # -----------------------------------------------------------------------
            if len(new_best_action) > 1:  # still a tie
                latest_created = 0
                newest_best_action = []
                for action in new_best_action:
                    if self.tiebreak_timestamp[action] >= latest_created:
                        latest_created = self.tiebreak_timestamp[action]

                for action in new_best_action:
                    if self.tiebreak_timestamp[action] == latest_created:
                        newest_best_action.append(action)
                # -----------------------------------------------------------------------
                if len( newest_best_action ) == 1:  # Prediction is completely tied - XCS has no useful information for making a prediction
                    self.decision = newest_best_action[0]
                else:
                    self.decision = random.choice( newest_best_action )
            else:
                self.decision = new_best_action[0]
        return self.decision
