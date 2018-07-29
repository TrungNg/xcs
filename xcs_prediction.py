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
import crandom
from xcs_constants import *
#-------------------------------------

class Prediction:
    def __init__(self, population):
        """ Constructs the voting array and determines the prediction decision. """
        self.decision = None
        self.prediction = {}
        denominator = {}
        self.tiebreak_numerosity = {}

        for action in cons.env.format_data.action_list:
            self.prediction[ action ] = 0.0
            denominator[ action ] = 0.0
            self.tiebreak_numerosity[ action ] = 0

        for ref in population.match_set:
            cl = population.pop_set[ref]
            self.prediction[cl.action] += cl.prediction * cl.fitness
            denominator[cl.action] += cl.fitness
            self.tiebreak_numerosity[cl.action] += cl.numerosity

        for action in cons.env.format_data.action_list:
            if denominator[ action ] != 0:
                self.prediction[ action ] /= denominator[ action ]


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

    def decide1(self, exploring=True):
        """ Returns prediction decision. """
        if exploring:
            self.decision = crandom.choice( cons.env.format_data.action_list )
        else:
            max_prediction = 0.0
            best_set = []  # Prediction is set up to handle best class ties for problems with more than 2 classes
            for action in cons.env.format_data.action_list:
                if self.prediction[action] >= max_prediction:
                    max_prediction = self.prediction[action]

            for action in cons.env.format_data.action_list:
                if self.prediction[action] == max_prediction:  # Tie for best class
                    best_set.append(action)
            if len( best_set ) == 1:
                self.decision = best_set[0]
            else:
                self.decision = crandom.choice( best_set )
        return self.decision

    def decide2(self, exploring=True):
        """ Returns prediction decision. """
        if exploring:
            self.decision = crandom.choice( cons.env.format_data.action_list )
        else:
            self.decision = cons.env.format_data.action_list[0]
            for action in cons.env.format_data.action_list:
                if self.prediction[ action ] > self.prediction[ self.decision ]:
                    self.decision = action
        return self.decision

    def decide(self, exploring=True):
        """ Returns prediction decision. """
        if exploring:
            self.decision = crandom.choice( cons.env.format_data.action_list )
        else:
            max_prediction = 0.0
            best_set = []  # Prediction is set up to handle best class ties for problems with more than 2 classes
            for action in cons.env.format_data.action_list:
                if self.prediction[action] >= max_prediction:
                    max_prediction = self.prediction[action]

            for action in cons.env.format_data.action_list:
                if self.prediction[action] == max_prediction:  # Tie for best class
                    best_set.append(action)
            if len( best_set ) == 1:
                self.decision = best_set[0]
            else:
                max_numerosity = 0
                new_best_action = []
                for action in best_set:
                    if self.tiebreak_numerosity[action] >= max_numerosity:
                        max_numerosity = self.tiebreak_numerosity[action]
                for action in best_set:
                    if self.tiebreak_numerosity[action] == max_numerosity:
                        new_best_action.append(action)
                # -----------------------------------------------------------------------
                if len(new_best_action) == 1:
                    self.decision = new_best_action[0]
                else:  # still a tie
                    self.decision = crandom.choice(new_best_action)
        return self.decision