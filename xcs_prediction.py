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
    def __init__(self, population, is_testing = False):
        """ Constructs the voting array and determines the prediction decision. """
        self.decision = None
        #-------------------------------------------------------
        # DISCRETE PHENOTYPES (CLASSES)
        #-------------------------------------------------------
        if cons.env.format_data.discrete_action:
            self.prediction = {}
            denominator = {}
            tiebreak_numerosity = {}
            tiebreak_timestamp = {}

            for action in cons.env.format_data.action_list:
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

            for action in cons.env.format_data.action_list:
                if denominator[ action ] != 0:
                    self.prediction[ action ] /= denominator[ action ]
                #else:
                #    self.prediction[eachClass] = 0

            max_prediction = 0.0
            self.best_set = [] #Prediction is set up to handle best class ties for problems with more than 2 classes
            for action in cons.env.format_data.action_list:
                if tiebreak_numerosity[ action ] != 0 and self.prediction[ action ] >= max_prediction:
                    max_prediction = self.prediction[ action ]

            for action in cons.env.format_data.action_list:
                if tiebreak_numerosity[ action ] != 0 and self.prediction[ action ] == max_prediction: #Tie for best class
                    self.best_set.append( action )
            self.possible_actions = [ k for k,_ in self.prediction.items() if tiebreak_numerosity[k] != 0 ]
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

        #-------------------------------------------------------
        # CONTINUOUS PHENOTYPES - NOT SURE HOW TO ADJUST THIS PART FOR XCS?
        #-------------------------------------------------------
        else:
            if len(population.match_set) < 1:
                print("empty matchSet")
                self.decision = None
            else:
                #IDEA - outputs a single continuous prediction value(closeness to this prediction accuracy will dictate accuracy). In determining this value we examine all ranges
                phenotype_range = cons.env.format_data.action_list[1] - cons.env.format_data.action_list[0] #Difference between max and min phenotype values observed in data.
                prediction_value = 0
                weight_sum = 0
                for ref in population.match_set:
                    cl = population.pop_set[ref]
                    local_range = cl.action[1] - cl.action[0]
                    value_weight = ( phenotype_range/float( local_range ) ) * cl.prediction * cl.fitness
                    local_average = cl.action[1]+cl.action[0] / 2.0

                    weight_sum += value_weight
                    prediction_value += value_weight * local_average
                if weight_sum == 0.0:
                    self.decision = None
                else:
                    self.decision = prediction_value / float( weight_sum )
                self.prediction[self.decision] = 0.0
                denominator[self.decision] = 0.0
                for ref in population.match_set:
                    cl = population.pop_set[ref]
                    if cl.action[1] >= self.decision and cl.action[0] <= self.decision:
                        self.prediction[self.decision] += cl.prediction * cl.fitness
                        denominator[self.decision] += cl.fitness
                    self.prediction[self.decision] /= denominator[self.decision]


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


    def getHighestPredictionAction(self):
        return self.best_set


    def getDecision(self):
        """ Returns prediction decision. """
        if self.decision == None or self.decision == 'Tie':
            if cons.env.format_data.discrete_action:
                self.decision = random.choice( self.possible_actions )
            else:
                self.decision = random.randrange(cons.env.format_data.action_list[0],cons.env.format_data.action_list[1],(cons.env.format_data.action_list[1]-cons.env.format_data.action_list[0])/float(1000))
        return self.decision