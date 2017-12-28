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
    def __init__(self, population, testingMode = False):
        """ Constructs the voting array and determines the prediction decision. """
        self.decision = None
        self.predictedPayoff = None
        #-------------------------------------------------------
        # DISCRETE PHENOTYPES (CLASSES)
        #-------------------------------------------------------
        if cons.env.format_data.discrete_action:
            self.prediction = {}
            self.denominator = {}
            self.tieBreak_Numerosity = {}
            self.tieBreak_TimeStamp = {}

            for eachClass in cons.env.format_data.action_list:
                self.prediction[eachClass] = None
                self.denominator[eachClass] = 0.0
                self.tieBreak_Numerosity[eachClass] = 0.0
                self.tieBreak_TimeStamp[eachClass] = 0.0

            for ref in population.match_set:
                cl = population.pop_set[ref]
                if self.prediction[cl.action] == None:
                    self.prediction[cl.action] = cl.prediction * cl.fitness
                else:
                    self.prediction[cl.action] += cl.prediction * cl.fitness# * cl.numerosity
                self.denominator[cl.action] += cl.fitness# * cl.numerosity
                self.tieBreak_Numerosity[cl.action] += cl.numerosity
                self.tieBreak_TimeStamp[cl.action] += cl.init_timestamp

            for eachClass in cons.env.format_data.action_list:
                if self.denominator[eachClass] != 0:
                    self.prediction[eachClass] /= self.denominator[eachClass]
                #else:
                #    self.prediction[eachClass] = 0

            highVal = 0.0
            self.bestClass = [] #Prediction is set up to handle best class ties for problems with more than 2 classes
            for thisClass in cons.env.format_data.action_list:
                if self.prediction[thisClass] != None and self.prediction[thisClass] >= highVal:
                    highVal = self.prediction[thisClass]

            for thisClass in cons.env.format_data.action_list:
                if self.prediction[thisClass] != None and self.prediction[thisClass] == highVal: #Tie for best class
                    self.bestClass.append(thisClass)
            if random.random() >= cons.exploration or testingMode == True:
                # select by exploitation
                #---------------------------
                if highVal == 0.0:
                    self.decision = None
                #-----------------------------------------------------------------------
                elif len(self.bestClass) > 1: #Randomly choose between the best tied classes
                    bestNum = 0
                    newBestClass = []
                    for thisClass in self.bestClass:
                        if self.tieBreak_Numerosity[thisClass] >= bestNum:
                            bestNum = self.tieBreak_Numerosity[thisClass]

                    for thisClass in self.bestClass:
                        if self.tieBreak_Numerosity[thisClass] == bestNum:
                            newBestClass.append(thisClass)
                    #-----------------------------------------------------------------------
                    if len(newBestClass) > 1:  #still a tie
                        bestStamp = 0
                        newestBestClass = []
                        for thisClass in newBestClass:
                            if self.tieBreak_TimeStamp[thisClass] >= bestStamp:
                                bestStamp = self.tieBreak_TimeStamp[thisClass]

                        for thisClass in newBestClass:
                            if self.tieBreak_TimeStamp[thisClass] == bestStamp:
                                newestBestClass.append(thisClass)
                        #-----------------------------------------------------------------------
                        if len(newestBestClass) > 1: # Prediction is completely tied - XCS has no useful information for making a prediction
                            self.decision = 'Tie'
                    else:
                        self.decision = newBestClass[0]
                #----------------------------------------------------------------------
                else: #One best class determined by fitness vote
                    self.decision = self.bestClass[0]
            else:
                # select by exploration
                self.decision = random.choice([k for k,v in self.prediction.items() if v is not None])

        #-------------------------------------------------------
        # CONTINUOUS PHENOTYPES - NOT SURE HOW TO ADJUST THIS PART FOR XCS?
        #-------------------------------------------------------
        else:
            if len(population.match_set) < 1:
                print("empty matchSet")
                self.decision = None
            else:
                #IDEA - outputs a single continuous prediction value(closeness to this prediction accuracy will dictate accuracy). In determining this value we examine all ranges
                phenotypeRange = cons.env.format_data.action_list[1] - cons.env.format_data.action_list[0] #Difference between max and min phenotype values observed in data.
                predictionValue = 0
                valueWeightSum = 0
                for ref in population.match_set:
                    cl = population.pop_set[ref]
                    localRange = cl.action[1] - cl.action[0]
                    valueWeight = (phenotypeRange/float(localRange)) * cl.prediction * cl.fitness
                    localAverage = cl.action[1]+cl.action[0] / 2.0

                    valueWeightSum += valueWeight
                    predictionValue += valueWeight * localAverage
                if valueWeightSum == 0.0:
                    self.decision = None
                else:
                    self.decision = predictionValue / float(valueWeightSum)
                self.prediction[self.decision] = 0.0
                self.denominator[self.decision] = 0.0
                for ref in population.match_set:
                    cl = population.pop_set[ref]
                    if cl.action[1] >= self.decision and cl.action[0] <= self.decision:
                        self.prediction[self.decision] += cl.prediction * cl.fitness
                        self.denominator[self.decision] += cl.fitness
                    self.prediction[self.decision] /= self.denominator[self.decision]


    def getFitnessSum(self,population,low,high):
        """ Get the fitness sum of rules in the rule-set. For continuous phenotype prediction. """
        fitSum = 0
        for ref in population.match_set:
            cl = population.pop_set[ref]
            if cl.action[0] <= low and cl.action[1] >= high: #if classifier range subsumes segment range.
                fitSum += cl.fitness
        return fitSum


    def getPredictionArray(self):
        return self.prediction


    def getPredictedPayoff(self):
        self.predictedPayoff = self.prediction[self.decision]
        return self.predictedPayoff


    def getHighestPredictionAction(self):
        return self.bestClass


    def getDecision(self):
        """ Returns prediction decision. """
        if self.decision == None or self.decision == 'Tie':
            if cons.env.format_data.discrete_action:
                self.decision = random.choice([k for k,v in self.prediction.items() if v is not None])
            else:
                self.decision = random.randrange(cons.env.format_data.action_list[0],cons.env.format_data.action_list[1],(cons.env.format_data.action_list[1]-cons.env.format_data.action_list[0])/float(1000))
        return self.decision