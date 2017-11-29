"""
Name:        xcs_offline_environment.py
Authors:     Bao Trung
Contact:     baotrung@ecs.vuw.ac.nz
Created:     July, 2017
Description:
---------------------------------------------------------------------------------------------------------------------------------------------------------
XCS: Michigan-style Learning Classifier System - A LCS for Reinforcement Learning.  This XCS follows the version descibed in "An Algorithmic Description of XCS" published by Martin Butz and Stewart Wilson (2002).

---------------------------------------------------------------------------------------------------------------------------------------------------------
"""

#Import Required Modules--------------------------------------
from xcs_data_management import DataManagement
from xcs_constants import *
import sys
#-------------------------------------------------------------

class Offline_Environment:
    def __init__(self):
        #Initialize global variables-------------------------------------------------
        self.dataRef = 0
        self.storeDataRef = 0
        self.format_data = DataManagement(cons.trainFile, cons.testFile)

        #Initialize the first dataset instance to be passed to XCS
        self.currentTrainState = self.format_data.trainFormatted[self.dataRef][0]
        self.currentTrainPhenotype = self.format_data.trainFormatted[self.dataRef][1]
        if cons.testFile == 'None':
            pass
        else:
            self.currentTestState = self.format_data.testFormatted[self.dataRef][0]
            self.currentTestPhenotype = self.format_data.testFormatted[self.dataRef][1]


    def getTrainInstance(self):
        """ Returns the current training instance. """
        return [self.currentTrainState, self.currentTrainPhenotype]


    def getTestInstance(self):
        """ Returns the current training instance. """
        return [self.currentTestState, self.currentTestPhenotype]


    def newInstance(self, isTraining):
        """  Shifts the environment to the next instance in the data. """
        #-------------------------------------------------------
        # Training Data
        #-------------------------------------------------------
        if isTraining:
            if self.dataRef < (self.format_data.numb_train_instances-1):
                self.dataRef += 1
                self.currentTrainState = self.format_data.trainFormatted[self.dataRef][0]
                self.currentTrainPhenotype = self.format_data.trainFormatted[self.dataRef][1]
            else:  #Once learning has completed an epoch (i.e. a cycle of iterations though the entire training dataset) it starts back at the first instance in the data)
                self.resetDataRef(isTraining)

        #-------------------------------------------------------
        # Testing Data
        #-------------------------------------------------------
        else:
            if self.dataRef < (self.format_data.numb_test_instances-1):
                self.dataRef += 1
                self.currentTestState = self.format_data.testFormatted[self.dataRef][0]
                self.currentTestPhenotype = self.format_data.testFormatted[self.dataRef][1]


    def resetDataRef(self, isTraining):
        """ Resets the environment back to the first instance in the current data set. """
        self.dataRef = 0
        if isTraining:
            self.currentTrainState = self.format_data.trainFormatted[self.dataRef][0]
            self.currentTrainPhenotype = self.format_data.trainFormatted[self.dataRef][1]
        else:
            self.currentTestState = self.format_data.testFormatted[self.dataRef][0]
            self.currentTestPhenotype = self.format_data.testFormatted[self.dataRef][1]


    def startEvaluationMode(self):
        """ Turns on evaluation mode.  Saves the instance we left off in the training data. """
        self.storeDataRef = self.dataRef


    def stopEvaluationMode(self):
        """ Turns off evaluation mode.  Re-establishes place in dataset."""
        self.dataRef = self.storeDataRef
