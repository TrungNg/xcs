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
from xcs_constants import cons
#import sys
#-------------------------------------------------------------

class Offline_Environment:
    def __init__(self):
        #Initialize global variables-------------------------------------------------
        self.data_ref = 0
        self.saved_dat_ref = 0
        self.format_data = DataManagement(cons.train_file, cons.test_file)

        #Initialize the first dataset instance to be passed to XCS
        self.train_inst_condition = self.format_data.formatted_train_data[self.data_ref][0]
        self.train_inst_phenotype = self.format_data.formatted_train_data[self.data_ref][1]
        if cons.test_file == 'None':
            pass
        else:
            self.test_inst_condition = self.format_data.formatted_test_data[self.data_ref][0]
            self.test_inst_phenotype = self.format_data.formatted_test_data[self.data_ref][1]


    def getTrainInstance(self):
        """ Returns the current training instance. """
        self.train_inst_condition = self.format_data.formatted_train_data[self.data_ref][0]
        self.train_inst_phenotype = self.format_data.formatted_train_data[self.data_ref][1]
        instance = [self.train_inst_condition, self.train_inst_phenotype]
        self.newInstance( True )
        return instance


    def getTestInstance(self):
        """ Returns the current training instance. """
        self.test_inst_condition = self.format_data.formatted_test_data[self.data_ref][0]
        self.test_inst_phenotype = self.format_data.formatted_test_data[self.data_ref][1]
        instance = [self.test_inst_condition, self.test_inst_phenotype]
        self.newInstance( False )
        return instance


    def newInstance(self, is_train):
        """  Shifts the environment to the next instance in the data. """
        #-------------------------------------------------------
        # Training Data
        #-------------------------------------------------------
        if is_train:
            if self.data_ref < (self.format_data.numb_train_instances-1):
                self.data_ref += 1
            else:  #Once learning has completed an epoch (i.e. a cycle of iterations though the entire training dataset) it starts back at the first instance in the data)
                self.resetDataRef(is_train)

        #-------------------------------------------------------
        # Testing Data
        #-------------------------------------------------------
        else:
            if self.data_ref < (self.format_data.numb_test_instances-1):
                self.data_ref += 1


    def resetDataRef(self, is_train):
        """ Resets the environment back to the first instance in the current data set. """
        self.data_ref = 0
        if is_train:
            self.train_inst_condition = self.format_data.formatted_train_data[self.data_ref][0]
            self.train_inst_phenotype = self.format_data.formatted_train_data[self.data_ref][1]
        else:
            self.test_inst_condition = self.format_data.formatted_test_data[self.data_ref][0]
            self.test_inst_phenotype = self.format_data.formatted_test_data[self.data_ref][1]


    def startEvaluationMode(self):
        """ Turns on evaluation mode.  Saves the instance we left off in the training data. """
        self.saved_dat_ref = self.data_ref


    def stopEvaluationMode(self):
        """ Turns off evaluation mode.  Re-establishes place in dataset."""
        self.data_ref = self.saved_dat_ref
