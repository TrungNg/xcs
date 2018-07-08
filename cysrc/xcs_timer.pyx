"""
Name:        xcs_timer.py
Authors:     Bao Trung
Contact:     baotrung@ecs.vuw.ac.nz
Created:     July, 2017
Description:
---------------------------------------------------------------------------------------------------------------------------------------------------------
XCS: Michigan-style Learning Classifier System - A LCS for Reinforcement Learning.  This XCS follows the version descibed in "An Algorithmic Description of XCS" published by Martin Butz and Stewart Wilson (2002).

---------------------------------------------------------------------------------------------------------------------------------------------------------
"""

#Import Required Modules---------------
from xcs_constants import *
import time
#--------------------------------------

class Timer:
    def __init__(self):
        # Global Time objects
        self.globalStartRef = time.time()
        self.globalTime = 0.0
        self.addedTime = 0.0

        # Match Time Variables
        self.startRefMatching = 0.0
        self.globalMatching = 0.0

        # Deletion Time Variables
        self.startRefDeletion = 0.0
        self.globalDeletion = 0.0

        # Subsumption Time Variables
        self.startRefSubsumption = 0.0
        self.globalSubsumption = 0.0

        # Selection Time Variables
        self.startRefSelection = 0.0
        self.globalSelection = 0.0

        # Evaluation Time Variables
        self.startRefEvaluation = 0.0
        self.globalEvaluation = 0.0


    # ************************************************************
    def startTimeMatching(self):
        """ Tracks MatchSet Time """
        self.startRefMatching = time.time()

    def stopTimeMatching(self):
        """ Tracks MatchSet Time """
        diff = time.time() - self.startRefMatching
        self.globalMatching += diff

    # ************************************************************
    def startTimeDeletion(self):
        """ Tracks Deletion Time """
        self.startRefDeletion = time.time()

    def stopTimeDeletion(self):
        """ Tracks Deletion Time """
        diff = time.time() - self.startRefDeletion
        self.globalDeletion += diff

    # ************************************************************
    def startTimeSubsumption(self):
        """Tracks Subsumption Time """
        self.startRefSubsumption = time.time()

    def stopTimeSubsumption(self):
        """Tracks Subsumption Time """
        diff = time.time() - self.startRefSubsumption
        self.globalSubsumption += diff

    # ************************************************************
    def startTimeSelection(self):
        """ Tracks Selection Time """
        self.startRefSelection = time.time()

    def stopTimeSelection(self):
        """ Tracks Selection Time """
        diff = time.time() - self.startRefSelection
        self.globalSelection += diff

    # ************************************************************
    def startTimeEvaluation(self):
        """ Tracks Evaluation Time """
        self.startRefEvaluation = time.time()

    def stopTimeEvaluation(self):
        """ Tracks Evaluation Time """
        diff = time.time() - self.startRefEvaluation
        self.globalEvaluation += diff

    # ************************************************************
    def returnGlobalTimer(self):
        """ Set the global end timer, call at very end of algorithm. """
        self.globalTime = (time.time() - self.globalStartRef) + self.addedTime #Reports time in minutes, addedTime is for population reboot.
        return self.globalTime/ 60.0


    def setTimerRestart(self, remakeFile):
        """ Sets all time values to the those previously evolved in the loaded popFile.  """
        try:
            fileObject = open(remakeFile+"_PopStats.txt", 'r')  # opens each datafile to read.
        except Exception as inst:
            print(type(inst))
            print(inst.args)
            print(inst)
            print('cannot open', remakeFile+"_PopStats.txt")
            raise

        timeDataRef = 18

        tempLine = None
        for i in range(timeDataRef):
            tempLine = fileObject.readline()
        tempList = tempLine.strip().split('\t')
        self.addedTime = float(tempList[1]) * 60 #previous global time added with Reboot.

        tempLine = fileObject.readline()
        tempList = tempLine.strip().split('\t')
        self.globalMatching = float(tempList[1]) * 60

        tempLine = fileObject.readline()
        tempList = tempLine.strip().split('\t')
        self.globalDeletion = float(tempList[1]) * 60

        tempLine = fileObject.readline()
        tempList = tempLine.strip().split('\t')
        self.globalSubsumption = float(tempList[1]) * 60

        tempLine = fileObject.readline()
        tempList = tempLine.strip().split('\t')
        self.globalSelection = float(tempList[1]) * 60

        tempLine = fileObject.readline()
        tempList = tempLine.strip().split('\t')
        self.globalEvaluation = float(tempList[1]) * 60

        fileObject.close()

    ##############################################################################################

    def reportTimes(self):
        """ Reports the time summaries for this run. Returns a string ready to be printed out."""
        outputTime = "Global Time\t"+str(self.globalTime/ 60.0)+ \
        "\nMatching Time\t" + str(self.globalMatching/ 60.0)+ \
        "\nDeletion Time\t" + str(self.globalDeletion/ 60.0)+ \
        "\nSubsumption Time\t" + str(self.globalSubsumption/ 60.0)+ \
        "\nSelection Time\t"+str(self.globalSelection/ 60.0)+ \
        "\nEvaluation Time\t"+str(self.globalEvaluation/ 60.0) + "\n"

        return outputTime