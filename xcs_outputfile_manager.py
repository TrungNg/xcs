"""
Name:        xcs_outputfile_manager.py
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
import copy
#-------------------------------------

class OutputFileManager:
    def writePopStats(self, outFile, trainEval, testEval, exploreIter, pop, correct):
        """ Makes output text file which includes all of the evaluation statistics for a complete analysis of all training and testing data on the current XCS rule population. """
        try:
            popStatsOut = open(outFile + '_'+ str(exploreIter)+'_PopStats.txt','w')
        except Exception as inst:
            print(type(inst))
            print(inst.args)
            print(inst)
            print('cannot open', outFile + '_'+ str(exploreIter)+'_PopStats.txt')
            raise
        else:
            print("Writing Population Statistical Summary File...")

        #Evaluation of pop
        popStatsOut.write("Performance Statistics:------------------------------------------------------------------------\n")
        popStatsOut.write("Training Accuracy\tTesting Accuracy\tTraining Coverage\tTesting Coverage\n")

        if cons.testFile != 'None':
            popStatsOut.write(str(trainEval[0])+"\t")
            popStatsOut.write(str(testEval[0])+"\t")
            popStatsOut.write(str(trainEval[1]) +"\t")
            popStatsOut.write(str(testEval[1])+"\n\n")
        elif cons.trainFile != 'None':
            popStatsOut.write(str(trainEval[0])+"\t")
            popStatsOut.write("NA\t")
            popStatsOut.write(str(trainEval[1]) +"\t")
            popStatsOut.write("NA\n\n")
        else:
            popStatsOut.write("NA\t")
            popStatsOut.write("NA\t")
            popStatsOut.write("NA\t")
            popStatsOut.write("NA\n\n")

        popStatsOut.write("Population Characterization:------------------------------------------------------------------------\n")
        popStatsOut.write("MacroPopSize\tMicroPopSize\tGenerality\n")
        popStatsOut.write(str(len(pop.pop_set))+"\t"+ str(pop.micro_size)+"\t"+str(pop.mean_generality)+"\n\n")

        popStatsOut.write("SpecificitySum:------------------------------------------------------------------------\n")
        headList = cons.env.format_data.train_headers #preserve order of original dataset

        for i in range(len(headList)):
            if i < len(headList)-1:
                popStatsOut.write(str(headList[i])+"\t")
            else:
                popStatsOut.write(str(headList[i])+"\n")

        # Prints out the Specification Sum for each attribute
        for i in range(len(pop.attribute_spec_list)):
            if i < len(pop.attribute_spec_list)-1:
                popStatsOut.write(str(pop.attribute_spec_list[i])+"\t")
            else:
                popStatsOut.write(str(pop.attribute_spec_list[i])+"\n")

        popStatsOut.write("\nAccuracySum:------------------------------------------------------------------------\n")
        for i in range(len(headList)):
            if i < len(headList)-1:
                popStatsOut.write(str(headList[i])+"\t")
            else:
                popStatsOut.write(str(headList[i])+"\n")

        # Prints out the Accuracy Weighted Specification Count for each attribute
        for i in range(len(pop.attribute_acc_list)):
            if i < len(pop.attribute_acc_list)-1:
                popStatsOut.write(str(pop.attribute_acc_list[i])+"\t")
            else:
                popStatsOut.write(str(pop.attribute_acc_list[i])+"\n")

        #Time Track ---------------------------------------------------------------------------------------------------------
        popStatsOut.write("\nRun Time(in minutes):------------------------------------------------------------------------\n")
        popStatsOut.write(cons.timer.reportTimes())
        popStatsOut.write("\nCorrectTrackerSave:------------------------------------------------------------------------\n")
        for i in range(len(correct)):
            popStatsOut.write(str(correct[i])+"\t")

        popStatsOut.close()


    def writePop(self, outFile, exploreIter, pop):
        """ Writes a tab delimited text file outputting the entire evolved rule population, including conditions, phenotypes, and all rule parameters. """
        try:
            rulePopOut = open(outFile + '_'+ str(exploreIter)+'_RulePop.txt','w')
        except Exception as inst:
            print(type(inst))
            print(inst.args)
            print(inst)
            print('cannot open', outFile + '_'+ str(exploreIter)+'_RulePop.txt')
            raise
        else:
            print("Writing Population as Data File...")

        #Write Header-----------------------------------------------------------------------------------------------------------------------------------------------
        dataLink = cons.env.format_data
        headList = dataLink.train_headers
        for i in range(len(headList)):
            rulePopOut.write(str(headList[i])+"\t")
        rulePopOut.write("Action\tPredict\tError\tFitness\tNumer\tGACount\tSubsumerCount\tActionSetSize\tTimeStampGA\tInitTimeStamp\tSpecificity\tDeletionProb\tActionCount\n")

        #sort classifiers based on numerosity-----------------------------------------------------------------------------------------------------------------------
        sorted_pop = sorted(pop.pop_set, key=lambda x: x.numerosity, reverse=True)
        #Write each classifier--------------------------------------------------------------------------------------------------------------------------------------
        for cl in sorted_pop:
            rulePopOut.write(str(cl.printClassifier()))

        rulePopOut.close()
