"""
Name:        xcs_algorithm.py
Authors:     Bao Trung
Contact:     baotrung@ecs.vuw.ac.nz
Created:     July, 2017
Description:
---------------------------------------------------------------------------------------------------------------------------------------------------------

---------------------------------------------------------------------------------------------------------------------------------------------------------
"""

#Import Required Modules-------------------------------
from xcs_constants import *
from xcs_classifierset import ClassifierSet
from xcs_prediction import *
from xcs_class_accuracy import ClassAccuracy
from xcs_outputfile_manager import OutputFileManager
import copy
import random
import math
from multiprocessing import Pool, cpu_count
#------------------------------------------------------

class XCS:
    def __init__(self):
        """ Initializes the XCS algorithm """
        print("XCS: Initializing Algorithm...")
        #Global Parameters-------------------------------------------------------------------------------------
        self.population = None          # The rule population (the 'solution/model' evolved by XCS)
        self.learn_track = None       # Output file that will store tracking information during learning
        self.pool = None
        if cons.multiprocessing:
            self.pool = Pool( processes=cpu_count()-1 )
        #-------------------------------------------------------
        # POPULATION REBOOT - Begin XCS learning from an existing saved rule population
        #-------------------------------------------------------
        if cons.do_pop_reboot:
            self.populationReboot()

        #-------------------------------------------------------
        # NORMAL XCS - Run XCS from scratch on given data
        #-------------------------------------------------------
        else:
            try:
                self.learn_track = open(cons.outFileName+'_LearnTrack.txt','w')
            except Exception as inst:
                print(type(inst))
                print(inst.args)
                print(inst)
                print('cannot open', cons.outFileName+'_LearnTrack.txt')
                raise
            else:
                self.learn_track.write("Explore_Iteration\tMacroPopSize\tMicroPopSize\tAccuracy_Estimate\tAveGenerality\tExpRules\tTime(min)\n")

            # Instantiate Population---------
            self.population = ClassifierSet()
            self.exploreIter = 0
            self.correct  = [0.0 for i in range(cons.trackingFrequency)]

        #Run the XCS algorithm-------------------------------------------------------------------------------
        self.run_XCS()


    def run_XCS(self):
        """ Runs the initialized XCS algorithm. """
        #--------------------------------------------------------------
        print("Learning Checkpoints: " +str(cons.learningCheckpoints))
        print("Maximum Iterations: " +str(cons.maxLearningIterations))
        print("Beginning XCS learning iterations.")
        print("------------------------------------------------------------------------------------------------------------------------------------------------------")

        #-------------------------------------------------------
        # MAJOR LEARNING LOOP
        #-------------------------------------------------------
        while self.exploreIter < cons.maxLearningIterations:
            #-------------------------------------------------------
            # GET NEW INSTANCE AND RUN A LEARNING ITERATION
            #-------------------------------------------------------
            state_phenotype = cons.env.getTrainInstance()
            self.runIteration(state_phenotype, self.exploreIter)

            #-------------------------------------------------------------------------------------------------------------------------------
            # EVALUATIONS OF ALGORITHM
            #-------------------------------------------------------------------------------------------------------------------------------
            cons.timer.startTimeEvaluation()

            #-------------------------------------------------------
            # TRACK LEARNING ESTIMATES
            #-------------------------------------------------------
            if (self.exploreIter%cons.trackingFrequency) == (cons.trackingFrequency - 1) and self.exploreIter > 0:
                self.population.runPopAveEval(self.exploreIter)
                trackedAccuracy = sum(self.correct)/float(cons.trackingFrequency) #Accuracy over the last "trackingFrequency" number of iterations.
                self.learn_track.write(self.population.getPopTrack(trackedAccuracy, self.exploreIter+1,cons.trackingFrequency)) #Report learning progress to standard out and tracking file.
            cons.timer.stopTimeEvaluation()

            #-------------------------------------------------------
            # CHECKPOINT - COMPLETE EVALUTATION OF POPULATION - strategy different for discrete vs continuous phenotypes
            #-------------------------------------------------------
            if (self.exploreIter + 1) in cons.learningCheckpoints:
                cons.timer.startTimeEvaluation()
                print("------------------------------------------------------------------------------------------------------------------------------------------------------")
                print("Running Population Evaluation after " + str(self.exploreIter + 1)+ " iterations.")

                self.population.runPopAveEval(self.exploreIter)
                self.population.runAttGeneralitySum(True)
                cons.env.startEvaluationMode()  #Preserves learning position in training data
                if cons.testFile != 'None': #If a testing file is available.
                    if cons.env.format_data.discrete_action:
                        trainEval = self.doPopEvaluation(True)
                        testEval = self.doPopEvaluation(False)
                    else:
                        trainEval = self.doContPopEvaluation(True)
                        testEval = self.doContPopEvaluation(False)
                else:  #Only a training file is available
                    if cons.env.format_data.discrete_action:
                        trainEval = self.doPopEvaluation(True)
                        testEval = None
                    else:
                        trainEval = self.doContPopEvaluation(True)
                        testEval = None

                cons.env.stopEvaluationMode() #Returns to learning position in training data
                cons.timer.stopTimeEvaluation()
                cons.timer.returnGlobalTimer()

                #Write output files----------------------------------------------------------------------------------------------------------
                OutputFileManager().writePopStats(cons.outFileName, trainEval, testEval, self.exploreIter + 1, self.population, self.correct)
                OutputFileManager().writePop(cons.outFileName, self.exploreIter + 1, self.population)
                #----------------------------------------------------------------------------------------------------------------------------

                print("Continue Learning...")
                print("------------------------------------------------------------------------------------------------------------------------------------------------------")

            #-------------------------------------------------------
            # ADJUST MAJOR VALUES FOR NEXT ITERATION
            #-------------------------------------------------------
            self.exploreIter += 1       # Increment current learning iteration
            cons.env.newInstance(True)  # Step to next instance in training set

        if cons.multiprocessing:
            self.pool.close()
        # Once XCS has reached the last learning iteration, close the tracking file
        self.learn_track.close()
        print("XCS Run Complete")


    def runIteration(self, state_phenotype, exploreIter):
        """ Run a single XCS learning iteration. """
        reward = 0
        #-----------------------------------------------------------------------------------------------------------------------------------------
        # FORM A MATCH SET - includes covering
        #-----------------------------------------------------------------------------------------------------------------------------------------
        self.population.makeMatchSet(state_phenotype[0], exploreIter, self.pool)
        if self.population.match_set == []:
            return
        #-----------------------------------------------------------------------------------------------------------------------------------------
        # MAKE A PREDICTION - utilized here for tracking estimated learning progress.  Typically used in the explore phase of many LCS algorithms.
        #-----------------------------------------------------------------------------------------------------------------------------------------
        cons.timer.startTimeEvaluation()
        prediction = Prediction(self.population)
        selectedPhenotype = prediction.getDecision()
        #-------------------------------------------------------
        # DISCRETE PHENOTYPE PREDICTION
        #-------------------------------------------------------
        if cons.env.format_data.discrete_action:
            if selectedPhenotype == state_phenotype[1]:
                reward = 1000
            if state_phenotype[1] in prediction.getHighestPredictionAction():
                self.correct[exploreIter%cons.trackingFrequency] = 1
            else:
                self.correct[exploreIter%cons.trackingFrequency] = 0
        #-------------------------------------------------------
        # CONTINUOUS PHENOTYPE PREDICTION
        #-------------------------------------------------------
        else:
            predictionError = math.fabs( selectedPhenotype - float( state_phenotype[ 1 ] ) )
            phenotypeRange = cons.env.format_data.action_list[ 1 ] - cons.env.format_data.action_list[ 0 ]
            accuracyEstimate = 1.0 - ( predictionError / float( phenotypeRange ) )
            self.correct[ exploreIter%cons.trackingFrequency ] = accuracyEstimate
            reward = 1000 * accuracyEstimate
        cons.timer.stopTimeEvaluation()
        #-----------------------------------------------------------------------------------------------------------------------------------------
        # FORM AN ACTION SET
        #-----------------------------------------------------------------------------------------------------------------------------------------
        #self.population.makeActionSet(state_phenotype[1])    # make Action Set for XCS
        self.population.makeActionSet( selectedPhenotype )
        #-----------------------------------------------------------------------------------------------------------------------------------------
        # UPDATE PARAMETERS
        #-----------------------------------------------------------------------------------------------------------------------------------------
        self.population.updateSets( exploreIter, reward )
        #-----------------------------------------------------------------------------------------------------------------------------------------
        # SUBSUMPTION - APPLIED TO MATCH SET - A heuristic for addition additional generalization pressure to XCS
        #-----------------------------------------------------------------------------------------------------------------------------------------
        if cons.do_subsumption:
            cons.timer.startTimeSubsumption()
            self.population.doActionSetSubsumption()
            cons.timer.stopTimeSubsumption()
        #-----------------------------------------------------------------------------------------------------------------------------------------
        # RUN THE GENETIC ALGORITHM - Discover new offspring rules from a selected pair of parents
        #-----------------------------------------------------------------------------------------------------------------------------------------
        self.population.runGA( exploreIter, state_phenotype[ 0 ] )
        #-----------------------------------------------------------------------------------------------------------------------------------------
        # SELECT RULES FOR DELETION - This is done whenever there are more rules in the population than 'N', the maximum population size.
        #-----------------------------------------------------------------------------------------------------------------------------------------
        #self.population.deletion(exploreIter)
        self.population.clearSets() #Clears the match and action sets for the next learning iteration


    def doPopEvaluation(self, isTrain):
        """ Performs a complete evaluation of the current rule population.  The population is unchanged throughout this evaluation. Works on both training and testing data. """
        if isTrain:
            myType = "TRAINING"
        else:
            myType = "TESTING"
        noMatch = 0                     # How often does the population fail to have a classifier that matches an instance in the data.
        tie = 0                         # How often can the algorithm not make a decision between classes due to a tie.
        cons.env.resetDataRef( isTrain ) # Go to the first instance in dataset
        phenotypeList = cons.env.format_data.action_list
        #----------------------------------------------
        classAccDict = {}
        for each in phenotypeList:
            classAccDict[each] = ClassAccuracy()
        #----------------------------------------------
        if isTrain:
            instances = cons.env.format_data.numb_train_instances
        else:
            instances = cons.env.format_data.numb_test_instances
        #----------------------------------------------------------------------------------------------
        for inst in range(instances):
            if isTrain:
                state_phenotype = cons.env.getTrainInstance()
            else:
                state_phenotype = cons.env.getTestInstance()
            #-----------------------------------------------------------------------------
            self.population.makeEvalMatchSet(state_phenotype[0])
            prediction = Prediction(self.population, True)
            phenotypeSelection = prediction.getDecision()
            #-----------------------------------------------------------------------------

            if phenotypeSelection == None:
                noMatch += 1
            elif phenotypeSelection == 'Tie':
                tie += 1
            else: #Instances which failed to be covered are excluded from the accuracy calculation
                for each in phenotypeList:
                    thisIsMe = False
                    accuratePhenotype = False
                    truePhenotype = state_phenotype[1]
                    if each == truePhenotype:
                        thisIsMe = True
                    if phenotypeSelection == truePhenotype:
                        accuratePhenotype = True
                    classAccDict[each].updateAccuracy(thisIsMe, accuratePhenotype)

            cons.env.newInstance(isTrain) # prepare next instance
            self.population.clearSets()
        #----------------------------------------------------------------------------------------------
        #Calculate Standard Accuracy--------------------------------------------
        instancesCorrectlyClassified = classAccDict[phenotypeList[0]].T_myClass + classAccDict[phenotypeList[0]].T_otherClass
        instancesIncorrectlyClassified = classAccDict[phenotypeList[0]].F_myClass + classAccDict[phenotypeList[0]].F_otherClass
        standardAccuracy = float(instancesCorrectlyClassified) / float(instancesCorrectlyClassified + instancesIncorrectlyClassified)

        #Calculate Balanced Accuracy---------------------------------------------
        T_mySum = 0
        T_otherSum = 0
        F_mySum = 0
        F_otherSum = 0
        for each in phenotypeList:
            T_mySum += classAccDict[each].T_myClass
            T_otherSum += classAccDict[each].T_otherClass
            F_mySum += classAccDict[each].F_myClass
            F_otherSum += classAccDict[each].F_otherClass
        balancedAccuracy = ((0.5*T_mySum / (float(T_mySum + F_otherSum)) + 0.5*T_otherSum / (float(T_otherSum + F_mySum)))) # BalancedAccuracy = (Specificity + Sensitivity)/2

        #Adjustment for uncovered instances - to avoid positive or negative bias we incorporate the probability of guessing a phenotype by chance (e.g. 50% if two phenotypes)
        predictionFail = float(noMatch)/float(instances)
        predictionTies = float(tie)/float(instances)
        instanceCoverage = 1.0 - predictionFail
        predictionMade = 1.0 - (predictionFail + predictionTies)

        adjustedStandardAccuracy = (standardAccuracy * predictionMade) + ((1.0 - predictionMade) * (1.0 / float(len(phenotypeList))))
        adjustedBalancedAccuracy = (balancedAccuracy * predictionMade) + ((1.0 - predictionMade) * (1.0 / float(len(phenotypeList))))

        #Adjusted Balanced Accuracy is calculated such that instances that did not match have a consistent probability of being correctly classified in the reported accuracy.
        print("-----------------------------------------------")
        print(str(myType)+" Accuracy Results:-------------")
        print("Instance Coverage = "+ str(instanceCoverage*100.0)+ '%')
        print("Prediction Ties = "+ str(predictionTies*100.0)+ '%')
        print(str(instancesCorrectlyClassified) + ' out of ' + str(instances) + ' instances covered and correctly classified.')
        print("Standard Accuracy (Adjusted) = " + str(adjustedStandardAccuracy))
        print("Balanced Accuracy (Adjusted) = " + str(adjustedBalancedAccuracy))
        #Balanced and Standard Accuracies will only be the same when there are equal instances representative of each phenotype AND there is 100% covering.
        resultList = [adjustedBalancedAccuracy, instanceCoverage]
        return resultList


    def doContPopEvaluation(self, isTrain):
        """ Performs evaluation of population via the copied environment. Specifically developed for continuous phenotype evaulation.
        The population is maintained unchanging throughout the evaluation.  Works on both training and testing data. """
        if isTrain:
            myType = "TRAINING"
        else:
            myType = "TESTING"
        noMatch = 0 #How often does the population fail to have a classifier that matches an instance in the data.
        cons.env.resetDataRef(isTrain) #Go to first instance in data set
        accuracyEstimateSum = 0

        if isTrain:
            instances = cons.env.format_data.numb_train_instances
        else:
            instances = cons.env.format_data.numb_test_instances
        #----------------------------------------------------------------------------------------------
        for inst in range(instances):
            if isTrain:
                state_phenotype = cons.env.getTrainInstance()
            else:
                state_phenotype = cons.env.getTestInstance()
            #-----------------------------------------------------------------------------
            self.population.makeEvalMatchSet(state_phenotype[0])
            prediction = Prediction(self.population)
            phenotypePrediction = prediction.getDecision()
            #-----------------------------------------------------------------------------
            if phenotypePrediction == None:
                noMatch += 1
            else: #Instances which failed to be covered are excluded from the initial accuracy calculation
                predictionError = math.fabs(float(phenotypePrediction) - float(state_phenotype[1]))
                phenotypeRange = cons.env.format_data.action_list[1] - cons.env.format_data.action_list[0]
                accuracyEstimateSum += 1.0 - (predictionError / float(phenotypeRange))

            cons.env.newInstance(isTrain) #next instance
            self.population.clearSets()
        #----------------------------------------------------------------------------------------------
        #Accuracy Estimate
        if instances == noMatch:
            accuracyEstimate = 0
        else:
            accuracyEstimate = accuracyEstimateSum / float(instances - noMatch)

        #Adjustment for uncovered instances - to avoid positive or negative bias we incorporate the probability of guessing a phenotype by chance (e.g. 50% if two phenotypes)
        instanceCoverage = 1.0 - (float(noMatch)/float(instances))
        adjustedAccuracyEstimate = accuracyEstimateSum / float(instances) #noMatchs are treated as incorrect predictions (can see no other fair way to do this)

        print("-----------------------------------------------")
        print(str(myType)+" Accuracy Results:-------------")
        print("Instance Coverage = "+ str(instanceCoverage*100.0)+ '%')
        print("Estimated Prediction Accuracy (Ignore uncovered) = " + str(accuracyEstimate))
        print("Estimated Prediction Accuracy (Penalty uncovered) = " + str(adjustedAccuracyEstimate))
        #Balanced and Standard Accuracies will only be the same when there are equal instances representative of each phenotype AND there is 100% covering.
        resultList = [adjustedAccuracyEstimate, instanceCoverage]
        return resultList


    def populationReboot(self):
        """ Manages the reformation of a previously saved XCS classifier population. """
        cons.timer.setTimerRestart(cons.popRebootPath) #Rebuild timer objects
        #--------------------------------------------------------------------
        try: #Re-open track learning file for continued tracking of progress.
            self.learn_track = open(cons.outFileName+'_LearnTrack.txt','a')
        except Exception as inst:
            print(type(inst))
            print(inst.args)
            print(inst)
            print('cannot open', cons.outFileName+'_LearnTrack.txt')
            raise

        #Extract last iteration from file name---------------------------------------------
        temp = cons.popRebootPath.split('_')
        iterRef = len(temp)-1
        completedIterations = int(temp[iterRef])
        print("Rebooting rule population after " +str(completedIterations)+ " iterations.")
        self.exploreIter = completedIterations-1
        for i in range(len(cons.learningCheckpoints)):
            cons.learningCheckpoints[i] += completedIterations
        cons.maxLearningIterations += completedIterations

        #Rebuild existing population from text file.--------
        self.population = ClassifierSet(cons.popRebootPath)
        #---------------------------------------------------
        try: #Obtain correct track
            f = open(cons.popRebootPath+"_PopStats.txt", 'r')
        except Exception as inst:
            print(type(inst))
            print(inst.args)
            print(inst)
            print('cannot open', cons.popRebootPath+"_PopStats.txt")
            raise
        else:
            correctRef = 26 #File reference position
            tempLine = None
            for i in range(correctRef):
                tempLine = f.readline()
            tempList = tempLine.strip().split('\t')
            self.correct = tempList
            if cons.env.format_data.discrete_action:
                for i in range(len(self.correct)):
                    self.correct[i] = int(self.correct[i])
            else:
                for i in range(len(self.correct)):
                    self.correct[i] = float(self.correct[i])
            f.close()
