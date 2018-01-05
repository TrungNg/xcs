"""
Name:        xcs_classifierset.py
Authors:     Bao Trung - Written at Victoria University of Wellington, New Zealand
Contact:     baotrung@ecs.vuw.ac.nz
Created:     July, 2017
Description:
---------------------------------------------------------------------------------------------------------------------------------------------------------
XCS: Michigan-style Learning Classifier System - A LCS for Reinforcement Learning.  This XCS follows the version descibed in "An Algorithmic Description of XCS" published by Martin Butz and Stewart Wilson (2002).

---------------------------------------------------------------------------------------------------------------------------------------------------------
"""

#Import Required Modules---------------------
from xcs_constants import *
from xcs_classifier import Classifier
import random
import copy
import sys
#--------------------------------------------

class ClassifierSet:
    def __init__(self, a=None):
        """ Overloaded initialization: Handles creation of a new population or a rebooted population (i.e. a previously saved population). """
        # Major Parameters
        self.pop_set = []           # List of classifiers/rules
        self.match_set = []         # List of references to rules in population that match
        self.action_set = []        # List of references to rules in population that match and has action with highest prediction payoff
        self.prev_action_set = []   # List of references to rules in population that was in action set in previous time step
        self.micro_pop_size = 0     # Tracks the current micro population size, i.e. the population size which takes rule numerosity into account.
        self.covered_cl_counter = 0             # Track number of classifiers created by covering
        self.crossovered_cl_counter = 0         # Track number of classifiers created by GA
        self.mutated_cl_counter = 0             # Track number of classifiers created by mutation

        # Evaluation Parameters-------------------------------
        self.average_generality = 0.0
        self.attribute_spec_list = []
        self.attribute_pred_list = []

        # Set Constructors-------------------------------------
        if a==None:
            self.makePop() #Initialize a new population
        elif isinstance(a,str):
            self.rebootPop(a) #Initialize a population based on an existing saved rule population
        else:
            print("ClassifierSet: Error building population.")

    #--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    # POPULATION CONSTRUCTOR METHODS
    #--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    def makePop(self):
        """ Initializes the rule population """
        self.pop_set = []


    def rebootPop(self, remake_file):
        """ Remakes a previously evolved population from a saved text file. """
        print("Rebooting the following population: " + str(remake_file))
        #*******************Initial file handling**********************************************************
        dataset_list = []
        try:
            f = open(remake_file, 'r')
        except Exception as inst:
            print(type(inst))
            print(inst.args)
            print(inst)
            print('cannot open', remake_file)
            raise
        else:
            _ = f.readline().rstrip('\n').split('\t')   #strip off first row
            for line in f:
                line_list = line.strip('\n').split('\t')
                dataset_list.append(line_list)
            f.close()

        #**************************************************************************************************
        for each in dataset_list:
            cl = Classifier(each)
            self.pop_set.append(cl)
            numerosity_ref = cons.env.number_of_attributes + 3
            self.micro_pop_size += int(each[numerosity_ref])
        print("Rebooted Rule Population has "+str(len(self.pop_set))+" Macro Pop Size.")


    #--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    # CLASSIFIER SET CONSTRUCTOR METHODS
    #--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    def makeMatchSet(self, state, explore_iter):
        """ Constructs a match set from the population. Covering is initiated if the match set is empty or total prediction of rules in match set is too low. """
        #Initial values
        do_covering = True # Covering check: Twofold (1)checks that a match is present, and (2) that total Prediction in Match Set is greater than a threshold compared to mean preadiction.
        match_set_phenotypes = []
        #-------------------------------------------------------
        # MATCHING
        #-------------------------------------------------------
        cons.timer.startTimeMatching()
        for i in range( len( self.pop_set ) ):                  # Go through the population
            cl = self.pop_set[i]                                # One classifier at a time
            if cl.match(state):                                 # Check for match
                self.match_set.append( i )                      # If match - add classifier to match set
                if cl.action not in match_set_phenotypes:
                    match_set_phenotypes.append( cl.action )
        cons.timer.stopTimeMatching()
        if len( match_set_phenotypes ) >= cons.theta_mna:
            do_covering = False
        #-------------------------------------------------------
        # COVERING
        #-------------------------------------------------------
        while do_covering:
            new_cl = Classifier(explore_iter, state, random.choice(list(set(cons.env.action_list) - set(match_set_phenotypes))))
            new_cl.origin = 'covered'
            self.addClassifierToPopulation( new_cl, do_covering )
            self.covered_cl_counter += 1
            self.match_set.append( len(self.pop_set)-1 )        # Add covered classifier to match set
            match_set_phenotypes.append( new_cl.action )
            if len( match_set_phenotypes ) >= cons.theta_mna:
                # if there is sufficient number of different phenotypes in match set, stop covering
                self.deletion()
                self.match_set = []
                do_covering = False


    def makeActionSet(self, chosen_action):
        """ Constructs a correct set out of the given match set. """
        for i in range(len(self.match_set)):
            ref = self.match_set[i]
            if self.pop_set[ref].action == chosen_action:
                self.action_set.append(ref)


    def makeEvalMatchSet(self, state):
        """ Constructs a match set for evaluation purposes which does not activate either covering or deletion. """
        for i in range(len(self.pop_set)):       # Go through the population
            cl = self.pop_set[i]                 # A single classifier
            if cl.match(state):                 # Check for match
                self.match_set.append(i)         # Add classifier to match set


    #--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    # CLASSIFIER DELETION METHODS
    #--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    def deletion(self):
        """ Returns the population size back to the maximum set by the user by deleting rules. """
        cons.timer.startTimeDeletion()
        while self.micro_pop_size > cons.N:
            self.deleteFromPopulation()
        cons.timer.stopTimeDeletion()


    def deleteFromPopulation(self):
        """ Deletes one classifier in the population.  The classifier that will be deleted is chosen by roulette wheel selection
        considering the deletion vote. Returns the macro-classifier which got decreased by one micro-classifier. """
        mean_fitness = self.getPopFitnessSum()/float(self.micro_pop_size)

        #Calculate total wheel size------------------------------
        sum_cl = 0.0
        vote_list = []
        for cl in self.pop_set:
            vote = cl.getDelProb(mean_fitness)
            sum_cl += vote
            vote_list.append(vote)
        #--------------------------------------------------------
        choice_point = sum_cl * random.random() #Determine the choice point

        new_sum=0.0
        for i in range(len(vote_list)):
            cl = self.pop_set[i]
            new_sum = new_sum + vote_list[i]
            if new_sum > choice_point: #Select classifier for deletion
                #Delete classifier----------------------------------
                cl.updateNumerosity(-1)
                self.micro_pop_size -= 1
                if cl.numerosity < 1: # When all micro-classifiers for a given classifier have been depleted.
                    self.removeMacroClassifier(i)
                    self.deleteFromMatchSet(i)
                    self.deleteFromActionSet(i)
                    self.deleteFromPrevActionSet(i)
                return

        print("ClassifierSet: No eligible rules found for deletion in deleteFromPopulation.")
        return


    def removeMacroClassifier(self, ref):
        """ Removes the specified (macro-) classifier from the population. """
        self.pop_set.pop(ref)


    def deleteFromMatchSet(self, delete_ref):
        """ Delete reference to classifier in population, contained in self.match_set."""
        if delete_ref in self.match_set:
            self.match_set.remove(delete_ref)

        #Update match set reference list--------
        for j in range(len(self.match_set)):
            ref = self.match_set[j]
            if ref > delete_ref:
                self.match_set[j] -= 1


    def deleteFromPrevActionSet(self, delete_ref):
        """ Delete reference to classifier in population, contained in self.action_set."""
        if delete_ref in self.prev_action_set:
            self.prev_action_set.remove(delete_ref)

        #Update match set reference list--------
        for j in range(len(self.prev_action_set)):
            ref = self.prev_action_set[j]
            if ref > delete_ref:
                self.prev_action_set[j] -= 1


    def deleteFromActionSet(self, delete_ref):
        """ Delete reference to classifier in population, contained in self.action_set."""
        if delete_ref in self.action_set:
            self.action_set.remove(delete_ref)

        #Update match set reference list--------
        for j in range(len(self.action_set)):
            ref = self.action_set[j]
            if ref > delete_ref:
                self.action_set[j] -= 1


    #--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    # GENETIC ALGORITHM
    #--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    def runGA(self, explore_iter, action_set, state):
        """ The genetic discovery mechanism in XCS is controlled here. """
        #-------------------------------------------------------
        # GA RUN REQUIREMENT
        #-------------------------------------------------------
        if (explore_iter - self.getIterStampAverage()) < cons.theta_GA:  #Does the action set meet the requirements for activating the GA?
            return

        self.setIterStamps(explore_iter) #Updates the iteration time stamp for all rules in the match set (which the GA opperates in).
        changed = False

        #-------------------------------------------------------
        # SELECT PARENTS - Niche GA - selects parents from the match set
        #-------------------------------------------------------
        cons.timer.startTimeSelection()
        if cons.selection_method == "roulette":
            selected_cl = self.selectClassifierRW( action_set )
            clP1 = selected_cl[0]
            clP2 = selected_cl[1]
        elif cons.selection_method == "tournament":
            selected_cl = self.selectClassifierT( action_set )
            clP1 = selected_cl[0]
            clP2 = selected_cl[1]
        else:
            print("ClassifierSet: Error - requested GA selection method not available.")
        cons.timer.stopTimeSelection()
        clP1.updateGACount()
        clP2.updateGACount()
        #-------------------------------------------------------
        # INITIALIZE OFFSPRING
        #-------------------------------------------------------
        cl1  = Classifier(clP1, explore_iter)
        if clP2 == None:
            cl2 = Classifier(clP1, explore_iter)
        else:
            cl2 = Classifier(clP2, explore_iter)

        #-------------------------------------------------------
        # CROSSOVER OPERATOR - Uniform Crossover Implemented (i.e. all attributes have equal probability of crossing over between two parents)
        #-------------------------------------------------------
        if not cl1.equals(cl2) and random.random() < cons.chi:
            if cons.crossover_method == 'uniform':
                changed = cl1.uniformCrossover(cl2)
            elif cons.crossover_method == 'twopoint':
                changed = cl1.twoPointCrossover(cl2)

        #-------------------------------------------------------
        # INITIALIZE KEY OFFSPRING PARAMETERS
        #-------------------------------------------------------
        if changed:
            cl1.setPrediction((cl1.prediction + cl2.prediction)/2)
            cl1.setError((cl1.error + cl2.error)/2.0)
            cl1.setFitness(cons.fitness_reduction * (cl1.fitness + cl2.fitness)/2.0)
            cl2.setPrediction(cl1.prediction)
            cl2.setError(cl1.error)
            cl2.setFitness(cl1.fitness)

        cl1.setFitness(cons.fitness_reduction * cl1.fitness)
        cl2.setFitness(cons.fitness_reduction * cl2.fitness)
        #-------------------------------------------------------
        # MUTATION OPERATOR
        #-------------------------------------------------------
        nowchanged = cl1.Mutation(state)
        howaboutnow = cl2.Mutation(state)
        #-------------------------------------------------------
        # ADD OFFSPRING TO POPULATION
        #-------------------------------------------------------
        if changed:
            cl1.origin = 'crossovered'
            cl2.origin = 'crossovered'
        if nowchanged:
            cl1.origin = 'mutated'
        if howaboutnow:
            cl2.origin = 'mutated'
        if changed or nowchanged or howaboutnow:
            self.insertDiscoveredClassifiers(cl1, cl2, clP1, clP2) #Subsumption
        self.deletion()


    #--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    # SELECTION METHODS
    #--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    def selectClassifierRW(self, action_set):
        """ Selects parents using roulette wheel selection according to the fitness of the classifiers. """
        #Prepare for actionSet set or 'niche' selection.
        set_list = copy.deepcopy(action_set)

        if len(set_list) > 2:
            selected_cl = [None, None]
            count = 0 #Pick two parents
            #-----------------------------------------------
            while count < 2:
                fitness_sum = self.getFitnessSum(set_list)

                choice_point = random.random() * fitness_sum
                i=0
                sum_cl = self.pop_set[set_list[i]].fitness
                while choice_point > sum_cl:
                    i=i+1
                    sum_cl += self.pop_set[set_list[i]].fitness

                selected_cl[count] = self.pop_set[set_list[i]]
                set_list.remove(set_list[i])
                count += 1
            #-----------------------------------------------
        elif len(set_list) == 2:
            selected_cl = [self.pop_set[set_list[0]],self.pop_set[set_list[1]]]
        elif len(set_list) == 1:
            selected_cl = [self.pop_set[set_list[0]],self.pop_set[set_list[0]]]
        else:
            print("ClassifierSet: Error in parent selection.")

        return selected_cl


    def selectClassifierT(self, action_set):
        """  Selects parents using tournament selection according to the fitness of the classifiers. """
        selected_cl = [None, None]
        count = 0
        set_list = action_set #actionSet set is a list of reference IDs

        while count < 2:
            tournament_size = int(len( set_list ) * cons.theta_sel)
            tournament_list = random.sample(set_list, tournament_size)

            highest_fitness = 0
            best_cl = action_set[0]
            for j in tournament_list:
                if self.pop_set[j].fitness > highest_fitness:
                    highest_fitness = self.pop_set[j].fitness
                    best_cl = j

            selected_cl[count] = self.pop_set[best_cl]
            count += 1

        return selected_cl


    #--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    # SUBSUMPTION METHODS
    #--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    def subsumeClassifier(self, cl=None, cl1P=None, cl2P=None):
        """ Tries to subsume a classifier in the parents. If no subsumption is possible it tries to subsume it in the current set. """
        if cl1P!=None and cl1P.subsumes(cl):
            self.micro_pop_size += 1
            cl1P.updateNumerosity(1)
        elif cl2P!=None and cl2P.subsumes(cl):
            self.micro_pop_size += 1
            cl2P.updateNumerosity(1)
        else:
            self.addClassifierToPopulation(cl)
            #self.subsumeClassifier2(cl); #Try to subsume in the match set.


    def subsumeClassifier2(self, cl):
        """ Tries to subsume a classifier in the match set. If no subsumption is possible the classifier is simply added to the population considering
        the possibility that there exists an identical classifier. """
        choices = []
        for ref in self.match_set:
            if self.pop_set[ref].subsumes(cl):
                choices.append(ref)

        if len(choices) > 0: #Randomly pick one classifier to be subsumer
            choice = int(random.random()*len(choices))
            self.pop_set[choices[choice]].updateNumerosity(1)
            self.micro_pop_size += 1
            return

        self.addClassifierToPopulation(cl) #If no subsumer was found, check for identical classifier, if not then add the classifier to the population


    def doActionSetSubsumption(self, action_set):
        """ Executes match set subsumption.  The match set subsumption looks for the most general subsumer classifier in the match set
        and subsumes all classifiers that are more specific than the selected one. """
        subsumer = None
        for ref in action_set:
            cl = self.pop_set[ref]
            if cl.isPossibleSubsumer():
                if subsumer == None or len( subsumer.specified_att_list ) > len( cl.specified_att_list ) or ( ( len(subsumer.specified_att_list ) == len(cl.specified_att_list) and random.random() < 0.5 ) ):
                    subsumer = cl

        if subsumer != None: #If a subsumer was found, subsume all more specific classifiers in the match set
            i=0
            while i < len(action_set):
                ref = action_set[i]
                if subsumer.isMoreGeneral(self.pop_set[ref]):
                    subsumer.updateNumerosity(self.pop_set[ref].numerosity)
                    self.removeMacroClassifier(ref)
                    self.deleteFromMatchSet(ref)
                    self.deleteFromActionSet(ref)
                    self.deleteFromPrevActionSet(ref)
                    i = i - 1
                i = i + 1


    #--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    # OTHER KEY METHODS
    #--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    def addClassifierToPopulation(self, cl, covering = False):
        """ Adds a classifier to the set and increases the micro_pop_size value accordingly."""
        old_cl = None
        if not covering:
            old_cl = self.getIdenticalClassifier(cl)
        if old_cl != None: #found identical classifier
            old_cl.updateNumerosity(1)
        else:
            self.pop_set.append(cl)
        self.micro_pop_size += 1


    def insertDiscoveredClassifiers(self, cl1, cl2, clP1, clP2):
        """ Inserts both discovered classifiers and activates GA subsumption if turned on. Also checks for default rule (i.e. rule with completely general condition) and
        prevents such rules from being added to the population, as it offers no predictive value within XCS. """
        #-------------------------------------------------------
        # SUBSUMPTION
        #-------------------------------------------------------
        if cons.do_subsumption:
            cons.timer.startTimeSubsumption()

            if len(cl1.specified_att_list) > 0:
                self.subsumeClassifier(cl1, clP1, clP2)
            if len(cl2.specified_att_list) > 0:
                self.subsumeClassifier(cl2, clP1, clP2)

            cons.timer.stopTimeSubsumption()
        #-------------------------------------------------------
        # ADD OFFSPRING TO POPULATION
        #-------------------------------------------------------
        else: #Just add the new classifiers to the population.
            if len(cl1.specified_att_list) > 0:
                self.addClassifierToPopulation(cl1) #False passed because this is not called for a covered rule.
            if len(cl2.specified_att_list) > 0:
                self.addClassifierToPopulation(cl2) #False passed because this is not called for a covered rule.


    def updateSets(self, reward, max_prediction, action_set):
        """ Updates all relevant parameters in the current match and match sets. """
        actionset_numerosity = 0
        for ref in action_set:
            actionset_numerosity += self.pop_set[ref].numerosity
        accuracy_sum = 0.0
        for ref in action_set:
            self.pop_set[ref].updateActionExp()
            self.pop_set[ref].updateActionSetSize( actionset_numerosity )
            self.pop_set[ref].updateXCSParameters( reward, max_prediction )
            accuracy_sum += self.pop_set[ref].accuracy * self.pop_set[ref].numerosity
        for ref in action_set:
            self.pop_set[ref].setAccuracy( 1000 * self.pop_set[ref].accuracy * self.pop_set[ref].numerosity / accuracy_sum )
            self.pop_set[ref].updateFitness()


    #--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    # OTHER METHODS
    #--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    def getIterStampAverage(self):
        """ Returns the average of the time stamps in the match set. """
        sum_cl=0.0
        sum_numer=0.0
        for i in range(len(self.action_set)):
            ref = self.action_set[i]
            sum_cl += self.pop_set[ref].timestamp_GA * self.pop_set[ref].numerosity
            sum_numer += self.pop_set[ref].numerosity #numerosity sum of match set
        return sum_cl/float(sum_numer)


    def setIterStamps(self, explore_iter):
        """ Sets the time stamp of all classifiers in the set to the current time. The current time
        is the number of exploration steps executed so far.  """
        for i in range(len(self.action_set)):
            ref = self.action_set[i]
            self.pop_set[ref].updateTimeStamp(explore_iter)


    def getFitnessSum(self, set_list):
        """ Returns the sum of the fitnesses of all classifiers in the set. """
        sum_cl=0.0
        for i in range(len(set_list)):
            ref = set_list[i]
            sum_cl += self.pop_set[ref].fitness
        return sum_cl


    def getPopFitnessSum(self):
        """ Returns the sum of the fitnesses of all classifiers in the set. """
        sum_cl = 0.0
        for cl in self.pop_set:
            sum_cl += cl.fitness
        return sum_cl


    def getIdenticalClassifier(self, new_cl):
        """ Looks for an identical classifier in the population. """
        for cl in self.pop_set:
            if new_cl.equals( cl ):
                return cl
        return None


    def clearSets(self, is_end_problem = False):
        """ Clears out references in the match and action sets for the next learning iteration. """
        self.match_set = []
        if is_end_problem == False:
            self.prev_action_set = copy.deepcopy(self.action_set)
        else:
            self.prev_action_set = []
        self.action_set = []

    #--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    # EVALUTATION METHODS
    #--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    def runPopAveEval(self):
        """ Calculates some summary evaluations across the rule population including average generality. """
        generality_sum = 0
        self.covered_cl_counter = 0
        self.crossovered_cl_counter = 0
        self.mutated_cl_counter = 0
        #agedCount = 0
        for cl in self.pop_set:
            # Count the accurately predicted classifiers' origins
            if cl.origin == 'covered' and cl.error < 10:
                self.covered_cl_counter += 1
            if cl.origin == 'crossovered' and cl.error < 10:
                self.crossovered_cl_counter += 1
            if cl.origin == 'mutated' and cl.error < 10:
                self.mutated_cl_counter += 1
            generality_sum += (cons.env.number_of_attributes - len(cl.condition)) * cl.numerosity
        if self.micro_pop_size == 0:
            self.average_generality = 'NA'
        else:
            self.average_generality = generality_sum / float( self.micro_pop_size * cons.env.number_of_attributes )


    def runAttGeneralitySum(self, is_summary):
        """ Determine the population-wide frequency of attribute specification, and prediction weighted specification.  Used in complete rule population evaluations. """
        if is_summary:
            self.attribute_spec_list = []
            self.attribute_pred_list = []
            for _ in range(cons.env.number_of_attributes):
                self.attribute_spec_list.append(0)
                self.attribute_pred_list.append(0.0)
            for cl in self.pop_set:
                for ref in cl.specified_att_list: #for each attRef
                    self.attribute_spec_list[ref] += cl.numerosity
                    self.attribute_pred_list[ref] += cl.numerosity * cl.prediction

