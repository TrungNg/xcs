"""
Name:        xcs_classifier.py
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
import random
import copy
import math
#--------------------------------------

class Classifier:
    def __init__(self,a=None,b=None,c=None):
        #Major Parameters --------------------------------------------------
        self.specified_attributes = []      # Attribute Specified in classifier: Similar to Bacardit 2009 - ALKR + GABIL, continuous and discrete rule representation
        self.condition = []             # States of Attributes Specified in classifier: Similar to Bacardit 2009 - ALKR + GABIL, continuous and discrete rule representation
        self.action = None           # Class if the endpoint is discrete, and a continuous phenotype if the endpoint is continuous

        self.prediction = cons.init_pred  # Classifier payoff - initialized to a constant initial payoff value
        self.error = cons.init_err      # Classifier error - initialized to a constant initial error value
        self.fitness = cons.init_fit    # Classifier fitness - initialized to a constant initial fitness value
        self.accuracy = 0.0             # Classifier accuracy - Accuracy calculated using only instances in the dataset which this rule matched.
        self.numerosity = 1             # The number of rule copies stored in the population.  (Indirectly stored as incremented numerosity)
        self.mean_actionset_sz = 0       # A parameter used in deletion which reflects the size of match sets within this rule has been included.
        self.delete_vote = None        # The current deletion weight for this classifier.

        #Experience Management ---------------------------------------------
        self.ga_timestamp = None         # Time since rule last in a match set.
        self.init_timestamp = None       # Iteration in which the rule first appeared.

        #Classifier Accuracy Tracking --------------------------------------
        self.ga_count = 0
        self.subsumer_cnt = 0
        #self.matchCount = 0             # Known in many LCS implementations as experience i.e. the total number of times this classifier was in a match set
        self.actionCount = 0            # The total number of times this classifier was chosen in action set

        if isinstance(b,list):
            self.classifierCovering(a,b,c)
        elif isinstance(a,Classifier):
            self.classifierCopy(a, b)
        elif isinstance(a,list) and b == None:
            self.rebootClassifier(a)
        else:
            print("Classifier: Error building classifier.")

    #--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    # CLASSIFIER CONSTRUCTION METHODS
    #--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    def classifierCovering(self, exploreIter, state, phenotype):
        """ Makes a new classifier when the covering mechanism is triggered.  The new classifier will match the current training instance.
        Covering will NOT produce a default rule (i.e. a rule with a completely general condition). """
        #Initialize new classifier parameters----------
        self.ga_timestamp = exploreIter
        self.init_timestamp = exploreIter
        #self.aveMatchSetSize = setSize
        data_info = cons.env.format_data
        #-------------------------------------------------------
        # DISCRETE PHENOTYPE
        #-------------------------------------------------------
        if data_info.discrete_action:
            if phenotype != None:
                self.action = phenotype
            else:
                self.action = random.choice(cons.env.format_data.action_list)
        #-------------------------------------------------------
        # CONTINUOUS PHENOTYPE
        #-------------------------------------------------------
        else:
            phenotypeRange = data_info.action_list[1] - data_info.action_list[0]
            rangeRadius = random.randint(25,75)*0.01*phenotypeRange / 2.0 #Continuous initialization domain radius.
            if phenotype == None:
                phenotype = data_info.action_list[0] + rangeRadius + random.random() * ( phenotypeRange - 2 * rangeRadius )
            Low = float(phenotype) - rangeRadius
            High = float(phenotype) + rangeRadius
            self.action = [Low,High] #ALKR Representation, Initialization centered around training instance  with a range between 25 and 75% of the domain size.
        #-------------------------------------------------------
        # GENERATE MATCHING CONDITION
        #-------------------------------------------------------
        while len(self.specified_attributes) < 1:
            for attRef in range(len(state)):
                if random.random() < cons.p_spec and state[attRef] != cons.labelMissingData:
                    self.specified_attributes.append(attRef)
                    self.condition.append(self.buildMatch(attRef, state))


    def classifierCopy(self, clOld, exploreIter):
        """  Constructs an identical Classifier.  However, the experience of the copy is set to 0 and the numerosity
        is set to 1 since this is indeed a new individual in a population. Used by the genetic algorithm to generate
        offspring based on parent classifiers."""
        self.specified_attributes = copy.deepcopy(clOld.specified_attributes)
        self.condition = copy.deepcopy(clOld.condition)
        self.action = copy.deepcopy(clOld.action)
        self.ga_timestamp = exploreIter
        self.init_timestamp = exploreIter
        #self.aveMatchSetSize = copy.deepcopy(clOld.aveMatchSetSize)
        self.prediction = clOld.prediction
        self.error = clOld.error
        self.fitness = clOld.fitness
        #self.accuracy = clOld.accuracy


    def rebootClassifier(self, classifierList):
        """ Rebuilds a saved classifier as part of the population Reboot """
        numAttributes = cons.env.format_data.numb_attributes
        attInfo = cons.env.format_data.attribute_info
        for attRef in range(0,numAttributes):
            if classifierList[attRef] != '#':  #Attribute in rule is not wild
                if attInfo[attRef][0]: #Continuous Attribute
                    valueRange = classifierList[attRef].split(';')
                    self.condition.append(valueRange)
                    self.specified_attributes.append(attRef)
                else:
                    self.condition.append(classifierList[attRef])
                    self.specified_attributes.append(attRef)
        #-------------------------------------------------------
        # DISCRETE PHENOTYPE
        #-------------------------------------------------------
        if cons.env.format_data.discrete_action:
            self.action = str(classifierList[numAttributes])
        #-------------------------------------------------------
        # CONTINUOUS PHENOTYPE
        #-------------------------------------------------------
        else:
            self.action = classifierList[numAttributes].split(';')
            for i in range(2):
                self.action[i] = float(self.action[i])

        self.prediction = float(classifierList[numAttributes+1])
        self.error = float(classifierList[numAttributes+2])
        self.fitness = float(classifierList[numAttributes+3])
        self.numerosity = int(classifierList[numAttributes+4])
        self.ga_count = float(classifierList[numAttributes+5])
        self.subsumer_cnt = float(classifierList[numAttributes+6])
        self.aveMatchSetSize = float(classifierList[numAttributes+7])
        self.ga_timestamp = int(classifierList[numAttributes+8])
        self.init_timestamp = int(classifierList[numAttributes+9])

        self.delete_vote = float(classifierList[numAttributes+11])
        self.actionCount = int(classifierList[numAttributes+12])


    #--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    # MATCHING
    #--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    def match(self, state):
        """ Returns if the classifier matches in the current situation. """
        for i in range(len(self.condition)):
            attributeInfo = cons.env.format_data.attribute_info[self.specified_attributes[i]]
            #-------------------------------------------------------
            # CONTINUOUS ATTRIBUTE
            #-------------------------------------------------------
            if attributeInfo[0]:
                instanceValue = state[self.specified_attributes[i]]
                if self.condition[i][0] < instanceValue < self.condition[i][1] or instanceValue == cons.labelMissingData:
                    pass
                else:
                    return False
            #-------------------------------------------------------
            # DISCRETE ATTRIBUTE
            #-------------------------------------------------------
            else:
                stateRep = state[self.specified_attributes[i]]
                if stateRep == self.condition[i] or stateRep == cons.labelMissingData:
                    pass
                else:
                    return False
        return True


    #--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    # GENETIC ALGORITHM MECHANISMS
    #--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    def uniformCrossover(self, cl):
        """ Applies uniform crossover and returns if the classifiers changed. Handles both discrete and continuous attributes.
        #SWARTZ: self. is where for the better attributes are more likely to be specified
        #DEVITO: cl. is where less useful attribute are more likely to be specified
        """
        if cons.env.format_data.discrete_action: #Always crossover condition if the phenotype is discrete (if continuous phenotype, half the time phenotype crossover is performed instead)
            p_self_specifiedAttList = copy.deepcopy(self.specified_attributes)
            p_cl_specifiedAttList = copy.deepcopy(cl.specified_attributes)
            probability = 0.5  #Equal probability for attribute alleles to be exchanged.

            #Make list of attribute references appearing in at least one of the parents.-----------------------------
            comboAttList = []
            for i in p_self_specifiedAttList:
                comboAttList.append(i)
            for i in p_cl_specifiedAttList:
                if i not in comboAttList:
                    comboAttList.append(i)
                elif not cons.env.format_data.attribute_info[i][0]: #Attribute specified in both parents, and the attribute is discrete (then no reason to cross over)
                    comboAttList.remove(i)
            comboAttList.sort()
            #--------------------------------------------------------------------------------------------------------
            changed = False;
            for attRef in comboAttList:  #Each condition specifies different attributes, so we need to go through all attributes in the dataset.
                attributeInfo = cons.env.format_data.attribute_info[attRef]
                #-----------------------------
                ref = 0
                #if attRef in self.specified_attributes:
                if attRef in p_self_specifiedAttList:
                    ref += 1
                #if attRef in cl.specified_attributes:
                if attRef in p_cl_specifiedAttList:
                    ref += 1
                #-----------------------------

                if ref == 0:    #Attribute not specified in either condition (Attribute type makes no difference)
                    print("Error: UniformCrossover!")
                    pass

                elif ref == 1:  #Attribute specified in only one condition - do probabilistic switch of whole attribute state (Attribute type makes no difference)
                    if attRef in p_self_specifiedAttList and random.random() > probability:
                        i = self.specified_attributes.index(attRef) #reference to the position of the attribute in the rule representation
                        cl.condition.append(self.condition.pop(i)) #Take attribute from self and add to cl
                        cl.specified_attributes.append(attRef)
                        self.specified_attributes.remove(attRef)
                        changed = True #Remove att from self and add to cl


                    if attRef in p_cl_specifiedAttList and random.random() < probability:
                        i = cl.specified_attributes.index(attRef) #reference to the position of the attribute in the rule representation
                        self.condition.append(cl.condition.pop(i)) #Take attribute from self and add to cl
                        self.specified_attributes.append(attRef)
                        cl.specified_attributes.remove(attRef)
                        changed = True #Remove att from cl and add to self.


                else: #Attribute specified in both conditions - do random crossover between state alleles.  The same attribute may be specified at different positions within either classifier
                    #-------------------------------------------------------
                    # CONTINUOUS ATTRIBUTE
                    #-------------------------------------------------------
                    if attributeInfo[0]:
                        i_cl1 = self.specified_attributes.index(attRef) #pairs with self (classifier 1)
                        i_cl2 = cl.specified_attributes.index(attRef)   #pairs with cl (classifier 2)
                        tempKey = random.randint(0,3) #Make random choice between 4 scenarios, Swap minimums, Swap maximums, Self absorbs cl, or cl absorbs self.
                        if tempKey == 0:    #Swap minimum
                            temp = self.condition[i_cl1][0]
                            self.condition[i_cl1][0] = cl.condition[i_cl2][0]
                            cl.condition[i_cl2][0] = temp
                        elif tempKey == 1:  #Swap maximum
                            temp = self.condition[i_cl1][1]
                            self.condition[i_cl1][1] = cl.condition[i_cl2][1]
                            cl.condition[i_cl2][1] = temp
                        else: #absorb range
                            allList = self.condition[i_cl1] + cl.condition[i_cl2]
                            newMin = min(allList)
                            newMax = max(allList)
                            if tempKey == 2:  #self absorbs cl
                                self.condition[i_cl1] = [newMin,newMax]
                                #Remove cl
                                cl.condition.pop(i_cl2)
                                cl.specified_attributes.remove(attRef)
                            else: #cl absorbs self
                                cl.condition[i_cl2] = [newMin,newMax]
                                #Remove self
                                self.condition.pop(i_cl1)
                                self.specified_attributes.remove(attRef)
                    #-------------------------------------------------------
                    # DISCRETE ATTRIBUTE
                    #-------------------------------------------------------
                    else:
                        pass
            tempList1 = copy.deepcopy(p_self_specifiedAttList)
            tempList2 = copy.deepcopy(cl.specified_attributes)
            tempList1.sort()
            tempList2.sort()
            if changed and (tempList1 == tempList2):
                changed = False

            if self.action != cl.action and random.random() > probability:
                # Switch phenotypes of 2 classifiers if GA is run in match set
                temp = self.action
                self.action = cl.action
                cl.action = temp
                changed = True
            return changed
        #-------------------------------------------------------
        # CONTINUOUS PHENOTYPE CROSSOVER
        #-------------------------------------------------------
        elif random.random() < 0.5:
            return self.actionCrossover(cl)


    def twoPointCrossover(self, cl):
        """ Applies two point crossover and returns if the classifiers changed. Handles merely discrete attributes and phenotypes """
        points = []
        changed = False
        points.append( int( random.random() * ( cons.env.format_data.numb_attributes + 1 ) ) )
        secondPoint = int( random.random() * ( cons.env.format_data.numb_attributes + 1 ) )
        if points[0] > secondPoint:
            tempPoint = points[0]
            points[0] = secondPoint
            points.append( tempPoint )
        else:
            points.append( secondPoint )
        if cons.env.format_data.discrete_action:
            p_self_specifiedAttList = copy.deepcopy(self.specified_attributes)
            p_cl_specifiedAttList = copy.deepcopy(cl.specified_attributes)
            for i in range( points[1] ):
                if i >= points[0]:
                    if i in p_self_specifiedAttList:
                        if i not in p_cl_specifiedAttList:
                            index = self.specified_attributes.index(i)
                            cl.condition.append(self.condition.pop(index))
                            cl.specified_attributes.append(i)
                            self.specified_attributes.remove(i)
                            changed = True #Remove att from self and add to cl
                    elif i in p_cl_specifiedAttList:
                        index = cl.specified_attributes.index(i) #reference to the position of the attribute in the rule representation
                        self.condition.append(cl.condition.pop(index)) #Take attribute from self and add to cl
                        self.specified_attributes.append(i)
                        cl.specified_attributes.remove(i)
                        changed = True
        return changed



    def phenotypeCrossover(self, cl):
        """ Crossover a continuous phenotype """
        changed = False
        if self.action[0] == cl.action[0] and self.action[1] == cl.action[1]:
            return changed
        else:
            tempKey = random.random() < 0.5 #Make random choice between 4 scenarios, Swap minimums, Swap maximums, Children preserve parent phenotypes.
            if tempKey: #Swap minimum
                temp = self.action[0]
                self.action[0] = cl.action[0]
                cl.action[0] = temp
                changed = True
            elif tempKey:  #Swap maximum
                temp = self.action[1]
                self.action[1] = cl.action[1]
                cl.action[1] = temp
                changed = True

        return changed


    def Mutation(self, state):
        """ Mutates the condition of the classifier. Also handles phenotype mutation. This is a niche mutation, which means that the resulting classifier will still match the current instance.  """
        changed = False;
        #-------------------------------------------------------
        # MUTATE CONDITION
        #-------------------------------------------------------
        for attRef in range(cons.env.format_data.numb_attributes):  #Each condition specifies different attributes, so we need to go through all attributes in the dataset.
            attributeInfo = cons.env.format_data.attribute_info[attRef]
            if random.random() < cons.mu and state[attRef] != cons.labelMissingData:
                #MUTATION--------------------------------------------------------------------------------------------------------------
                if attRef not in self.specified_attributes: #Attribute not yet specified
                    self.specified_attributes.append(attRef)
                    self.condition.append(self.buildMatch(attRef, state)) #buildMatch handles both discrete and continuous attributes
                    changed = True

                elif attRef in self.specified_attributes: #Attribute already specified
                    i = self.specified_attributes.index(attRef) #reference to the position of the attribute in the rule representation
                    #-------------------------------------------------------
                    # DISCRETE OR CONTINUOUS ATTRIBUTE - remove attribute specification with 50% chance if we have continuous attribute, or 100% if discrete attribute.
                    #-------------------------------------------------------
                    if not attributeInfo[0] or random.random() > 0.5:
                        self.specified_attributes.remove(attRef)
                        self.condition.pop(i) #buildMatch handles both discrete and continuous attributes
                        changed = True
                    #-------------------------------------------------------
                    # CONTINUOUS ATTRIBUTE - (mutate range with 50% probability vs. removing specification of this attribute all together)
                    #-------------------------------------------------------
                    else:
                        #Mutate continuous range - based on Bacardit 2009 - Select one bound with uniform probability and add or subtract a randomly generated offset to bound, of size between 0 and 50% of att domain.
                        attRange = float(attributeInfo[1][1]) - float(attributeInfo[1][0])
                        mutateRange = random.random()*0.5*attRange
                        if random.random() > 0.5: #Mutate minimum
                            if random.random() > 0.5: #Add
                                self.condition[i][0] += mutateRange
                            else: #Subtract
                                self.condition[i][0] -= mutateRange
                        else: #Mutate maximum
                            if random.random() > 0.5: #Add
                                self.condition[i][1] += mutateRange
                            else: #Subtract
                                self.condition[i][1] -= mutateRange

                        #Repair range - such that min specified first, and max second.
                        self.condition[i].sort()
                        changed = True
                #-------------------------------------------------------
                # NO MUTATION OCCURS
                #-------------------------------------------------------
                else:
                    pass
        #-------------------------------------------------------
        # MUTATE PHENOTYPE
        #-------------------------------------------------------
        if cons.env.format_data.discrete_action:
            nowChanged = self.discreteActionMutation()
        #else:
        #    nowChanged = self.continuousActionMutation(phenotype)

        if changed or nowChanged:
            return True


    def discreteActionMutation(self):
        """ Mutate this rule's discrete phenotype. """
        changed = False
        if random.random() < cons.mu:
            phenotypeList = copy.deepcopy(cons.env.format_data.action_list)
            phenotypeList.remove(self.action)
            newPhenotype = random.sample(phenotypeList,1)
            self.action = newPhenotype[0]
            changed= True

        return changed


    def continuousActionMutation(self, phenotype):
        """ Mutate this rule's continuous phenotype. """
        changed = False
        if random.random() < cons.mu: #Mutate continuous phenotype
            phenRange = self.action[1] - self.action[0]
            mutateRange = random.random()*0.5*phenRange
            tempKey = random.randint(0,2) #Make random choice between 3 scenarios, mutate minimums, mutate maximums, mutate both
            if tempKey == 0: #Mutate minimum
                if random.random() > 0.5 or self.action[0] + mutateRange <= phenotype: #Checks that mutated range still contains current phenotype
                    self.action[0] += mutateRange
                else: #Subtract
                    self.action[0] -= mutateRange
                changed = True
            elif tempKey == 1: #Mutate maximum
                if random.random() > 0.5 or self.action[1] - mutateRange >= phenotype: #Checks that mutated range still contains current phenotype
                    self.action[1] -= mutateRange
                else: #Subtract
                    self.action[1] += mutateRange
                changed = True
            else: #mutate both
                if random.random() > 0.5 or self.action[0] + mutateRange <= phenotype: #Checks that mutated range still contains current phenotype
                    self.action[0] += mutateRange
                else: #Subtract
                    self.action[0] -= mutateRange
                if random.random() > 0.5 or self.action[1] - mutateRange >= phenotype: #Checks that mutated range still contains current phenotype
                    self.action[1] -= mutateRange
                else: #Subtract
                    self.action[1] += mutateRange
                changed = True

            #Repair range - such that min specified first, and max second.
            self.action.sort()
        #---------------------------------------------------------------------
        return changed

    #--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    # SUBSUMPTION METHODS
    #--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    def subsumes(self, cl):
        """ Returns if the classifier (self) subsumes cl """
        #-------------------------------------------------------
        # DISCRETE PHENOTYPE
        #-------------------------------------------------------
        if cons.env.format_data.discrete_action:
            if cl.action == self.action:
                if self.isPossibleSubsumer() and self.isMoreGeneral(cl):
                    self.subsumer_cnt += cl.numerosity
                    return True
            return False
        #-------------------------------------------------------
        # CONTINUOUS PHENOTYPE -  NOTE: for continuous phenotypes, the subsumption intuition is reversed, i.e. While a subsuming rule condition is more general, a subsuming phenotype is more specific.
        #-------------------------------------------------------
        else:
            if self.action[0] >= cl.action[0] and self.action[1] <= cl.action[1]:
                if self.isPossibleSubsumer() and self.isMoreGeneral(cl):
                    return True
            return False


    def isPossibleSubsumer(self):
        """ Returns if the classifier (self) is a possible subsumer. A classifier must be as or more accurate than the classifier it is trying to subsume.  """
        if self.actionCount > cons.theta_sub and self.error < cons.err_sub: #self.prediction < cons.err_sub: (why does it work?)
            return True
        return False


    def isMoreGeneral(self,cl):
        """ Returns if the classifier (self) is more general than cl. Check that all attributes specified in self are also specified in cl. """
        if len(self.specified_attributes) >= len(cl.specified_attributes):# and self.action != cl.action and self.prediction < cl.prediction and self.error > cl.error:
            return False
        for i in range(len(self.specified_attributes)): #Check each attribute specified in self.condition
            attributeInfo = cons.env.format_data.attribute_info[self.specified_attributes[i]]
            if self.specified_attributes[i] not in cl.specified_attributes:
                return False
            #-------------------------------------------------------
            # CONTINUOUS ATTRIBUTE
            #-------------------------------------------------------
            otherRef = cl.specified_attributes.index(self.specified_attributes[i])
            if attributeInfo[0]:
                #If self has a narrower ranger of values than it is a subsumer
                if self.condition[i][0] < cl.condition[otherRef][0]:
                    return False
                if self.condition[i][1] > cl.condition[otherRef][1]:
                    return False
            #else:   # discrete attributes
            #    if self.condition[i] != cl.condition[otherRef]:
            #        return False

        return True

    #--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    # DELETION METHOD
    #--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    def getDelProb(self, meanFitness):
        """  Returns the vote for deletion of the classifier. """
        self.delete_vote = self.mean_actionset_sz * self.numerosity
        if self.fitness < cons.delta*meanFitness * self.numerosity and self.actionCount > cons.theta_del:
            if self.fitness > 0.0:
                self.delete_vote *= meanFitness * self.numerosity / self.fitness
            else:
                self.delete_vote *= meanFitness / (cons.init_fit / self.numerosity)
        return self.delete_vote


    #--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    # OTHER METHODS
    #--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    def buildMatch(self, attRef, state):
        """ Builds a matching condition for the classifierCovering method. """
        attributeInfo = cons.env.format_data.attribute_info[attRef]
        #-------------------------------------------------------
        # CONTINUOUS ATTRIBUTE
        #-------------------------------------------------------
        if attributeInfo[0]:
            attRange = attributeInfo[1][1] - attributeInfo[1][0]
            rangeRadius = random.randint(25,75)*0.01*attRange / 2.0 #Continuous initialization domain radius.
            Low = state[attRef] - rangeRadius
            High = state[attRef] + rangeRadius
            condList = [Low,High] #ALKR Representation, Initialization centered around training instance  with a range between 25 and 75% of the domain size.
        #-------------------------------------------------------
        # DISCRETE ATTRIBUTE
        #-------------------------------------------------------
        else:
            condList = state[attRef] #State already formatted like GABIL in DataManagement

        return condList


    def equals(self, cl):
        """ Returns if the two classifiers are identical in condition and phenotype. This works for discrete or continuous attributes or phenotypes. """
        if cl.action == self.action and len(cl.specified_attributes) == len(self.specified_attributes): #Is phenotype the same and are the same number of attributes specified - quick equality check first.
            clRefs = sorted(cl.specified_attributes)
            selfRefs = sorted(self.specified_attributes)
            if clRefs == selfRefs:
                for i in range(len(cl.specified_attributes)):
                    tempIndex = self.specified_attributes.index(cl.specified_attributes[i])
                    if cl.condition[i] == self.condition[tempIndex]:
                        pass
                    else:
                        return False
                return True
        return False


    def updateXCSParameters(self, reward):
        """ Update the XCS classifier parameters: prediction payoff, prediction error and fitness. """
        payoff = reward
        if self.actionCount >= 1.0 / cons.beta:
            self.error = self.error + cons.beta * ( math.fabs( payoff - self.prediction ) - self.error )
            self.prediction = self.prediction + cons.beta * ( payoff - self.prediction )
        else:
            self.error = ( self.error * ( self.actionCount - 1 ) + math.fabs( payoff - self.prediction ) ) / self.actionCount
            self.prediction = ( self.prediction * ( self.actionCount - 1 ) + payoff ) / self.actionCount
        if self.error <= cons.offset_epsilon:
            self.accuracy = 1
        else:
            self.accuracy = cons.alpha * ( ( cons.offset_epsilon / self.error ) ** cons.nu ) #math.pow( cons.alpha, ( self.error - cons.offset_epsilon ) / cons.offset_epsilon )


    def updateFitness(self):
        if self.actionCount >= 1.0 / cons.beta:
            self.fitness = self.fitness + cons.beta * ( self.accuracy - self.fitness )
        else:
            self.fitness = ( self.fitness * ( self.actionCount - 1 ) + self.accuracy ) / self.actionCount


    def updateActionSetSize(self, actionSetSize):
        """  Updates the average action set size. """
        if self.actionCount >= 1.0 / cons.beta:
            self.mean_actionset_sz = self.mean_actionset_sz + cons.beta * (actionSetSize - self.mean_actionset_sz)
        else:
            self.mean_actionset_sz = (self.mean_actionset_sz * (self.actionCount-1)+ actionSetSize) / float(self.actionCount)


    def updateExperience(self):
        """ Increases the experience of the classifier by one. Once an epoch has completed, rule accuracy can't change."""
        self.matchCount += 1


    def updateActionExp(self):
        """ Increases the experience of the classifier by one. Once an epoch has completed, rule accuracy can't change."""
        self.actionCount += 1


    def updateGACount(self):
        """ Increment number of times the classifier is selected in GA by one, for statistics. """
        self.ga_count += 1


    def updateNumerosity(self, num):
        """ Updates the numberosity of the classifier.  Notice that 'num' can be negative! """
        self.numerosity += num


    def updateTimeStamp(self, ts):
        """ Sets the time stamp of the classifier. """
        self.ga_timestamp = ts


    def setPrediction(self,pred):
        """ Sets the accuracy of the classifier """
        self.prediction = pred


    def setError(self,err):
        """ Sets the accuracy of the classifier """
        self.error = err


    def setAccuracy(self,acc):
        """ Sets the accuracy of the classifier """
        self.accuracy = acc


    def setFitness(self, fit):
        """  Sets the fitness of the classifier. """
        self.fitness = fit


    #--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    # PRINT CLASSIFIER FOR POPULATION OUTPUT FILE
    #--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    def printClassifier(self):
        """ Formats and returns an output string describing this classifier. """
        classifierString = ""
        for attRef in range(cons.env.format_data.numb_attributes):
            attributeInfo = cons.env.format_data.attribute_info[attRef]
            if attRef in self.specified_attributes:  #If the attribute was specified in the rule
                i = self.specified_attributes.index(attRef)
                #-------------------------------------------------------
                # CONTINUOUS ATTRIBUTE
                #-------------------------------------------------------
                if attributeInfo[0]:
                    classifierString += str(self.condition[i][0])+';'+str(self.condition[i][1]) + "\t"
                #-------------------------------------------------------
                # DISCRETE ATTRIBUTE
                #-------------------------------------------------------
                else:
                    classifierString += str(self.condition[i]) + "\t"
            else: # Attribute is wild.
                classifierString += '#' + "\t"
        #-------------------------------------------------------------------------------
        specificity = len(self.condition) / float(cons.env.format_data.numb_attributes)

        if cons.env.format_data.discrete_action:
            classifierString += str(self.action)+"\t"
        else:
            classifierString += str(self.action[0])+';'+str(self.action[1])+"\t"
        #------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
        classifierString += '{:.1f}'.format(self.prediction)+"\t"+'{:.2f}'.format(self.error)+"\t"+'{:.2f}'.format(self.fitness)+"\t"+str(self.numerosity)+"\t"+str(self.ga_count)+"\t"
        classifierString += str(self.subsumer_cnt)+"\t\t"+'{:.1f}'.format(self.mean_actionset_sz)+"\t\t"+str(self.ga_timestamp)+"\t\t"+str(self.init_timestamp)+"\t\t"+'{:.2f}'.format(specificity)+"\t\t"
        classifierString += '{:.1f}'.format(self.delete_vote)+"\t\t"+str(self.actionCount)+"\n"

        #------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
        return classifierString