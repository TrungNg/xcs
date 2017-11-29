"""
Name:        xcs_classifier.py
Authors:     Bao Trung
Contact:     baotrung@ecs.vuw.ac.nz
Created:     July, 2017
Description:
---------------------------------------------------------------------------------------------------------------------------------------------------------
XCS: Michigan-style Learning Classifier System - A LCS for Reinforcement Learning.  This XCS follows the version descibed in "An Algorithmic Description of XCS" published by Martin Butz and Stewart Wilson ( 2002 ).

---------------------------------------------------------------------------------------------------------------------------------------------------------
"""

#Import Required Modules---------------
from xcs_constants import *
import random
import copy
import math
#--------------------------------------

class Classifier:
    def __init__( self,a=None,b=None,c=None ):
        #Major Parameters --------------------------------------------------
        self.specified_att_list = []      # Attribute Specified in classifier: Similar to Bacardit 2009 - ALKR + GABIL, continuous and discrete rule representation
        self.condition = []             # States of Attributes Specified in classifier: Similar to Bacardit 2009 - ALKR + GABIL, continuous and discrete rule representation
        self.action = None           # Class if the endpoint is discrete, and a continuous action if the endpoint is continuous

        self.prediction = cons.init_pred  # Classifier payoff - initialized to a constant initial payoff value
        self.error = cons.init_err      # Classifier error - initialized to a constant initial error value
        self.fitness = cons.init_fit    # Classifier fitness - initialized to a constant initial fitness value
        self.accuracy = 0.0             # Classifier accuracy - Accuracy calculated using only instances in the dataset which this rule matched.
        self.numerosity = 1             # The number of rule copies stored in the population.  ( Indirectly stored as incremented numerosity )
        self.ave_action_set_size = 0       # A parameter used in deletion which reflects the size of match sets within this rule has been included.
        self.deletion_vote = None        # The current deletion weight for this classifier.
        self.origin = None              # method creating this classifier in the first time

        #Experience Management ---------------------------------------------
        self.timestamp_GA = None         # Time since rule last in a match set.
        self.init_timestamp = None       # Iteration in which the rule first appeared.

        #Classifier Accuracy Tracking --------------------------------------
        self.GA_count = 0
        self.subsumer_count = 0
        #self.matchCount = 0             # Known in many LCS implementations as experience i.e. the total number of times this classifier was in a match set
        self.action_count = 0            # The total number of times this classifier was chosen in action set

        if isinstance( b,list ):
            self.classifierCovering( a,b,c )
        elif isinstance( a,Classifier ):
            self.classifierCopy( a, b )
        elif isinstance( a,list ) and b == None:
            self.rebootClassifier( a )
        else:
            print( "Classifier: Error building classifier." )

    #--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    # CLASSIFIER CONSTRUCTION METHODS
    #--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    def classifierCovering( self, explore_iter, state, action ):
        """ Makes a new classifier when the covering mechanism is triggered.  The new classifier will match the current training instance.
        Covering will NOT produce a default rule ( i.e. a rule with a completely general condition ). """
        #Initialize new classifier parameters----------
        self.timestamp_GA = explore_iter
        self.init_timestamp = explore_iter

        if action != None:
            self.action = action
        else:
            self.action = random.choice( cons.env.action_list )
        #-------------------------------------------------------
        # GENERATE MATCHING CONDITION
        #-------------------------------------------------------
        while len( self.specified_att_list ) < 1:
            for att_ref in range( len( state ) ):
                if random.random() < cons.p_spec and state[ att_ref ] != cons.label_missing_data:
                    self.specified_att_list.append( att_ref )
                    self.condition.append( state[ att_ref ] )


    def classifierCopy( self, clOld, explore_iter ):
        """  Constructs an identical Classifier.  However, the experience of the copy is set to 0 and the numerosity
        is set to 1 since this is indeed a new individual in a population. Used by the genetic algorithm to generate
        offspring based on parent classifiers."""
        self.specified_att_list = copy.deepcopy( clOld.specified_att_list )
        self.condition = copy.deepcopy( clOld.condition )
        self.action = copy.deepcopy( clOld.action )
        self.timestamp_GA = explore_iter
        self.init_timestamp = explore_iter
        self.prediction = clOld.prediction
        self.error = clOld.error
        self.fitness = clOld.fitness
        #self.accuracy = clOld.accuracy


    def rebootClassifier( self, classifier_list ):
        """ Rebuilds a saved classifier as part of the population Reboot """
        numb_attributes = cons.env.number_of_attributes
        for att_ref in range( 0,numb_attributes ):
            if classifier_list[att_ref] != '#':  #Attribute in rule is not wild
                self.condition.append( classifier_list[att_ref] )
                self.specified_att_list.append( att_ref )
        #-------------------------------------------------------
        # DISCRETE PHENOTYPE
        #-------------------------------------------------------
        self.action = str( classifier_list[numb_attributes] )

        self.prediction = float( classifier_list[numb_attributes+1] )
        self.error = float( classifier_list[numb_attributes+2] )
        self.fitness = float( classifier_list[numb_attributes+3] )
        self.numerosity = int( classifier_list[numb_attributes+4] )
        self.GA_count = float( classifier_list[numb_attributes+5] )
        self.subsumer_count = float( classifier_list[numb_attributes+6] )
        self.ave_action_set_size = float( classifier_list[numb_attributes+7] )
        self.timestamp_GA = int( classifier_list[numb_attributes+8] )
        self.init_timestamp = int( classifier_list[numb_attributes+9] )

        self.deletion_vote = float( classifier_list[numb_attributes+11] )
        self.action_count = int( classifier_list[numb_attributes+12] )


    #--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    # MATCHING
    #--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    def match( self, state ):
        """ Returns if the classifier matches in the current situation. """
        for i in range( len( self.condition ) ):
            state_rep = state[self.specified_att_list[i]]
            if state_rep == self.condition[i] or state_rep == cons.label_missing_data:
                pass
            else:
                return False
        return True


    #--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    # GENETIC ALGORITHM MECHANISMS
    #--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    def uniformCrossover( self, cl ):
        """ Applies uniform crossover and returns if the classifiers changed. Handles both discrete and continuous attributes.
        #SWARTZ: self. is where for the better attributes are more likely to be specified
        #DEVITO: cl. is where less useful attribute are more likely to be specified
        """
        p_self_specified_att_list = copy.deepcopy( self.specified_att_list )
        p_cl_specified_att_list = copy.deepcopy( cl.specified_att_list )
        probability = 0.5  #Equal probability for attribute alleles to be exchanged.

        #Make list of attribute references appearing in at least one of the parents.-----------------------------
        combo_att_list = []
        for i in p_self_specified_att_list:
            combo_att_list.append( i )
        for i in p_cl_specified_att_list:
            if i not in combo_att_list:
                combo_att_list.append( i )
            combo_att_list.remove( i )
        combo_att_list.sort()
        #--------------------------------------------------------------------------------------------------------
        changed = False;
        for att_ref in combo_att_list:  #Each condition specifies different attributes, so we need to go through all attributes in the dataset.
            #-----------------------------
            ref = 0
            #if att_ref in self.specified_att_list:
            if att_ref in p_self_specified_att_list:
                ref += 1
            #if att_ref in cl.specified_att_list:
            if att_ref in p_cl_specified_att_list:
                ref += 1
            #-----------------------------

            if ref == 0:    #Attribute not specified in either condition ( Attribute type makes no difference )
                print( "Error: UniformCrossover!" )
                pass

            elif ref == 1:  #Attribute specified in only one condition - do probabilistic switch of whole attribute state ( Attribute type makes no difference )
                if att_ref in p_self_specified_att_list and random.random() > probability:
                    i = self.specified_att_list.index( att_ref ) #reference to the position of the attribute in the rule representation
                    cl.condition.append( self.condition.pop( i ) ) #Take attribute from self and add to cl
                    cl.specified_att_list.append( att_ref )
                    self.specified_att_list.remove( att_ref )
                    changed = True #Remove att from self and add to cl


                if att_ref in p_cl_specified_att_list and random.random() < probability:
                    i = cl.specified_att_list.index( att_ref ) #reference to the position of the attribute in the rule representation
                    self.condition.append( cl.condition.pop( i ) ) #Take attribute from self and add to cl
                    self.specified_att_list.append( att_ref )
                    cl.specified_att_list.remove( att_ref )
                    changed = True #Remove att from cl and add to self.

        tempList1 = copy.deepcopy( p_self_specified_att_list )
        tempList2 = copy.deepcopy( cl.specified_att_list )
        tempList1.sort()
        tempList2.sort()
        if changed and ( tempList1 == tempList2 ):
            changed = False

        if self.action != cl.action and random.random() > probability:
            # Switch actions of 2 classifiers if GA is run in match set
            temp = self.action
            self.action = cl.action
            cl.action = temp
            changed = True
        return changed


    def twoPointCrossover( self, cl ):
        """ Applies two point crossover and returns if the classifiers changed. Handles merely discrete attributes and actions """
        points = []
        changed = False
        points.append( int( random.random() * ( cons.env.number_of_attributes + 1 ) ) )
        new_point = int( random.random() * ( cons.env.number_of_attributes + 1 ) )
        if points[0] > new_point:
            temp_point = points[0]
            points[0] = new_point
            points.append( temp_point )
        else:
            points.append( new_point )
        p_self_specified_att_list = copy.deepcopy( self.specified_att_list )
        p_cl_specified_att_list = copy.deepcopy( cl.specified_att_list )
        for i in range( points[1] ):
            if i >= points[0]:
                if i in p_self_specified_att_list:
                    if i not in p_cl_specified_att_list:
                        index = self.specified_att_list.index( i )
                        cl.condition.append( self.condition.pop( index ) )
                        cl.specified_att_list.append( i )
                        self.specified_att_list.remove( i )
                        changed = True #Remove att from self and add to cl
                elif i in p_cl_specified_att_list:
                    index = cl.specified_att_list.index( i ) #reference to the position of the attribute in the rule representation
                    self.condition.append( cl.condition.pop( index ) ) #Take attribute from self and add to cl
                    self.specified_att_list.append( i )
                    cl.specified_att_list.remove( i )
                    changed = True
        return changed



    def actionCrossover( self, cl ):
        """ Crossover a continuous action """
        changed = False
        if self.action[0] == cl.action[0] and self.action[1] == cl.action[1]:
            return changed
        else:
            tempKey = random.random() < 0.5 #Make random choice between 4 scenarios, Swap minimums, Swap maximums, Children preserve parent actions.
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


    def Mutation( self, state ):
        """ Mutates the condition of the classifier. Also handles action mutation. This is a niche mutation, which means that the resulting classifier will still match the current instance.  """
        changed = False;
        #-------------------------------------------------------
        # MUTATE CONDITION
        #-------------------------------------------------------
        for att_ref in range( cons.env.number_of_attributes ):  #Each condition specifies different attributes, so we need to go through all attributes in the dataset.
            if random.random() < cons.mu and state[ att_ref ] != cons.label_missing_data:
                #MUTATION--------------------------------------------------------------------------------------------------------------
                if att_ref not in self.specified_att_list: #Attribute not yet specified
                    self.specified_att_list.append( att_ref )
                    self.condition.append( self.buildMatch( att_ref, state ) ) #buildMatch handles both discrete and continuous attributes
                    changed = True

                elif att_ref in self.specified_att_list: #Attribute already specified
                    i = self.specified_att_list.index( att_ref ) #reference to the position of the attribute in the rule representation
                    #-------------------------------------------------------
                    # DISCRETE OR CONTINUOUS ATTRIBUTE - remove attribute specification with 50% chance if we have continuous attribute, or 100% if discrete attribute.
                    #-------------------------------------------------------
                    self.specified_att_list.remove( att_ref )
                    self.condition.pop( i ) #buildMatch handles both discrete and continuous attributes
                    changed = True

                #-------------------------------------------------------
                # NO MUTATION OCCURS
                #-------------------------------------------------------
                else:
                    pass
        #-------------------------------------------------------
        # MUTATE PHENOTYPE
        #-------------------------------------------------------
        nowChanged = self.discretePhenotypeMutation()

        if changed or nowChanged:
            return True


    def discretePhenotypeMutation( self ):
        """ Mutate this rule's discrete action. """
        changed = False
        if random.random() < cons.mu:
            action_list = copy.deepcopy( cons.env.action_list )
            action_list.remove( self.action )
            new_phenotype = random.sample( action_list, 1 )
            self.action = new_phenotype[0]
            changed= True

        return changed


    def continuousPhenotypeMutation( self, action ):
        """ Mutate this rule's continuous action. """
        changed = False
        if random.random() < cons.mu: #Mutate continuous action
            phenRange = self.action[1] - self.action[0]
            mutateRange = random.random()*0.5*phenRange
            tempKey = random.randint( 0,2 ) #Make random choice between 3 scenarios, mutate minimums, mutate maximums, mutate both
            if tempKey == 0: #Mutate minimum
                if random.random() > 0.5 or self.action[0] + mutateRange <= action: #Checks that mutated range still contains current action
                    self.action[0] += mutateRange
                else: #Subtract
                    self.action[0] -= mutateRange
                changed = True
            elif tempKey == 1: #Mutate maximum
                if random.random() > 0.5 or self.action[1] - mutateRange >= action: #Checks that mutated range still contains current action
                    self.action[1] -= mutateRange
                else: #Subtract
                    self.action[1] += mutateRange
                changed = True
            else: #mutate both
                if random.random() > 0.5 or self.action[0] + mutateRange <= action: #Checks that mutated range still contains current action
                    self.action[0] += mutateRange
                else: #Subtract
                    self.action[0] -= mutateRange
                if random.random() > 0.5 or self.action[1] - mutateRange >= action: #Checks that mutated range still contains current action
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
    def subsumes( self, cl ):
        """ Returns if the classifier ( self ) subsumes cl """
        if cl.action == self.action:
            if self.isPossibleSubsumer() and self.isMoreGeneral( cl ):
                self.subsumer_count += cl.numerosity
                return True
        return False


    def isPossibleSubsumer( self ):
        """ Returns if the classifier ( self ) is a possible subsumer. A classifier must be as or more accurate than the classifier it is trying to subsume.  """
        if self.action_count > cons.theta_sub and self.error < cons.err_sub: #self.prediction < cons.err_sub: ( why does it work? )
            return True
        return False


    def isMoreGeneral( self,cl ):
        """ Returns if the classifier ( self ) is more general than cl. Check that all attributes specified in self are also specified in cl. """
        if len( self.specified_att_list ) >= len( cl.specified_att_list ):# and self.action != cl.action and self.prediction < cl.prediction and self.error > cl.error:
            return False
        for i in range( len( self.specified_att_list ) ): #Check each attribute specified in self.condition
            if self.specified_att_list[i] not in cl.specified_att_list:
                return False
        return True

    #--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    # DELETION METHOD
    #--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    def getDelProb( self, mean_fitness ):
        """  Returns the vote for deletion of the classifier. """
        self.deletion_vote = self.ave_action_set_size * self.numerosity
        if self.fitness < cons.delta*mean_fitness * self.numerosity and self.action_count > cons.theta_del:
            if self.fitness > 0.0:
                self.deletion_vote *= mean_fitness * self.numerosity / self.fitness
            else:
                self.deletion_vote *= mean_fitness / ( cons.init_fit / self.numerosity )
        return self.deletion_vote


    #--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    # OTHER METHODS
    #--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    def buildMatch( self, att_ref, state ):
        """ Builds a matching condition for the classifierCovering method. """
        return state[ att_ref ]


    def equals( self, cl ):
        """ Returns if the two classifiers are identical in condition and action. This works for discrete or continuous attributes or actions. """
        if cl.action == self.action and len( cl.specified_att_list ) == len( self.specified_att_list ): #Is action the same and are the same number of attributes specified - quick equality check first.
            clRefs = sorted( cl.specified_att_list )
            selfRefs = sorted( self.specified_att_list )
            if clRefs == selfRefs:
                for i in range( len( cl.specified_att_list ) ):
                    tempIndex = self.specified_att_list.index( cl.specified_att_list[i] )
                    if cl.condition[i] == self.condition[tempIndex]:
                        pass
                    else:
                        return False
                return True
        return False


    def updateXCSParameters( self, reward, prediction ):
        """ Update the XCS classifier parameters: prediction payoff, prediction error and fitness. """
        predicted_payoff = reward + cons.gamma * prediction # check***
        if self.action_count >= 1.0 / cons.beta:
            self.error = self.error + cons.beta * ( math.fabs( predicted_payoff - self.prediction ) - self.error )
            self.prediction = self.prediction + cons.beta * ( predicted_payoff - self.prediction )
        else:
            self.error = ( self.error * ( self.action_count - 1 ) + math.fabs( predicted_payoff - self.prediction ) ) / self.action_count
            self.prediction = ( self.prediction * ( self.action_count - 1 ) + predicted_payoff ) / self.action_count
        if self.error <= cons.offset_epsilon:
            self.accuracy = 1
        else:
            self.accuracy = cons.alpha * ( ( cons.offset_epsilon / self.error ) ** cons.nu ) #math.pow( cons.alpha, ( self.error - cons.offset_epsilon ) / cons.offset_epsilon )


    def updateFitness( self ):
        if self.action_count >= 1.0 / cons.beta:
            self.fitness = self.fitness + cons.beta * ( self.accuracy - self.fitness )
        else:
            self.fitness = ( self.fitness * ( self.action_count - 1 ) + self.accuracy ) / self.action_count


    def updateActionSetSize( self, action_set_size ):
        """  Updates the average action set size. """
        if self.action_count >= 1.0 / cons.beta:
            self.ave_action_set_size = self.ave_action_set_size + cons.beta * ( action_set_size - self.ave_action_set_size )
        else:
            self.ave_action_set_size = ( self.ave_action_set_size * ( self.action_count-1 )+ action_set_size ) / float( self.action_count )


    def updateExperience( self ):
        """ Increases the experience of the classifier by one. Once an epoch has completed, rule accuracy can't change."""
        self.matchCount += 1


    def updateActionExp( self ):
        """ Increases the experience of the classifier by one. Once an epoch has completed, rule accuracy can't change."""
        self.action_count += 1


    def updateGACount( self ):
        """ Increment number of times the classifier is selected in GA by one, for statistics. """
        self.GA_count += 1


    def updateNumerosity( self, num ):
        """ Updates the numberosity of the classifier.  Notice that 'num' can be negative! """
        self.numerosity += num


    def updateTimeStamp( self, ts ):
        """ Sets the time stamp of the classifier. """
        self.timestamp_GA = ts


    def setPrediction( self, pred ):
        """ Sets the accuracy of the classifier """
        self.prediction = pred


    def setError( self, err ):
        """ Sets the accuracy of the classifier """
        self.error = err


    def setAccuracy( self, acc ):
        """ Sets the accuracy of the classifier """
        self.accuracy = acc


    def setFitness( self, fit ):
        """  Sets the fitness of the classifier. """
        self.fitness = fit


    #--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    # PRINT CLASSIFIER FOR POPULATION OUTPUT FILE
    #--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    def printClassifier( self ):
        """ Formats and returns an output string describing this classifier. """
        classifier_string = ""
        for att_ref in range( cons.env.number_of_attributes ):
            if att_ref in self.specified_att_list:  #If the attribute was specified in the rule
                i = self.specified_att_list.index( att_ref )
                classifier_string += str( self.condition[i] ) + "\t"
            else: # Attribute is wild.
                classifier_string += '#' + "\t"
        #-------------------------------------------------------------------------------
        specificity = len( self.condition ) / float( cons.env.number_of_attributes )

        classifier_string += str( self.action ) + "\t"
        classifier_string += str( self.prediction )+"\t"+str( self.error )+"\t"+str( self.fitness )+"\t"+str( self.numerosity )+"\t"+str( self.GA_count )+"\t"+str( self.subsumer_count )+"\t"+str( self.ave_action_set_size )+"\t"+str( self.timestamp_GA )+"\t"+str( self.init_timestamp )+"\t"+str( specificity )+"\t"
        classifier_string += str( self.deletion_vote ) + "\t" + str( self.action_count ) + "\n"

        #------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
        return classifier_string