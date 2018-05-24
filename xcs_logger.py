"""
Name:        XCS_Logging.py
Authors:     Bao Trung
Contact:     baotrung@ecs.vuw.ac.nz
Created:     July, 2017
Description:
---------------------------------------------------------------------------------------------------------------------------------------------------------
XCS: Michigan-style Learning Classifier System - A LCS for Reinforcement Learning.  This XCS follows the version descibed in "An Algorithmic Description of XCS" published by Martin Butz and Stewart Wilson ( 2002 ).

---------------------------------------------------------------------------------------------------------------------------------------------------------
"""

#Import Required Modules-------------------------------
from xcs_constants import *
import copy
#------------------------------------------------------

class Logger:
    def __init__( self, base_name, mode='w' ):
        try:
            self._base_name = base_name
            self._logfile = open( base_name + '_LearnTrack.txt', mode )
        except Exception as inst:
            print( type( inst ) )
            print( inst.args )
            print( inst )
            print( 'cannot open', base_name )
            raise
        else:
            self._logfile.write( "Explore_Iteration\tMacroPopSize\tMicroPopSize\tAverageSteps\tPercentSolved\tAveGenerality\tTime( min )\n" )


    def writePopStats( self, train_eval, explore_iter, pop, steps_to_goal ):
        """ Makes output text file which includes all of the evaluation statistics for a complete analysis of all training and testing data on the current XCS rule population. """
        try:
            pop_stats_out = open( self._base_name + '_'+ str( explore_iter )+'_PopStats.txt','w' )
        except Exception as inst:
            print( type( inst ) )
            print( inst.args )
            print( inst )
            print( 'cannot open', self._base_name + '_'+ str( explore_iter )+'_PopStats.txt' )
            raise
        else:
            print( "Writing Population Statistical Summary File..." )

        #Evaluation of pop
        pop_stats_out.write( "Performance Statistics:------------------------------------------------------------------------\n" )
        pop_stats_out.write( "Average Steps" + str( train_eval[0] ) + "\n Training Coverage" + str( train_eval[1] ) + "\n" )

        pop_stats_out.write( "Population Characterization:------------------------------------------------------------------------\n" )
        pop_stats_out.write( "MacroPopSize\tMicroPopSize\tGenerality\n" )
        pop_stats_out.write( str( len( pop.pop_set ) )+"\t"+ str( pop.micro_pop_size )+"\t"+str( pop.average_generality )+"\n\n" )

        pop_stats_out.write( "SpecificitySum:------------------------------------------------------------------------\n" )
        attribute_list = []
        for i in range( cons.env.number_of_attributes ):
            attribute_list.append( "A" + str( i ) )

        for i in range( len( attribute_list ) - 1 ):
            pop_stats_out.write( str( attribute_list[i] ) + "\t" )
        pop_stats_out.write( str( attribute_list[ -1 ] ) + "\n" )

        # Prints out the Specification Sum for each attribute
        for i in range( len( pop.attribute_spec_list ) ):
            if i < len( pop.attribute_spec_list )-1:
                pop_stats_out.write( str( pop.attribute_spec_list[i] )+"\t" )
            else:
                pop_stats_out.write( str( pop.attribute_spec_list[i] )+"\n" )

        pop_stats_out.write( "\nAccuracySum:------------------------------------------------------------------------\n" )
        for i in range( len( attribute_list ) ):
            if i < len( attribute_list )-1:
                pop_stats_out.write( str( attribute_list[i] )+"\t" )
            else:
                pop_stats_out.write( str( attribute_list[i] )+"\n" )

        # Prints out the Accuracy Weighted Specification Count for each attribute
        for i in range( len( pop.attribute_pred_list ) ):
            if i < len( pop.attribute_pred_list )-1:
                pop_stats_out.write( str( pop.attribute_pred_list[i] )+"\t" )
            else:
                pop_stats_out.write( str( pop.attribute_pred_list[i] )+"\n" )

        #Time Track ---------------------------------------------------------------------------------------------------------
        pop_stats_out.write( "\nRun Time( in minutes ):------------------------------------------------------------------------\n" )
        pop_stats_out.write( cons.timer.reportTimes() )
        pop_stats_out.write( "\nSpeedTrackerSave:------------------------------------------------------------------------\n" )
        for i in range( len( steps_to_goal ) ):
            pop_stats_out.write( str( steps_to_goal[i] )+"\t" )

        pop_stats_out.close()


    def writePop( self, explore_iter, pop ):
        """ Writes a tab delimited text file outputting the entire evolved rule population, including conditions, phenotypes, and all rule parameters. """
        try:
            rule_pop_out = open( self._base_name + '_'+ str( explore_iter )+'_RulePop.txt','w' )
        except Exception as inst:
            print( type( inst ) )
            print( inst.args )
            print( inst )
            print( 'cannot open', self._base_name + '_'+ str( explore_iter )+'_RulePop.txt' )
            raise
        else:
            print( "Writing Population as Data File..." )

        #Write Header-----------------------------------------------------------------------------------------------------------------------------------------------
        attribute_list = []
        for i in range( cons.env.number_of_attributes ):
            attribute_list.append( "A" + str( i ) )
        for i in range( len( attribute_list ) ):
            rule_pop_out.write( str( attribute_list[i] )+"\t" )
        rule_pop_out.write( "Phenotype\tPrediction\tError\tFitness\tNumerosity\tGACount\tAveActionSetSize\tTimeStampGA\tInitTimeStamp\tSpecificity\tDeletionProb\tActionCount\n" )

        #Write each classifier--------------------------------------------------------------------------------------------------------------------------------------
        for cl in pop.pop_set:
            rule_pop_out.write( str( cl.printClassifier() ) )

        rule_pop_out.close()


    def track( self, population, average_steps, trials_count, tracking_frequency, solved_percentage ):
        """ Returns a formated output string to be printed to the Learn Track output file. """
        trackString = str( trials_count )+ "\t" + str( len( population.pop_set ) ) + "\t" + str( population.micro_pop_size ) + "\t" + str( average_steps ) + "\t" \
        + str( solved_percentage ) + "\t" + str( population.average_generality ) + "\t" + str( cons.timer.returnGlobalTimer() )+ "\n"
        print( ( "Epoch: "+str( int( trials_count/tracking_frequency ) )+"\t Trials: " + str( trials_count ) + "\t MacroPop: " + str( len( population.pop_set ) )
                 + "\t Covered: " + str( population.covered_cl_counter ) + "\t Crossovered: " + str( population.crossovered_cl_counter ) + "\t Mutated: " + str( population.mutated_cl_counter )
                 + "\t MicroPop: " + str( population.micro_pop_size ) + "\t AveSteps: " + str( average_steps ) + "\t PercentSolved: " + str( solved_percentage )
                 + "\t AveGen: " + str( population.average_generality )  + "\t Time: " + str( cons.timer.returnGlobalTimer() ) ) )

        return self._logfile.write( trackString )


    def closeLogFile( self ):
        self._logfile.close()
