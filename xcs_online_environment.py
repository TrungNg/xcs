"""
Name:        xcs_online_environment.py
Authors:     Bao Trung
Contact:     baotrung@ecs.vuw.ac.nz
Created:     July, 2017
Description:
---------------------------------------------------------------------------------------------------------------------------------------------------------
XCS: Michigan-style Learning Classifier System - A LCS for Reinforcement Learning.  This XCS follows the version descibed in "An Algorithmic Description of XCS" published by Martin Butz and Stewart Wilson (2002).

---------------------------------------------------------------------------------------------------------------------------------------------------------
"""

#Import Required Modules--------------------------------------
from xcs_constants import *
import random
import sys
#-------------------------------------------------------------

class Online_Environment:
    def __init__(self, problem, sizes):
        #Initialize global variables-------------------------------------------------
        self.data_ref = 0
        self.saved_dat_ref = 0
        options={ 'multiplexer':MulplexerGenerator,
                  'parity_countone':ParityCountOne }
        self.format_data = options[ problem.lower() ]( sizes )
        print( "Problem: " + problem + " size " + str( sizes ) )


    def getTrainInstance(self):
        """ Returns the a new training instance. """
        return self.format_data.generateInstance()


    def getTestInstance(self):
        """ Returns the a new testing instance. """
        return self.format_data.generateInstance()


    def resetDataRef(self, _):
        """ Do nothing. """
        return


    def startEvaluationMode(self):
        """ Do nothing. """
        return


    def stopEvaluationMode(self):
        """ Do nothing."""
        return


class DataGenerator:
    def __init__(self, sizes):
        """ Initialize general online data generator. """
        self.train_headers = self.test_headers = self.headers
        self.numb_train_instances = cons.tracking_frequency
        self.numb_test_instances = cons.tracking_frequency
        self.numb_attributes = sizes[ 0 ]
        self.attribute_info = [ [ 0, [] ] ] * sizes[ 0 ]
        self.discrete_action = True
        self.action_list = [ '0', '1' ]


class MulplexerGenerator( DataGenerator ):
    def __init__(self, sizes):
        """ Initialize online data generator for Multiplexer problem (maximum address bit size is 1000). """
        self.address_size = self._findAddressSize( sizes[ 0 ] )
        self.headers = []
        for i in range( self.address_size ):
            self.headers.append( 'A' + str( i ) )
        for i in range( self.address_size, sizes[ 0 ] ):
            self.headers.append( 'B' + str( i - self.address_size ) )
        super().__init__( sizes )


    def generateInstance(self):
        """ Return new Multiplexer instance of size provided by generating randomly. """
        condition = []
        #Generate random boolean string
        for _ in range( self.numb_attributes ):
            condition.append( str( random.randint( 0, 1 ) ) )

        gates=""
        for j in range( self.address_size ):
            gates += condition[ j ]
        gates_decimal = int( gates, 2 )
        output = condition[ self.address_size + gates_decimal ]

        return [ condition, output ]


    def _findAddressSize(self, num_bits):
        for i in range( 1000 ):
            if i + 2**i == num_bits:
                return i
            if i + 2**i > num_bits:
                break
        return None


class HiddenMultiplexer( DataGenerator ):
    def __init__(self, sizes):
        """ Initialize online data generator for Multiplexer problem (maximum address bit size is 1000). """
        self.parity_size = sizes[ 1 ]
        self.multiplexer_size = sizes[ 2 ]
        self.multiplexer_addr_size = self._findAddressSize( sizes[ 2 ] )
        self.headers = []
        if self.parity_size * self.multiplexer_size != sizes[ 0 ]:
            raise ValueError( "HiddenMultiplexer.__init__() failed because of inappropriate sizes provided." )
        for i in range( self.multiplexer_size ):
            if i < self.multiplexer_addr_size:
                prefix = 'A' + str( i )
            else:
                prefix = 'B' + str( i - self.multiplexer_addr_size )
            for j in range( self.parity_size ):
                self.headers.append( prefix + str( j ) )
        super().__init__( sizes )


    def generateInstance(self):
        """ Return new Multiplexer instance of size provided by generating randomly. """
        condition = []
        #Generate random boolean string
        for _ in range( self.numb_attributes ):
            condition.append( str( random.randint( 0, 1 ) ) )

        gates=""
        for i in range( self.multiplexer_addr_size ):
            count = 0
            for j in range( self.parity_size ):
                if condition[ i * self.parity_size + j ] == '1':
                    count += 1
            if count % 2 == 0:
                count = 0
            else:
                count = 1
            gates += str( count )
        gates_decimal = int( gates, 2 )
        output_parity_block = ( self.multiplexer_addr_size + gates_decimal ) * self.parity_size
        count = 0
        for i in range( self.parity_size ):
            if condition[ output_parity_block + i ] == '1':
                count += 1
        if count % 2 == 0:
            output = '0'
        else:
            output = '1'

        return [ condition, output ]


    def _findAddressSize(self, num_bits):
        for i in range( 1000 ):
            if i + 2**i == num_bits:
                return i
            if i + 2**i > num_bits:
                break
        return None


class ParityCountOne( DataGenerator ):
    def __init__(self, sizes):
        """ Initialize Paratiy - Count one data generator. Lowest level are blocks of data with same size.
        Their outputs are calculated by parity rule and then the outputs contribute to count one problems. """
        self.parity_size = sizes[ 1 ]
        self.countone_size = sizes[ 2 ]
        self.headers = []
        if self.parity_size * self.countone_size != sizes[ 0 ]:
            raise ValueError( "ParityCountOne.__init__() failed because of inappropriate sizes provided." )
        for i in range( self.countone_size ):
            prefix = 'B' + str( i )
            for j in range( self.parity_size ):
                self.headers.append( prefix + str( j ) )
        super().__init__( sizes )


    def generateInstance(self):
        """ Generate and return new instance with correct output. """
        condition = []
        #Generate random boolean string
        for _ in range( self.parity_size * self.countone_size ):
            condition.append( str( random.randint( 0, 1 ) ) )
        counts=[]
        for j in range( self.countone_size ):
            counts.append( 0 )
            for k in range( self.parity_size ):
                if condition[ j * self.parity_size + k ] == '1':
                    counts[ j ] += 1
            if counts[ j ] % 2 == 0:
                counts[ j ] = 0
            else:
                counts[ j ] = 1
        if sum( counts ) > self.countone_size / 2:
            output = '1'
        else:
            output = '0'

        return [ condition, output ]
