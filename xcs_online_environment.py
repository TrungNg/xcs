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
#import crandom as random
import random
#-------------------------------------------------------------


class Online_Environment:
    def __init__(self, problem, sizes):
        #Initialize global variables-------------------------------------------------
        self.data_ref = 0
        self.saved_dat_ref = 0
        options={ 'multiplexer':MultiplexerGenerator,
                  'even_parity':EvenParityGenerator,
                  'carry':CarryGenerator,
                  'majorityon':MajorityOnGenerator,
                  'hidden_multiplexer':HiddenMultiplexer,
                  'hidden_parity':HiddenParityGenerator,
                  'hidden_carry':HiddenCarryGenerator,
                  'hidden_majorityon':HiddenMajorityOn }
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
        self.action_list = [ 0, 1 ]


class MultiplexerGenerator( DataGenerator ):
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
        condition = [None] * self.numb_attributes
        #Generate random boolean string
        for i in range( self.numb_attributes ):
            condition[i] = random.randint( 0, 1 )
        gates = 0
        for j in range( self.address_size ):
            gates *= 2
            gates += condition[j]
        output = condition[ self.address_size + gates ]
        return [ condition, output ]

    def _findAddressSize(self, num_bits):
        for i in range( 1000 ):
            if i + 2**i == num_bits:
                return i
            if i + 2**i > num_bits:
                break
        return None


class MajorityOnGenerator( DataGenerator ):
    def __init__(self, sizes):
        """ Initialize online data generator for Multiplexer problem (maximum address bit size is 1000). """
        self.headers = []
        for i in range( sizes[0] ):
            self.headers.append( 'B' + str(i) )
        super().__init__( sizes )

    def generateInstance(self):
        """ Return new Multiplexer instance of size provided by generating randomly. """
        condition = [None] * self.numb_attributes
        #Generate random boolean string
        for i in range( self.numb_attributes ):
            condition[i] = random.randint(0, 1)
        #Find output for generated condition
        count = sum( condition )
        return [ condition, int( count > self.numb_attributes / 2 ) ]


class CarryGenerator( DataGenerator ):
    def __init__(self, sizes):
        """ Initialize online data generator for Carry problem. """
        if sizes[0] % 2 == 1:
            raise ValueError("illegal size for the Carry Problem")
        self.half_length = int( sizes[0]/2 )
        self.headers = [None] * sizes[0]
        for i in range( self.half_length ):
            self.headers[i] = 'B' + str(i)
            self.headers[i + self.half_length] = 'A' + str(i)
        super().__init__( sizes )

    def generateInstance(self):
        """ Return new Multiplexer instance of size provided by generating randomly. """
        """ Return new Multiplexer instance of size provided by generating randomly. """
        condition = [None] * self.numb_attributes
        #Generate random boolean string
        for i in range( self.numb_attributes ):
            condition[i] = random.randint( 0, 1 )
        #Find output for generated condition
        for i in range(0, self.half_length):
            #output = int( ( output + cond_int[ half_condition-1-i ] + cond_int[ half_condition-1-i+half_condition ] ) / 2 )
            if condition[i] == condition[ i+self.half_length ]:
                return [ condition, condition[i] ]
        return [ condition, 0 ]


class HiddenCarryGenerator( DataGenerator ):
    def __init__(self, sizes):
        """ Initialize online data generator for Carry problem. """
        self.parity_size = sizes[ 1 ]
        if sizes[2] % 2 == 1:
            raise ValueError("illegal size for the Carry Problem")
        self.half_length = int( sizes[2]/2 )
        self.headers = [None] * sizes[0]
        for i in range( self.half_length ):
            for j in range( self.parity_size ):
                self.headers[ i*self.parity_size ] = 'A' + str(i) + str(j)
                self.headers[ (i + self.half_length)*self.parity_size ] = 'B' + str(i) + str(j)
        super().__init__( sizes )

    def generateInstance(self):
        """ Return new Multiplexer instance of size provided by generating randomly. """
        condition = [None] * self.numb_attributes
        #Generate random boolean string
        for i in range( self.numb_attributes ):
            condition[i] = random.randint(0, 1)
        #Find output for generated condition
        #output = 0
        for i in range(0, self.half_length):
            parity_block_a = i*self.parity_size
            parity_block_b = (i+self.half_length)*self.parity_size
            a = sum( condition[ parity_block_a:( parity_block_a+self.parity_size ) ] )
            b = sum( condition[ parity_block_b:( parity_block_b+self.parity_size ) ] )
            a = 1-a%2
            b = 1-b%2
            if a == b:
                return [ condition, a ]
        return [ condition, 0 ]


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
        condition = [None] * self.numb_attributes
        #Generate random boolean string
        for i in range( self.numb_attributes ):
            condition[i] = random.randint(0, 1)
        gates = 0
        for i in range( self.multiplexer_addr_size ):
            parity_block = i * self.parity_size
            count = sum( condition[ parity_block:( parity_block+self.parity_size ) ] )
            gates *= 2
            gates += 1 - count%2
        output_parity_block = ( self.multiplexer_addr_size+gates ) * self.parity_size
        count = sum( condition[ output_parity_block:( output_parity_block+self.parity_size ) ] )
        return [ condition, int( count%2 == 0 ) ]

    def _findAddressSize(self, num_bits):
        for i in range( 1000 ):
            if i + 2**i == num_bits:
                return i
            if i + 2**i > num_bits:
                break
        return None


class HiddenMajorityOn( DataGenerator ):
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
        condition = [None] * self.numb_attributes
        #Generate random boolean string
        for i in range( self.parity_size * self.countone_size ):
            condition[i] = random.randint(0, 1)
        parity_counts = [0] * self.countone_size
        for j in range( self.countone_size ):
            parity_block = j*self.parity_size
            parity_counts[j] = sum( condition[ parity_block:( parity_block+self.parity_size ) ] )
            parity_counts[j] = 1 - parity_counts[j] % 2
        return [ condition, int( sum( parity_counts ) > self.countone_size / 2 ) ]


class HiddenParityGenerator( DataGenerator ):
    def __init__(self, sizes):
        """ Initialize Hidden Parity, lowest level are blocks of data with same size (3).
        Their outputs are calculated by parity rule and then the outputs contribute to a higher-level Parity problem. """
        self.level_sizes = sizes[1:]
        self.headers = []
        if self.level_sizes[0] * self.level_sizes[1] != sizes[ 0 ]:
            raise ValueError( "ParityCountOne.__init__() failed because of inappropriate sizes provided." )
        for i in range( self.level_sizes[1] ):
            prefix = 'B' + str( i )
            for j in range( self.level_sizes[0] ):
                self.headers.append( prefix + str( j ) )
        super().__init__( sizes )

    def generateInstance(self):
        """ Generate and return new instance with correct output. """
        condition = [None] * self.numb_attributes
        #Generate random boolean string
        for i in range( self.level_sizes[0] * self.level_sizes[1] ):
            condition[i] = random.randint(0, 1)
        parity_counts = [0] * self.level_sizes[1]
        for j in range( self.level_sizes[1] ):
            parity_block = j*self.level_sizes[0]
            parity_counts[j] = 1 - sum( condition[ parity_block:( parity_block+self.level_sizes[0] ) ] ) % 2
        return [ condition, 1 - sum( parity_counts ) % 2 ]


class EvenParityGenerator( DataGenerator ):
    def __init__(self, sizes):
        """ Initialize online data generator for Multiplexer problem (maximum address bit size is 1000). """
        self.headers = []
        for i in range( sizes[0] ):
            self.headers.append( 'B' + str(i) )
        super().__init__(sizes)

    def generateInstance(self):
        """ Return new Multiplexer instance of size provided by generating randomly. """
        condition = [0] * self.numb_attributes
        count = 0
        #Generate random boolean string
        for i in range( self.numb_attributes ):
            condition[i] = random.randint(0,1)
        for j in range( self.numb_attributes ):
            count += condition[j]
        return [ condition, int( count%2 == 0 ) ]
