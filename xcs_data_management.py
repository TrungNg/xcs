"""
Name:        xcs_data_management.py
Authors:     Bao Trung
Contact:     baotrung@ecs.vuw.ac.nz
Created:     July, 2017
Description:
---------------------------------------------------------------------------------------------------------------------------------------------------------
XCS: Michigan-style Learning Classifier System - A LCS for Reinforcement Learning.  This XCS follows the version descibed in "An Algorithmic Description of XCS" published by Martin Butz and Stewart Wilson (2002).

---------------------------------------------------------------------------------------------------------------------------------------------------------
"""

#Import Required Modules---------------
from xcs_constants import cons
#import crandom as random
import random
#import sys
#--------------------------------------

class DataManagement:
    def __init__(self, train_file, test_file):
        #Initialize global variables-------------------------------------------------
        self.numb_attributes = None       # The number of attributes in the input file.
        self.has_ID_column = False     # Does the dataset contain a column of Instance IDs? (If so, it will not be included as an attribute)
        self.instanceID_ref = None       # The column reference for Instance IDs
        self.action_ref = None        # The column reference for the Class/Phenotype column
        self.discrete_action = True   # Is the Class/Phenotype Discrete? (False = Continuous)
        self.attribute_info = []         # Stores Discrete (0) or Continuous (1) for each attribute
        self.action_list = []         # Stores all possible discrete phenotype states/classes or maximum and minimum values for a continuous phenotype
        self.action_range = None      # Stores the difference between the maximum and minimum values for a continuous phenotype

        #Train/Test Specific-----------------------------------------------------------------------------
        self.train_headers = []       # The dataset column headers for the training data
        self.test_headers = []        # The dataset column headers for the testing data
        self.numb_train_instances = None   # The number of instances in the training data
        self.numb_test_instances = None    # The number of instances in the testing data

        print("----------------------------------------------------------------------------")
        print("XCS Code Demo:")
        print("----------------------------------------------------------------------------")
        print("Environment: Formatting Data... ")

        #Detect Features of training data--------------------------------------------------------------------------
        raw_train_data = self.loadData(train_file, True) #Load the raw data.

        self.characterizeDataset(raw_train_data)  #Detect number of attributes, instances, and reference locations.

        if cons.test_file == 'None': #If no testing data is available, formatting relies solely on training data.
            for_formatting = raw_train_data
        else:
            raw_test_data = self.loadData(test_file, False) #Load the raw data.
            self.compareDataset(raw_test_data) #Ensure that key features are the same between training and testing datasets.
            for_formatting = raw_train_data + raw_test_data #Merge Training and Testing datasets

        self.discriminatePhenotype(for_formatting) #Determine if endpoint/phenotype is discrete or continuous.
        if self.discrete_action:
            self.discriminateClasses(for_formatting) #Detect number of unique phenotype identifiers.
        else:
            self.characterizePhenotype(for_formatting)

        self.discriminateAttributes(for_formatting) #Detect whether attributes are discrete or continuous.
        self.characterizeAttributes(for_formatting) #Determine potential attribute states or ranges.

        #Format and Shuffle Datasets----------------------------------------------------------------------------------------
        if cons.test_file != 'None':
            self.formatted_test_data = self.formatData(raw_test_data) #Stores the formatted testing data set used throughout the algorithm.

        self.formatted_train_data = self.formatData(raw_train_data) #Stores the formatted training data set used throughout the algorithm.
        print("----------------------------------------------------------------------------")


    def loadData(self, dat_file, is_train):
        """ Load the data file. """
        print("DataManagement: Loading Data... " + str(dat_file))
        read_data = []
        try:
            f = open(dat_file,'r')
        except Exception as inst:
            print(type(inst))
            print(inst.args)
            print(inst)
            print('cannot open', dat_file)
            raise
        else:
            if is_train:
                self.train_headers = f.readline().rstrip('\n').split('\t')   #strip off first row
            else:
                self.test_headers = f.readline().rstrip('\n').split('\t')   #strip off first row
            for line in f:
                line_elements = line.strip('\n').split('\t')
                read_data.append(line_elements)
            f.close()

        return read_data


    def characterizeDataset(self, raw_train_data):
        " Detect basic dataset parameters "
        #Detect Instance ID's and save location if they occur.  Then save number of attributes in data.
        if cons.ID_label in self.train_headers:
            self.has_ID_column = True
            self.instanceID_ref = self.train_headers.index(cons.ID_label)
            print("DataManagement: Instance ID Column location = "+str(self.instanceID_ref))
            self.numb_attributes = len(self.train_headers)-2 #one column for InstanceID and another for the phenotype.
        else:
            self.numb_attributes = len(self.train_headers)-1

        #Identify location of phenotype column
        if cons.class_label in self.train_headers:
            self.action_ref = self.train_headers.index(cons.class_label)
            print("DataManagement: Phenotype Column Location = "+str(self.action_ref))
        else:
            print("DataManagement: Error - Phenotype column not found!  Check data set to ensure correct phenotype column label, or inclusion in the data.")

        #Adjust training header list to just include attributes labels
        if self.has_ID_column:
            if self.action_ref > self.instanceID_ref:
                self.train_headers.pop(self.action_ref)
                self.train_headers.pop(self.instanceID_ref)
            else:
                self.train_headers.pop(self.instanceID_ref)
                self.train_headers.pop(self.action_ref)
        else:
            self.train_headers.pop(self.action_ref)

        #Store number of instances in training data
        self.numb_train_instances = len(raw_train_data)
        print("DataManagement: Number of Attributes = " + str(self.numb_attributes))
        print("DataManagement: Number of Instances = " + str(self.numb_train_instances))


    def discriminatePhenotype(self, raw_data):
        """ Determine whether the phenotype is Discrete(class-based) or Continuous """
        print("DataManagement: Analyzing Phenotype...")
        inst = 0
        class_dict = {}
        while self.discrete_action and len(list(class_dict.keys())) <= cons.discrete_attribute_limit and inst < self.numb_train_instances:  #Checks which discriminate between discrete and continuous attribute
            target = raw_data[inst][self.action_ref]
            if target in list(class_dict.keys()):  #Check if we've seen this attribute state yet.
                class_dict[target] += 1
            elif target == cons.missing_label: #Ignore missing data
                print("DataManagement: Warning - Individual detected with missing phenotype information!")
                pass
            else: #New state observed
                class_dict[target] = 1
            inst += 1

        if len(list(class_dict.keys())) > cons.discrete_attribute_limit:
            self.discrete_action = False
            self.action_list = [float(target),float(target)]
            print("DataManagement: Phenotype Detected as Continuous.")
        else:
            print("DataManagement: Phenotype Detected as Discrete.")


    def discriminateClasses(self, raw_data):
        """ Determines number of classes and their identifiers. Only used if phenotype is discrete. """
        print("DataManagement: Detecting Classes...")
        inst = 0
        class_cnt = {}
        while inst < self.numb_train_instances:
            target = raw_data[inst][self.action_ref]
            if target in self.action_list:
                class_cnt[target] += 1
            else:
                self.action_list.append(target)
                class_cnt[target] = 1
            inst += 1
        print("DataManagement: Following Classes Detected:" + str(self.action_list))
        for each in list(class_cnt.keys()):
            print("Class: "+str(each)+ " count = "+ str(class_cnt[each]))


    def compareDataset(self, raw_test_data):
        " Ensures that the attributes in the testing data match those in the training data.  Also stores some information about the testing data. "
        if self.has_ID_column:
            if self.action_ref > self.instanceID_ref:
                self.test_headers.pop(self.action_ref)
                self.test_headers.pop(self.instanceID_ref)
            else:
                self.test_headers.pop(self.instanceID_ref)
                self.test_headers.pop(self.action_ref)
        else:
            self.test_headers.pop(self.action_ref)

        if self.train_headers != self.test_headers:
            print("DataManagement: Error - Training and Testing Dataset Headers are not equivalent")

        # Stores the number of instances in the testing data.
        self.numb_test_instances = len(raw_test_data)
        print("DataManagement: Number of Attributes = " + str(self.numb_attributes))
        print("DataManagement: Number of Instances = " + str(self.numb_test_instances))


    def discriminateAttributes(self, raw_data):
        """ Determine whether attributes in dataset are discrete or continuous and saves this information. """
        print("DataManagement: Detecting Attributes...")
        self.discrete_cnt = 0
        self.continuous_cnt = 0
        for att in range(len(raw_data[0])):
            if att != self.instanceID_ref and att != self.action_ref:  #Get just the attribute columns (ignores phenotype and instanceID columns)
                attIsDiscrete = True
                inst = 0
                state_dict = {}
                while attIsDiscrete and len(list(state_dict.keys())) <= cons.discrete_attribute_limit and inst < self.numb_train_instances:  #Checks which discriminate between discrete and continuous attribute
                    target = raw_data[inst][att]
                    if target in list(state_dict.keys()):  #Check if we've seen this attribute state yet.
                        state_dict[target] += 1
                    elif target == cons.missing_label: #Ignore missing data
                        pass
                    else: #New state observed
                        state_dict[target] = 1
                    inst += 1

                if len(list(state_dict.keys())) > cons.discrete_attribute_limit:
                    attIsDiscrete = False
                if attIsDiscrete:
                    self.attribute_info.append([0,[]])
                    self.discrete_cnt += 1
                else:
                    self.attribute_info.append([1,[float(target),float(target)]])   #[min,max]
                    self.continuous_cnt += 1
        print("DataManagement: Identified "+str(self.discrete_cnt)+" discrete and "+str(self.continuous_cnt)+" continuous attributes.") #Debug


    def characterizeAttributes(self, raw_data):
        """ Determine range (if continuous) or states (if discrete) for each attribute and saves this information"""
        print("DataManagement: Characterizing Attributes...")
        attributeID = 0
        for att in range(len(raw_data[0])):
            if att != self.instanceID_ref and att != self.action_ref:  #Get just the attribute columns (ignores phenotype and instanceID columns)
                for inst in range(len(raw_data)):
                    target = raw_data[inst][att]
                    if not self.attribute_info[attributeID][0]: #If attribute is discrete
                        if target in self.attribute_info[attributeID][1] or target == cons.missing_label:
                            pass  #NOTE: Could potentially store state frequency information to guide learning.
                        else:
                            self.attribute_info[attributeID][1].append(target)
                    else: #If attribute is continuous

                        #Find Minimum and Maximum values for the continuous attribute so we know the range.
                        if target == cons.missing_label:
                            pass
                        elif float(target) > self.attribute_info[attributeID][1][1]:  #error
                            self.attribute_info[attributeID][1][1] = float(target)
                        elif float(target) < self.attribute_info[attributeID][1][0]:
                            self.attribute_info[attributeID][1][0] = float(target)
                        else:
                            pass
                attributeID += 1


    def characterizePhenotype(self, raw_data):
        """ Determine range of phenotype values. """
        print("DataManagement: Characterizing Phenotype...")
        for inst in range(len(raw_data)):
            target = raw_data[inst][self.action_ref]

            #Find Minimum and Maximum values for the continuous phenotype so we know the range.
            if target == cons.missing_label:
                pass
            elif float(target) > self.action_list[1]:
                self.action_list[1] = float(target)
            elif float(target) < self.action_list[0]:
                self.action_list[0] = float(target)
            else:
                pass
        self.action_range = self.action_list[1] - self.action_list[0]


    def formatData(self,raw_data):
        """ Get the data into a format convenient for the algorithm to interact with. Specifically each instance is stored in a list as follows; [Attribute States, Phenotype, InstanceID] """
        formatted = []
        #Initialize data format---------------------------------------------------------
        for _ in range(len(raw_data)):
            formatted.append([None,None,None]) #[Attribute States, Phenotype, InstanceID]

        for inst in range(len(raw_data)):
            state_list = []
            attributeID = 0
            for att in range(len(raw_data[0])):
                if att != self.instanceID_ref and att != self.action_ref:  #Get just the attribute columns (ignores phenotype and instanceID columns)
                    target = raw_data[inst][att]

                    if self.attribute_info[attributeID][0]: #If the attribute is continuous
                        if target == cons.missing_label:
                            state_list.append(target) #Missing data saved as text label
                        else:
                            state_list.append(float(target)) #Save continuous data as floats.
                    else: #If the attribute is discrete - Format the data to correspond to the GABIL (DeJong 1991)
                        state_list.append(target) #missing data, and discrete variables, all stored as string objects
                    attributeID += 1

            #Final Format-----------------------------------------------
            formatted[inst][0] = state_list                           #Attribute states stored here
            if self.discrete_action:
                formatted[inst][1] = raw_data[inst][self.action_ref]        #phenotype stored here
            else:
                formatted[inst][1] = float(raw_data[inst][self.action_ref])
            if self.has_ID_column:
                formatted[inst][2] = raw_data[inst][self.instanceID_ref]   #Instance ID stored here
            else:
                pass    #instance ID neither given nor required.
            #-----------------------------------------------------------
        random.shuffle(formatted) #One time randomization of the order the of the instances in the data, so that if the data was ordered by phenotype, this potential learning bias (based on instance ordering) is eliminated.
        return formatted
