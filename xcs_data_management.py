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
from xcs_constants import *
import random
import sys
#--------------------------------------

class DataManagement:
    def __init__(self, trainFile, testFile, infoList = None):
        #Set random seed if specified.-----------------------------------------------
        if cons.useSeed:
            random.seed(cons.randomSeed)
        else:
            random.seed(None)

        #Initialize global variables-------------------------------------------------
        self.numb_attributes = None       # The number of attributes in the input file.
        self.areInstanceIDs = False     # Does the dataset contain a column of Instance IDs? (If so, it will not be included as an attribute)
        self.instanceIDRef = None       # The column reference for Instance IDs
        self.phenotypeRef = None        # The column reference for the Class/Phenotype column
        self.discrete_action = True   # Is the Class/Phenotype Discrete? (False = Continuous)
        self.attribute_info = []         # Stores Discrete (0) or Continuous (1) for each attribute
        self.action_list = []         # Stores all possible discrete phenotype states/classes or maximum and minimum values for a continuous phenotype
        self.phenotypeRange = None      # Stores the difference between the maximum and minimum values for a continuous phenotype

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
        rawTrainData = self.loadData(trainFile, True) #Load the raw data.

        self.characterizeDataset(rawTrainData)  #Detect number of attributes, instances, and reference locations.

        if cons.testFile == 'None': #If no testing data is available, formatting relies solely on training data.
            data4Formating = rawTrainData
        else:
            rawTestData = self.loadData(testFile, False) #Load the raw data.
            self.compareDataset(rawTestData) #Ensure that key features are the same between training and testing datasets.
            data4Formating = rawTrainData + rawTestData #Merge Training and Testing datasets

        self.discriminatePhenotype(data4Formating) #Determine if endpoint/phenotype is discrete or continuous.
        if self.discrete_action:
            self.discriminateClasses(data4Formating) #Detect number of unique phenotype identifiers.
        else:
            self.characterizePhenotype(data4Formating)

        self.discriminateAttributes(data4Formating) #Detect whether attributes are discrete or continuous.
        self.characterizeAttributes(data4Formating) #Determine potential attribute states or ranges.

        #Format and Shuffle Datasets----------------------------------------------------------------------------------------
        if cons.testFile != 'None':
            self.testFormatted = self.formatData(rawTestData) #Stores the formatted testing data set used throughout the algorithm.

        self.trainFormatted = self.formatData(rawTrainData) #Stores the formatted training data set used throughout the algorithm.
        print("----------------------------------------------------------------------------")


    def loadData(self, dataFile, doTrain):
        """ Load the data file. """
        print("DataManagement: Loading Data... " + str(dataFile))
        datasetList = []
        try:
            f = open(dataFile,'r')
        except Exception as inst:
            print(type(inst))
            print(inst.args)
            print(inst)
            print('cannot open', dataFile)
            raise
        else:
            if doTrain:
                self.train_headers = f.readline().rstrip('\n').split('\t')   #strip off first row
            else:
                self.test_headers = f.readline().rstrip('\n').split('\t')   #strip off first row
            for line in f:
                lineList = line.strip('\n').split('\t')
                datasetList.append(lineList)
            f.close()

        return datasetList


    def characterizeDataset(self, rawTrainData):
        " Detect basic dataset parameters "
        #Detect Instance ID's and save location if they occur.  Then save number of attributes in data.
        if cons.labelInstanceID in self.train_headers:
            self.areInstanceIDs = True
            self.instanceIDRef = self.train_headers.index(cons.labelInstanceID)
            print("DataManagement: Instance ID Column location = "+str(self.instanceIDRef))
            self.numb_attributes = len(self.train_headers)-2 #one column for InstanceID and another for the phenotype.
        else:
            self.numb_attributes = len(self.train_headers)-1

        #Identify location of phenotype column
        if cons.labelPhenotype in self.train_headers:
            self.phenotypeRef = self.train_headers.index(cons.labelPhenotype)
            print("DataManagement: Phenotype Column Location = "+str(self.phenotypeRef))
        else:
            print("DataManagement: Error - Phenotype column not found!  Check data set to ensure correct phenotype column label, or inclusion in the data.")

        #Adjust training header list to just include attributes labels
        if self.areInstanceIDs:
            if self.phenotypeRef > self.instanceIDRef:
                self.train_headers.pop(self.phenotypeRef)
                self.train_headers.pop(self.instanceIDRef)
            else:
                self.train_headers.pop(self.instanceIDRef)
                self.train_headers.pop(self.phenotypeRef)
        else:
            self.train_headers.pop(self.phenotypeRef)

        #Store number of instances in training data
        self.numb_train_instances = len(rawTrainData)
        print("DataManagement: Number of Attributes = " + str(self.numb_attributes))
        print("DataManagement: Number of Instances = " + str(self.numb_train_instances))


    def discriminatePhenotype(self, rawData):
        """ Determine whether the phenotype is Discrete(class-based) or Continuous """
        print("DataManagement: Analyzing Phenotype...")
        inst = 0
        classDict = {}
        while self.discrete_action and len(list(classDict.keys())) <= cons.discreteAttributeLimit and inst < self.numb_train_instances:  #Checks which discriminate between discrete and continuous attribute
            target = rawData[inst][self.phenotypeRef]
            if target in list(classDict.keys()):  #Check if we've seen this attribute state yet.
                classDict[target] += 1
            elif target == cons.labelMissingData: #Ignore missing data
                print("DataManagement: Warning - Individual detected with missing phenotype information!")
                pass
            else: #New state observed
                classDict[target] = 1
            inst += 1

        if len(list(classDict.keys())) > cons.discreteAttributeLimit:
            self.discrete_action = False
            self.action_list = [float(target),float(target)]
            print("DataManagement: Phenotype Detected as Continuous.")
        else:
            print("DataManagement: Phenotype Detected as Discrete.")


    def discriminateClasses(self, rawData):
        """ Determines number of classes and their identifiers. Only used if phenotype is discrete. """
        print("DataManagement: Detecting Classes...")
        inst = 0
        classCount = {}
        while inst < self.numb_train_instances:
            target = rawData[inst][self.phenotypeRef]
            if target in self.action_list:
                classCount[target] += 1
            else:
                self.action_list.append(target)
                classCount[target] = 1
            inst += 1
        print("DataManagement: Following Classes Detected:" + str(self.action_list))
        for each in list(classCount.keys()):
            print("Class: "+str(each)+ " count = "+ str(classCount[each]))


    def compareDataset(self, rawTestData):
        " Ensures that the attributes in the testing data match those in the training data.  Also stores some information about the testing data. "
        if self.areInstanceIDs:
            if self.phenotypeRef > self.instanceIDRef:
                self.test_headers.pop(self.phenotypeRef)
                self.test_headers.pop(self.instanceIDRef)
            else:
                self.test_headers.pop(self.instanceIDRef)
                self.test_headers.pop(self.phenotypeRef)
        else:
            self.test_headers.pop(self.phenotypeRef)

        if self.train_headers != self.test_headers:
            print("DataManagement: Error - Training and Testing Dataset Headers are not equivalent")

        # Stores the number of instances in the testing data.
        self.numb_test_instances = len(rawTestData)
        print("DataManagement: Number of Attributes = " + str(self.numb_attributes))
        print("DataManagement: Number of Instances = " + str(self.numb_test_instances))


    def discriminateAttributes(self, rawData):
        """ Determine whether attributes in dataset are discrete or continuous and saves this information. """
        print("DataManagement: Detecting Attributes...")
        self.discreteCount = 0
        self.continuousCount = 0
        for att in range(len(rawData[0])):
            if att != self.instanceIDRef and att != self.phenotypeRef:  #Get just the attribute columns (ignores phenotype and instanceID columns)
                attIsDiscrete = True
                inst = 0
                stateDict = {}
                while attIsDiscrete and len(list(stateDict.keys())) <= cons.discreteAttributeLimit and inst < self.numb_train_instances:  #Checks which discriminate between discrete and continuous attribute
                    target = rawData[inst][att]
                    if target in list(stateDict.keys()):  #Check if we've seen this attribute state yet.
                        stateDict[target] += 1
                    elif target == cons.labelMissingData: #Ignore missing data
                        pass
                    else: #New state observed
                        stateDict[target] = 1
                    inst += 1

                if len(list(stateDict.keys())) > cons.discreteAttributeLimit:
                    attIsDiscrete = False
                if attIsDiscrete:
                    self.attribute_info.append([0,[]])
                    self.discreteCount += 1
                else:
                    self.attribute_info.append([1,[float(target),float(target)]])   #[min,max]
                    self.continuousCount += 1
        print("DataManagement: Identified "+str(self.discreteCount)+" discrete and "+str(self.continuousCount)+" continuous attributes.") #Debug


    def characterizeAttributes(self, rawData):
        """ Determine range (if continuous) or states (if discrete) for each attribute and saves this information"""
        print("DataManagement: Characterizing Attributes...")
        attributeID = 0
        for att in range(len(rawData[0])):
            if att != self.instanceIDRef and att != self.phenotypeRef:  #Get just the attribute columns (ignores phenotype and instanceID columns)
                for inst in range(len(rawData)):
                    target = rawData[inst][att]
                    if not self.attribute_info[attributeID][0]: #If attribute is discrete
                        if target in self.attribute_info[attributeID][1] or target == cons.labelMissingData:
                            pass  #NOTE: Could potentially store state frequency information to guide learning.
                        else:
                            self.attribute_info[attributeID][1].append(target)
                    else: #If attribute is continuous

                        #Find Minimum and Maximum values for the continuous attribute so we know the range.
                        if target == cons.labelMissingData:
                            pass
                        elif float(target) > self.attribute_info[attributeID][1][1]:  #error
                            self.attribute_info[attributeID][1][1] = float(target)
                        elif float(target) < self.attribute_info[attributeID][1][0]:
                            self.attribute_info[attributeID][1][0] = float(target)
                        else:
                            pass
                attributeID += 1


    def characterizePhenotype(self, rawData):
        """ Determine range of phenotype values. """
        print("DataManagement: Characterizing Phenotype...")
        for inst in range(len(rawData)):
            target = rawData[inst][self.phenotypeRef]

            #Find Minimum and Maximum values for the continuous phenotype so we know the range.
            if target == cons.labelMissingData:
                pass
            elif float(target) > self.action_list[1]:
                self.action_list[1] = float(target)
            elif float(target) < self.action_list[0]:
                self.action_list[0] = float(target)
            else:
                pass
        self.phenotypeRange = self.action_list[1] - self.action_list[0]


    def formatData(self,rawData):
        """ Get the data into a format convenient for the algorithm to interact with. Specifically each instance is stored in a list as follows; [Attribute States, Phenotype, InstanceID] """
        formatted = []
        #Initialize data format---------------------------------------------------------
        for i in range(len(rawData)):
            formatted.append([None,None,None]) #[Attribute States, Phenotype, InstanceID]

        for inst in range(len(rawData)):
            stateList = []
            attributeID = 0
            for att in range(len(rawData[0])):
                if att != self.instanceIDRef and att != self.phenotypeRef:  #Get just the attribute columns (ignores phenotype and instanceID columns)
                    target = rawData[inst][att]

                    if self.attribute_info[attributeID][0]: #If the attribute is continuous
                        if target == cons.labelMissingData:
                            stateList.append(target) #Missing data saved as text label
                        else:
                            stateList.append(float(target)) #Save continuous data as floats.
                    else: #If the attribute is discrete - Format the data to correspond to the GABIL (DeJong 1991)
                        stateList.append(target) #missing data, and discrete variables, all stored as string objects
                    attributeID += 1

            #Final Format-----------------------------------------------
            formatted[inst][0] = stateList                           #Attribute states stored here
            if self.discrete_action:
                formatted[inst][1] = rawData[inst][self.phenotypeRef]        #phenotype stored here
            else:
                formatted[inst][1] = float(rawData[inst][self.phenotypeRef])
            if self.areInstanceIDs:
                formatted[inst][2] = rawData[inst][self.instanceIDRef]   #Instance ID stored here
            else:
                pass    #instance ID neither given nor required.
            #-----------------------------------------------------------
        random.shuffle(formatted) #One time randomization of the order the of the instances in the data, so that if the data was ordered by phenotype, this potential learning bias (based on instance ordering) is eliminated.
        return formatted
