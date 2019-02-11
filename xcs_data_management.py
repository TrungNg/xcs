"""
Name:        xcs_data_management.py
Authors:     Bao Trung, based on eLCS by R. Urbanowicz
Contact:     baotrung@ecs.vuw.ac.nz
Created:     July, 2017
Description:
---------------------------------------------------------------------------------------------------------------------------------------------------------
XCS: Michigan-style Learning Classifier System - A LCS for Reinforcement Learning.  This XCS follows the version descibed in "An Algorithmic Description of XCS" published by Martin Butz and Stewart Wilson (2002).

---------------------------------------------------------------------------------------------------------------------------------------------------------
"""
#Import Required Modules---------------
from javarandom import Random as JRandom
import random
from xcs_constants import cons
#--------------------------------------

class DataManagement:
    def __init__(self, train_file, test_file):
        #Initialize global variables-------------------------------------------------
        self.jrnd = JRandom(0)
        self.numb_attributes = None       # The number of attributes in the input file.
        self.are_instanceIDs = False     # Does the dataset contain a column of Instance IDs? (If so, it will not be included as an attribute)
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
            tobe_formatted = raw_train_data
        else:
            raw_test_data = self.loadData(test_file, False) #Load the raw data.
            self.compareDataset(raw_test_data) #Ensure that key features are the same between training and testing datasets.
            tobe_formatted = raw_train_data + raw_test_data #Merge Training and Testing datasets

        self.discriminatePhenotype(tobe_formatted) #Determine if endpoint/phenotype is discrete or continuous.
        if self.discrete_action:
            self.discriminateClasses(tobe_formatted) #Detect number of unique phenotype identifiers.
        else:
            self.characterizePhenotype(tobe_formatted)

        self.discriminateAttributes(tobe_formatted) #Detect whether attributes are discrete or continuous.
        self.characterizeAttributes(tobe_formatted) #Determine potential attribute states or ranges.

        #Format and Shuffle Datasets----------------------------------------------------------------------------------------
        if cons.test_file != 'None':
            self.formatted_test_data = self.formatData(raw_test_data) #Stores the formatted testing data set used throughout the algorithm.

        self.formatted_train_data = self.formatData(raw_train_data) #Stores the formatted training data set used throughout the algorithm.
        print("----------------------------------------------------------------------------")


    def loadData(self, dat_file, do_train):
        """ Load the data file. """
        print("DataManagement: Loading Data... " + str(dat_file))
        dataset_list = []
        try:
            f = open(dat_file,'r')
        except Exception as inst:
            print(type(inst))
            print(inst.args)
            print(inst)
            print('cannot open', dat_file)
            raise
        else:
            if do_train:
                self.train_headers = f.readline().rstrip('\n').split('\t')   #strip off first row
            else:
                self.test_headers = f.readline().rstrip('\n').split('\t')   #strip off first row
            for line in f:
                line_list = line.strip('\n').split('\t')
                dataset_list.append(line_list)
            f.close()

        return dataset_list


    def characterizeDataset(self, raw_train_data):
        " Detect basic dataset parameters "
        #Detect Instance ID's and save location if they occur.  Then save number of attributes in data.
        if cons.ID_label in self.train_headers:
            self.are_instanceIDs = True
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
        if self.are_instanceIDs:
            if self.action_ref > self.instanceID_ref:
                self.train_headers.pop(self.action_ref)
                self.train_headers.pop(self.instanceID_ref)
            else:
                self.train_headers.pop(self.instanceID_ref)
                self.train_headers.pop(self.action_ref)
        else:
            self.train_headers.pop(self.action_ref)

        #Store number of instances in training data
        print("DataManagement: Number of Attributes = " + str(self.numb_attributes))
        self.numb_train_instances = len(raw_train_data)
        if cons.kfold_cv == False:
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
        class_count = {}
        while inst < self.numb_train_instances:
            target = raw_data[inst][self.action_ref]
            if int(target) in self.action_list:
                class_count[target] += 1
            else:
                self.action_list.append( int(target) )
                class_count[target] = 1
            inst += 1
        print("DataManagement: Following Classes Detected:" + str(self.action_list))
        for each in list(class_count.keys()):
            print("Class: "+str(each)+ " count = "+ str(class_count[each]))


    def compareDataset(self, raw_test_data):
        " Ensures that the attributes in the testing data match those in the training data.  Also stores some information about the testing data. "
        if self.are_instanceIDs:
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
        self.discrete_count = 0
        self.continuous_count = 0
        for att in range(len(raw_data[0])):
            if att != self.instanceID_ref and att != self.action_ref:  #Get just the attribute columns (ignores phenotype and instanceID columns)
                is_att_discrete = True
                inst = 0
                state_dict = {}
                while is_att_discrete and len(list(state_dict.keys())) <= cons.discrete_attribute_limit and inst < self.numb_train_instances:  #Checks which discriminate between discrete and continuous attribute
                    target = raw_data[inst][att]
                    if target in list(state_dict.keys()):  #Check if we've seen this attribute state yet.
                        state_dict[target] += 1
                    elif target == cons.missing_label: #Ignore missing data
                        pass
                    else: #New state observed
                        state_dict[target] = 1
                    inst += 1

                if len(list(state_dict.keys())) > cons.discrete_attribute_limit:
                    is_att_discrete = False
                if is_att_discrete:
                    self.attribute_info.append([0,[]])
                    self.discrete_count += 1
                else:
                    self.attribute_info.append([1,[float(target),float(target)]])   #[min,max]
                    self.continuous_count += 1
        print("DataManagement: Identified "+str(self.discrete_count)+" discrete and "+str(self.continuous_count)+" continuous attributes.") #Debug


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
            state_list = [0] * self.numb_attributes
            attributeID = 0
            for att in range(len(raw_data[0])):
                if att != self.instanceID_ref and att != self.action_ref:  #Get just the attribute columns (ignores phenotype and instanceID columns)
                    target = raw_data[inst][att]

                    if self.attribute_info[attributeID][0]: #If the attribute is continuous
                        if target == cons.missing_label:
                            state_list[attributeID] = target #Missing data saved as text label
                        else:
                            state_list[attributeID] = float(target) #Save continuous data as floats.
                    else: #If the attribute is discrete - Format the data to correspond to the GABIL (DeJong 1991)
                        if target == cons.missing_label:
                            state_list[attributeID] = target
                        else:
                            state_list[attributeID] = int(target) #missing data, and discrete variables, all stored as string objects
                    attributeID += 1

            #Final Format-----------------------------------------------
            formatted[inst][0] = state_list                           #Attribute states stored here
            if self.discrete_action:
                formatted[inst][1] = int( raw_data[inst][self.action_ref] )       #phenotype stored here
            else:
                formatted[inst][1] = float( raw_data[inst][self.action_ref] )
            if self.are_instanceIDs:
                formatted[inst][2] = int( raw_data[inst][self.instanceID_ref] )   #Instance ID stored here
            else:
                pass    #instance ID neither given nor required.
            #-----------------------------------------------------------
        #random.shuffle(formatted) #One time randomization of the order the of the instances in the data, so that if the data was ordered by phenotype, this potential learning bias (based on instance ordering) is eliminated.
        randomize( formatted, self.jrnd )
        return formatted

    def splitFolds(self, kfold=10):
        """ divide data set into kfold sets. """
        self.formatted_train_data = stratify( self.formatted_train_data, kfold )
        data_size = len( self.formatted_train_data )
        self.folds = [ [] for _ in range(kfold) ]
        for fold_id in range(kfold):
            fold_size = int( data_size/kfold )
            if fold_id < data_size % kfold:
                fold_size += 1
                offset = fold_id
            else:
                offset = data_size % kfold
            first = fold_id * ( int( data_size/kfold ) ) + offset
            self.folds[fold_id] = self.formatted_train_data[ first : ( first+fold_size ) ]

    def splitData(self):
        """ divide data set into kfold sets. """
        class_counts = [0] * len( self.action_list )
        for instance in self.formatted_train_data:
            class_counts[ self.action_list.index( instance[1] ) ] += 1
        training_sizes_for_actions = [0] * len( self.action_list )
        for i in range( len(self.action_list) ):
            training_sizes_for_actions[i] = int( class_counts[i] * cons.training_portion + 0.5 )
        numb_instances_for_actions = [0] * len(self.action_list)
        train_data = []
        test_data = []
        for instance in self.formatted_train_data:
            action_index = self.action_list.index( instance[1] )
            if numb_instances_for_actions[action_index] < training_sizes_for_actions[action_index]:
                train_data.append(instance)
                numb_instances_for_actions[action_index] += 1
            else:
                test_data.append(instance)
        return train_data, test_data

    def splitData2(self):
        """ divide data set into kfold sets. """
        num_train_folds = cons.training_portion * cons.kfold
        if num_train_folds != int( num_train_folds ):
            train_data, test_data = self.splitData()
        else:
            self.splitFolds( cons.kfold )
            train_data = []
            test_data = []
            for i in range( cons.kfold ):
                if i < num_train_folds:
                    train_data += self.folds[i]
                else:
                    test_data += self.folds[i]
        self.formatted_train_data = train_data
        self.formatted_test_data = test_data
        self.numb_train_instances = len( self.formatted_train_data )
        self.numb_test_instances = len( self.formatted_test_data )
        print("DataManagement: Number of Training Instances = " + str( self.numb_train_instances ))
        print("DataManagement: Number of Testing Instances = " + str( self.numb_test_instances ))

    def selectTrainTestSets(self, fold_id):
        """ select one fold for testing and the rest for training (k-fold cross validation. """
        self.formatted_train_data = []
        for i in range( cons.kfold ):
            if i != fold_id:
                self.formatted_train_data += self.folds[i]
        randomize( self.formatted_train_data, self.jrnd )
        self.formatted_test_data = self.folds[fold_id]
        self.numb_train_instances = len(self.formatted_train_data)
        self.numb_test_instances = len(self.formatted_test_data)
        print("DataManagement: Number of Instances = " + str(self.numb_train_instances))
        print("DataManagement: Number of Instances = " + str(self.numb_test_instances))


def stratify(all_data, kfold=10):
    """ divide data set into kfold sets. """
    # sort by class
    index = 1
    numb_instances = len(all_data)
    while index < numb_instances:
        instance1 = all_data[index - 1]
        for j in range( index, numb_instances ):
            instance2 = all_data[j]
            if instance1[1] == instance2[1]:
                #swap(index, j)
                temp = all_data[index]
                all_data[index] = all_data[j]
                all_data[j] = temp
                index += 1
        index += 1
    # rearrange classes to kfold trunks.
    stratified_data = []
    start = 0
    while len(stratified_data) < numb_instances:
        j = start
        while j < numb_instances:
            stratified_data.append( all_data[j] )
            j += kfold
        start += 1
    return stratified_data


def randomize( formatted_data, javarandom ):
    """ shuffle data """
    for i in range( len(formatted_data)-1, 0, -1 ):
        temp = formatted_data[i]
        j = javarandom.nextInt(i + 1)
        formatted_data[i] = formatted_data[j]
        formatted_data[j] = temp
