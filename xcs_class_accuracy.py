"""
Name:        xcs_class_accuracy.py
Authors:     Bao Trung
Contact:     baotrung@ecs.vuw.ac.nz
Created:     July, 2017
Description:
---------------------------------------------------------------------------------------------------------------------------------------------------------

---------------------------------------------------------------------------------------------------------------------------------------------------------
"""

class ClassAccuracy:
    def __init__(self):
        """ Initialize the accuracy calculation for a single class """
        self.T_myClass = 0      #For binary class problems this would include true positives
        self.T_otherClass = 0   #For binary class problems this would include true negatives
        self.F_myClass = 0      #For binary class problems this would include false positives
        self.F_otherClass = 0   #For binary class problems this would include false negatives


    def updateAccuracy(self, my_class, is_correct):
        """ Increment the appropriate cell of the confusion matrix """
        if my_class and is_correct:
            self.T_myClass += 1
        elif is_correct:
            self.T_otherClass += 1
        elif my_class:
            self.F_myClass += 1
        else:
            self.F_otherClass += 1


    def reportClassAccuracy(self):
        """ Print to standard out, summary on the class accuracy. """
        print("-----------------------------------------------")
        print("TP = "+str(self.T_myClass))
        print("TN = "+str(self.T_otherClass))
        print("FP = "+str(self.F_myClass))
        print("FN = "+str(self.F_otherClass))
        print("-----------------------------------------------")