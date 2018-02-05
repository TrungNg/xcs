'''
Name: data_generator.py
Author: Bao-Trung Nguyen
Created: July 13, 2017
Description:  Module for generating datasets.
'''
import random


def generate_parity_countone_data(myfile, numb_parity,numb_countone, instances):
    """ """
    fp=open(myfile,"w")
    #Make File Header
    for i in range(numb_parity*numb_countone):
        fp.write('B_'+str(i)+"\t")
    fp.write("Class" + "\n") #State found at Register Bit

    for i in range(instances):
        state_phenotype = generate_parity_countone_instance(numb_parity,numb_countone)
        for j in state_phenotype[0]:
            fp.write(str(j)+"\t")
        fp.write(str(state_phenotype[1])+ "\n")


def generate_parity_countone_instance(numb_parity,numb_countone):
    """ """
    condition = []
    #Generate random boolean string
    for i in range(numb_parity*numb_countone):
        condition.append(str(random.randint(0,1)))

    counts=[]
    for j in range(numb_countone):
        counts.append(0)
        for k in range(numb_parity):
            if condition[j*numb_parity + k] == '1':
                counts[j]+=1
        if counts[j]%2 == 0:
            counts[j]=0
        else:
            counts[j]=1

    if sum(counts) > numb_countone/2:
        output='1'
    else:
        output='0'

    return [condition,output]


def generate_complete_parity_countone(myfile,numb_parity,numb_countone):
    length=numb_parity*numb_countone
    try:
        fp=open(myfile,"w")
        #Make File Header
        for i in range(length):
            fp.write('B_'+str(i)+"\t")
        fp.write("Class" + "\n") #State found at Register Bit

        for i in range(2**length):
            binary_str=bin(i)
            string_array=binary_str.split('b')
            binary=string_array[1]

            while len(binary)<length:
                binary="0" + binary

            counts=[]
            for j in range(numb_countone):
                counts.append(0)
                for k in range(numb_parity):
                    if binary[j*numb_parity + k] == '1':
                        counts[j]+=1
                if counts[j]%2 == 0:
                    counts[j]=0
                else:
                    counts[j]=1

            if sum(counts) > numb_countone/2:
                output='1'
            else:
                output='0'

            #fp.write(str(i)+"\t")
            for j in binary:
                fp.write(j+ "\t")
            fp.write(output+ "\n")

    except:
        print("Data set generation: ERROR - Cannot generate all data instances due to computational limitations")


generate_complete_parity_countone('Demo_Datasets/11Majority_Data_Complete.txt',1,11)
#generate_parity_countone_data('Demo_Datasets/3Parity_5CountOne_Data_20000.txt',3,5,20000)