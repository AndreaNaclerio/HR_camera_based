#########################################
#########################################
########## UBFC-RPPG Dataset ############
#########################################
#########################################

'''
DATASET DESCRIPTION

Each subject has a folder. The folder is composed by 2 files:
- .avi file : contain the acquired video of the patient 
- .txt file : it can be seen as a list of 4 elements:
    - 0 --> PPG signal values
    - 1 --> HR values (we have directly the ground truth already extracted from the PPG signal)
    - 2 --> time interval axis
    - 3 --> is empty so it has removed
The three signals are divided by a \n 
'''

def import_UBFC_RPPG_ground_truth(num_patient):
    if num_patient in (0,2,6,7,19,21,28,29) or num_patient >= 50:
        print('there is no patient associated to this number. Select a number between 1 ad 50 (no 2,6,7,19,21,28,29)')
        return

    signal_link = '/content/drive/MyDrive/DATASET_2/subject' + str(num_patient) + '/ground_truth.txt'

    # open and read the txt file
    file = open(signal_link,'r')
    data = file.read()
    file.close()

    # separate the signal
    signals = data.split('\n')
    string_single_signals = signals[0:3] #remove the last element of the element of the list since it was empty

    # the separated signals are written in a string format. Now we want to transform this in a float to work with them
    # string --> float
    float_single_signals = []*3
    for i in string_single_signals:
        temp = []
        for c in i.split(): # we are selecting each single number that is inserted in the string 
            temp.append(float(c))
        float_single_signals.append(temp)

    return float_single_signals


'''
input: PPG signal (already croppped) and temporal vector (same dimension)
output: vector of HR, average HR (simple mean of the vector values)
'''
import numpy as np
from scipy.signal import find_peaks
import heartpy as hp


def PPG_to_HR(signal, time_interval, type = 'Tool_box'):
    if type == 'Tool_box':
        working_data, measures = hp.process(signal, sample_rate=30)
        HR_toolbox = measures['bpm']
        return HR_toolbox
    
    elif type == 'Manual':
        peaks_idx = np.array(find_peaks(signal)[0])
        time_values = time_interval[peaks_idx]

        HR_vector = []
        for i in range(np.array(time_values).shape[0]-1):
            temp = time_values[i+1]-time_values[i]
            HR = 60/temp
            if HR > 40 and HR < 150:
                HR_vector.append(HR)
        return HR_vector
    
    else:
        print("Type can be 'Tool_box' or 'Manual'")








    
