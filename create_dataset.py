
import numpy as np
import pandas as pd
import os
import random
from shutil import copyfile
import pydicom as dicom
import cv2


# In[ ]:


# set parameters here
savepath = 'data'



# path to covid-19 dataset from https://github.com/ieee8023/covid-chestxray-dataset
imgpath = '../covid-chestxray-dataset/images'
csvpath = '../covid-chestxray-dataset/metadata.csv'


train = []
test = []
test_count = {'normal': 0, 'pneumonia': 0, 'COVID-19': 0}
train_count = {'normal': 0, 'pneumonia': 0, 'COVID-19': 0}

mapping = dict()
mapping['COVID-19'] = 'COVID-19'
mapping['SARS'] = 'pneumonia'
mapping['MERS'] = 'pneumonia'
mapping['Streptococcus'] = 'pneumonia'
mapping['Normal'] = 'normal'
mapping['Lung Opacity'] = 'pneumonia'
mapping['1'] = 'pneumonia'

# train/test split
split = 0.1


csv = pd.read_csv(csvpath, nrows=None)
idx_pa = csv["view"] == "PA"  # Keep only the PA view
csv = csv[idx_pa]

pneumonias = ["COVID-19", "SARS", "MERS", "ARDS", "Streptococcus"]
pathologies = ["Pneumonia","Viral Pneumonia", "Bacterial Pneumonia", "No Finding"] + pneumonias
pathologies = sorted(pathologies)



filename_label = {'normal': [], 'pneumonia': [], 'COVID-19': []}
count = {'normal': 0, 'pneumonia': 0, 'COVID-19': 0}
for index, row in csv.iterrows():
    f = row['finding']
    if f in mapping:
        count[mapping[f]] += 1
        entry = [int(row['patientid']), row['filename'], mapping[f]]
        filename_label[mapping[f]].append(entry)

print('Data distribution from covid-chestxray-dataset:')
print(count)




for key in filename_label.keys():
    arr = np.array(filename_label[key])
    if arr.size == 0:
        continue
    # split by patients
    # num_diff_patients = len(np.unique(arr[:,0]))
    # num_test = max(1, round(split*num_diff_patients))
    # select num_test number of random patients
    Cov = ['19', '20', '36', '42', '86']
    if key == 'pneumonia':
        test_patients = ['8', '31']
    elif key == 'COVID-19':
        test_patients = ['19', '20', '36', '42', '86'] # random.sample(list(arr[:,0]), num_test)
    else:
        test_patients = []
    print('Key: ', key)
    print('Test patients: ', test_patients)
    # go through all the patients
    for patient in arr:
        if patient[0] in test_patients:
            if patient[0] == '8' or patient[0] == '31':
                copyfile(os.path.join(imgpath, patient[1]), os.path.join(savepath, 'test/pneumonia', patient[1]))
                test.append(patient)
                test_count[patient[2]] += 1

            if patient[0] in Cov:
                copyfile(os.path.join(imgpath, patient[1]), os.path.join(savepath, 'test/COVID-19', patient[1]))
                test.append(patient)
                test_count[patient[2]] += 1

        else:

            if patient[2] =="COVID-19":
                copyfile(os.path.join(imgpath, patient[1]), os.path.join(savepath, 'train/COVID-19', patient[1]))
                train.append(patient)
                train_count[patient[2]] += 1

            if patient[2] =="pneumonia":
                copyfile(os.path.join(imgpath, patient[1]), os.path.join(savepath, 'train/pneumonia', patient[1]))
                train.append(patient)
                train_count[patient[2]] += 1

print('test count: ', test_count)
print('train count: ', train_count)




print('Final stats')
print('Train count: ', train_count)
print('Test count: ', test_count)
print('Total length of train: ', len(train))
print('Total length of test: ', len(test))
