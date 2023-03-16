#Author: Matt Williams
#Version: 10/5/2022

import os
import pandas as pd
import random
import cv2
import math
from sklearn.utils import shuffle
import numpy as np

CWD = os.path.abspath(os.getcwd())
DATASET_DIR = os.path.join(CWD, "Weed-4class-0")
DATASET_CSV = os.path.join(DATASET_DIR, "Weed-4class-0-labels.csv")
SPECIES_COL = "Species"
FILENAME_COL = "Filename"
LABEL_COL = "Label"


def get_data(): 
    dataset_df = pd.read_csv(DATASET_CSV)
    species_list = list(set(dataset_df[SPECIES_COL]))
    return dataset_df, species_list


def get_img_gray_mat(filename, to_flatten = False, return_hist = False, normalize = False, color = False):
    src = os.path.join(DATASET_DIR, filename)
    bgr_mat = cv2.imread(src)
    current_mat = cv2.cvtColor(bgr_mat, cv2.COLOR_BGR2RGB) \
                if color else \
                cv2.cvtColor(bgr_mat, cv2.COLOR_BGR2GRAY) 

    if return_hist: 
        current_mat = cv2.calcHist([current_mat], [0], None, [256], [0, 256])
    if return_hist or to_flatten: # must flatten if returning a hist
        current_mat = current_mat.flatten()
    if normalize:
        cv2.normalize(current_mat, current_mat, alpha = 0, beta = 1, norm_type=cv2.NORM_MINMAX)

    return current_mat 


def get_dataset(remove_negatives = False, to_flatten = True, labels_to_int = False,
                 return_hist = True, normalize = False, color = False, perc_test = 0.2): 
    train_data = []
    train_labels = []
    test_data = []
    test_labels = []

    dataset_df, species_list = get_data()

    cur_spec_int_label = 0

    for species in species_list:

        if remove_negatives and species == "Negative": 
            continue

        cur_species_df = dataset_df.loc[dataset_df[SPECIES_COL] == species]
        num_test_set = math.floor(len(cur_species_df.index) * perc_test)

        cur_species_test_df = cur_species_df.sample(n = num_test_set, replace = False)
        cur_species_train_df = cur_species_df.drop(index = cur_species_test_df.index, inplace=False)
        
        for _, row_ds in cur_species_train_df.iterrows(): 
            filename = row_ds[FILENAME_COL]
            gray_result = get_img_gray_mat(filename, to_flatten, return_hist, normalize, color)
            train_data.append(gray_result)
            train_labels.append(cur_spec_int_label if labels_to_int else species)

        for _, row_ds in cur_species_test_df.iterrows(): 
            filename = row_ds[FILENAME_COL]
            gray_result = get_img_gray_mat(filename, to_flatten, return_hist, normalize, color)
            test_data.append(gray_result)
            test_labels.append(cur_spec_int_label if labels_to_int else species)

        cur_spec_int_label += 1
    
    train_data, train_labels = shuffle(train_data, train_labels)
    test_data, test_labels = shuffle(test_data, test_labels)

    train_data = np.array(train_data)
    train_labels = np.array(train_labels)
    test_data = np.array(test_data)
    test_labels = np.array(test_labels)


    return train_data, train_labels, test_data, test_labels
        
