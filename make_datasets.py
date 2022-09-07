# Author: Matt Williams
# Version: 9/07/2022

import pandas as pd
from shutil import copyfile, rmtree
from itertools import combinations
import os 
from zipfile import ZipFile

CWD = os.path.abspath(os.getcwd())
LABELS_FILE = os.path.join(CWD, "labels.csv")
IMAGES_DIR = os.path.join(CWD, "Images")
NEGATIVES_DIR = os.path.join(CWD, "Negatives")
NEGATIVES_FILE = os.path.join(NEGATIVES_DIR, "Negatives.csv")
COMBO_FILE_NAME = os.path.join(CWD, "combinations.csv")
DATASETS_DIR = os.path.join(CWD, "Datasets")

NEGATIVES_ZIP_FILE = "Negatives.zip"
DATASET_DIRS_BASE_NAME = "Weed-4class-"
SPECIES_COL = "Species"
FILENAME_COL = "Filename"
LABEL_COL = "Label"
DIR_COL = "Directory Name"
COMBO_COL = "Class Combination"

NEGATIVE_CLASS = "Negative"
NUM_CLASSES = 4


def sort_dataset():
    labels_df = pd.read_csv(LABELS_FILE)

    species_dict = {}

    for i in range(len(labels_df.index)): 
        file_df = labels_df.iloc[[i]]
        species = labels_df[SPECIES_COL].values[i]
        if species not in species_dict.keys(): 
            species_dict[species] = pd.DataFrame(columns=[FILENAME_COL, LABEL_COL, SPECIES_COL])

        species_dict[species] = pd.concat([species_dict[species], file_df])

    return species_dict


def make_negatives_zip(negative_df):
    os.mkdir(NEGATIVES_DIR)
    os.mkdir(DATASETS_DIR)

    with ZipFile(file = os.path.join(DATASETS_DIR, NEGATIVES_ZIP_FILE), mode = "w") as zip: 
        for i in range(len(negative_df.index)): 
            file = negative_df[FILENAME_COL].values[i]

            src = os.path.join(IMAGES_DIR, file)
            dst = os.path.join(NEGATIVES_DIR, file)
            copyfile(src, dst)
            zip.write(os.path.join(os.path.basename(NEGATIVES_DIR), file))

        negative_df.to_csv(NEGATIVES_FILE, index = False)
        zip.write(os.path.join(os.path.basename(NEGATIVES_DIR), os.path.basename(NEGATIVES_FILE)))

    rmtree(NEGATIVES_DIR, ignore_errors=True)
    

def make_dataset_zip_files(species_dict): 
    species_list = [species for species in species_dict.keys() if species != NEGATIVE_CLASS]
    
    combos = combinations(species_list, NUM_CLASSES)    
    combo_df = pd.DataFrame(columns=[DIR_COL, COMBO_COL])

    for i, combo in enumerate(combos):        
        directory_name = DATASET_DIRS_BASE_NAME + str(i)
        new_directory = os.path.join(CWD, directory_name)
        os.mkdir(new_directory)

        single_combo_df = pd.DataFrame(data = [[directory_name, combo]], columns=[DIR_COL, COMBO_COL])
        combo_df = pd.concat([combo_df, single_combo_df])

        new_labels_df = pd.DataFrame(columns = [FILENAME_COL, LABEL_COL, SPECIES_COL])
        with ZipFile(file = os.path.join(DATASETS_DIR, directory_name+".zip"), mode = "w") as zip: 
            for species in combo:
                species_df = species_dict[species]
                
                for j in range(len(species_df.index)): 
                    file_df = species_df.iloc[[j]]
                    file = species_df[FILENAME_COL].values[j]

                    src = os.path.join(IMAGES_DIR, file)
                    dst = os.path.join(CWD, directory_name, file)
                    copyfile(src, dst)
                    zip.write(os.path.join(directory_name, file))
                    new_labels_df = pd.concat([new_labels_df, file_df])
            
            new_labels_filename = DATASET_DIRS_BASE_NAME + "{}-labels.csv".format(i)
            new_labels_filepath = os.path.join(new_directory, new_labels_filename)
            new_labels_df.to_csv(new_labels_filepath, index = False)
            zip.write(os.path.join(directory_name, new_labels_filename))
        
        rmtree(path = new_directory, ignore_errors=True)

    combo_df.to_csv(COMBO_FILE_NAME, index = False)


if __name__ == "__main__": 
    
    species_dict = sort_dataset()
    make_negatives_zip(negative_df = species_dict[NEGATIVE_CLASS])
    make_dataset_zip_files(species_dict = species_dict)

