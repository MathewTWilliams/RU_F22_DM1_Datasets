# Author: Matt Williams
# Version: 9/2/2022


from calendar import c
import pandas as pd
from shutil import copyfile
from itertools import combinations
import os 

CWD = os.path.abspath(os.getcwd())
LABELS_FILE = os.path.join(CWD, "labels.csv")
NEGATIVES_FILE = os.path.join(CWD, "Negatives.csv")
IMAGES_DIR = os.path.join(CWD, "Images")

SPECIES_COL = "Species"
FILENAME_COL = "Filename"
LABEL_COL = "Label"
DIR_COL = "Directory Name"
COMBO_COL = "Class Combination"

NEGATIVE_CLASS = "Negative"
NUM_CLASSES = 4

if __name__ == "__main__": 
    labels_df = pd.read_csv(LABELS_FILE)

    species_dict = {}

    for i in range(len(labels_df.index)): 
        file_df = labels_df.iloc[[i]]
        species = file_df[SPECIES_COL].iloc[0]
        if species not in species_dict.keys(): 
            species_dict[species] = pd.DataFrame(columns=[FILENAME_COL, LABEL_COL, SPECIES_COL])

        species_dict[species] = pd.concat([species_dict[species], file_df])

    species_dict[NEGATIVE_CLASS].to_csv(NEGATIVES_FILE, index = False)
    species_list = [species for species in species_dict.keys() if species != NEGATIVE_CLASS]
    
    combos = combinations(species_list, NUM_CLASSES)    
    combo_df = pd.DataFrame(columns=[DIR_COL, COMBO_COL])

    for i, combo in enumerate(combos):        
        directory_name = "Weed-4class-{}".format(i)
        os.mkdir(os.path.join(CWD, directory_name))

        single_combo_df = pd.DataFrame(data = [[directory_name, combo]], columns=[DIR_COL, COMBO_COL])
        combo_df = pd.concat([combo_df, single_combo_df])

        new_labels_df = pd.DataFrame(columns = [FILENAME_COL, LABEL_COL, SPECIES_COL])

        for species in combo:
            species_df = species_dict[species]
            

            for j in range(len(species_df.index)): 
                file_df = species_df.iloc[[j]]
                file = file_df[FILENAME_COL].iloc[0]

                src = os.path.join(IMAGES_DIR, file)
                dst = os.path.join(CWD, directory_name, file)
                copyfile(src, dst)
                
                new_labels_df = pd.concat([new_labels_df, file_df])
        
        new_labels_filename = "Weed-4class-{}-labels.csv".format(i)
        new_labels_df.to_csv(os.path.join(CWD, directory_name, new_labels_filename), index = False)

    combo_filename = "combinations.csv"
    combo_df.to_csv(os.path.join(CWD, combo_filename), index = False)
