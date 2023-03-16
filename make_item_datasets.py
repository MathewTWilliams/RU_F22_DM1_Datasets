# Author: Matt Williams
# Version: 10/11/2022


import pandas as pd
import os
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, association_rules


CWD = os.path.abspath(os.getcwd())
ITEM_DATASETS_DIR = os.path.join(CWD, "Item Datasets")
ITEMS_CSV = os.path.join(CWD, "basket.csv")
ITEMS_DATASET_BASENAME = "Grocery_Items_"
MIN_SUPPORT = 0.0075
MIN_THRESHOLD = 0.01

def make_datasets():
    n_samples = 8000
    n_datasets = 70
    item_sets_df = pd.read_csv(ITEMS_CSV)

    os.mkdir(ITEM_DATASETS_DIR)
    for i in range(n_datasets):
        sample_df = item_sets_df.sample(n = 8000, replace = False)
        sample_df.to_csv(os.path.join(ITEM_DATASETS_DIR, ITEMS_DATASET_BASENAME + f"{i}.csv" ), index=False)


def test_dataset():
    my_dataset = os.path.join(ITEM_DATASETS_DIR,ITEMS_DATASET_BASENAME + "0.csv")
    dataset_df = pd.read_csv(my_dataset)

    dataset_nd_ary = dataset_df.values

    final_list = []
    for transaction in dataset_nd_ary:
        final_list.append(transaction[~pd.isnull(transaction)].tolist())
    
    encoder = TransactionEncoder()
    encode_ary = encoder.fit(final_list).transform(final_list)
    dataset_one_hot_df = pd.DataFrame(encode_ary, columns = encoder.columns_)
    frequent_itemsets = apriori(dataset_one_hot_df, min_support = MIN_SUPPORT, use_colnames=True)
    print(frequent_itemsets)

    assoc_rules = association_rules(frequent_itemsets, min_threshold=MIN_THRESHOLD)
    print(assoc_rules)

if __name__ == "__main__":
    #make_datasets()
    test_dataset()