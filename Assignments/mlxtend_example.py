#Author: Matt Williams
#Version: 10/16/2022

import pandas as pd
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, association_rules
import os

# CHANGE TO YOUR ASSIGNED NUMBER
DATASET_NUMBER = "0"

CWD = os.path.abspath(os.getcwd())
ITEMS_DATASET = os.path.join(CWD, f"Grocery_Items_{DATASET_NUMBER}.csv")

MIN_SUPPORT = 0.01
MIN_THRESHOLD = 0.01


def main():
    dataset_df = pd.read_csv(ITEMS_DATASET)

    dataset_nd_ary = dataset_df.values

    # removing NaN values from all transactions
    transactions_list = []
    for transaction in dataset_nd_ary:
        transactions_list.append(transaction[~pd.isnull(transaction)].tolist())


    encoder = TransactionEncoder()
    encode_ary = encoder.fit(transactions_list).transform(transactions_list)
    dataset_one_hot_df = pd.DataFrame(encode_ary, columns=encoder.columns_)
    print(dataset_one_hot_df)
    frequent_itemsets = apriori(dataset_one_hot_df, min_support=MIN_SUPPORT)
    print(frequent_itemsets)
    assoc_rules = association_rules(frequent_itemsets, min_threshold=MIN_THRESHOLD)
    print(assoc_rules)

if __name__ == "__main__":
    main()