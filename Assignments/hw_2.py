# Author: Matt Williams
# Version: 10/5/2022

import math
import os 
import cv2
import pandas as pd
import matplotlib.pyplot as plt
from utils import *
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.utils import shuffle
from sklearn.model_selection import cross_val_score
from sklearn.metrics import classification_report, ConfusionMatrixDisplay

K_FOLD = 5

def get_img_gray_mat(filename):
    src = os.path.join(DATASET_DIR, filename)
    bgr_mat = cv2.imread(src)
    gray_mat = cv2.cvtColor(bgr_mat, cv2.COLOR_BGR2GRAY)
    return gray_mat 

        


def run_knn(train_data, train_labels, test_data, test_labels): 
    knn_k_list = [1,3,5,7]

    accuracy_scores = []


    for k in knn_k_list: 
        knn = KNeighborsClassifier(n_neighbors=k)
        scores = cross_val_score(knn, train_data, train_labels, cv = K_FOLD, scoring="accuracy")
        accuracy_scores.append(scores.mean())

    plt.bar(knn_k_list, accuracy_scores)
    plt.show()

    highest_score = 0
    k_with_high_score = 0

    for k, score in zip(knn_k_list, accuracy_scores): 
        if score > highest_score: 
            highest_score = score
            k_with_high_score = k
    


    knn = KNeighborsClassifier(n_neighbors=k_with_high_score)
    knn.fit(train_data, train_labels)

    predicted_labels = knn.predict(test_data)

    class_report = classification_report(test_labels, predicted_labels, output_dict=True)
    print(f"Test Accuracy: {class_report['accuracy']}")



def run_k_fold_with_test(model, model_name, train_data, train_labels, test_data, test_labels):

    scores = cross_val_score(model, train_data, train_labels, cv=K_FOLD, scoring="accuracy", verbose=4, n_jobs=3)
    print(f"{model_name} 5-fold average accuracy: {scores.mean()}")

    model.fit(train_data, train_labels)

    predicted_labels = model.predict(test_data)
    class_report = classification_report(test_labels, predicted_labels, output_dict=True)
    print(f"{model_name} Test accuracy: {class_report['accuracy']}")

    conf_matrix_display = ConfusionMatrixDisplay.from_predictions(test_labels, predicted_labels)
    conf_matrix_display.plot()



if __name__ == "__main__": 
 
    #train_data, train_labels, test_data, test_labels = get_dataset()
    #run_knn(train_data, train_labels, test_data, test_labels)


    train_data, train_labels, test_data, test_labels = get_dataset(remove_negatives=True, to_flatten=True, return_hist=True)
    print(train_data.shape)
    print(type(train_data))

    #svc = SVC(kernel="rbf", C=10)
    #run_k_fold_with_test(svc, "Support Vector Machine", train_data, \
    #                        train_labels, test_data, test_labels)

   
    #mlp = MLPClassifier(learning_rate="adaptive")
    #run_k_fold_with_test(mlp, "Multi-Layer Perceptron", train_data, \
    #                        train_labels, test_data, test_labels)

    nb = GaussianNB()
    run_k_fold_with_test(nb, "Gaussian Naive Bayes", train_data, \
                            train_labels, test_data, test_labels)

    #dt = DecisionTreeClassifier()
    #run_k_fold_with_test(dt, "Decision Tree", train_data, \
    #                        train_labels, test_data, test_labels)

    #rf = RandomForestClassifier()
    #run_k_fold_with_test(rf, "Random Forest", train_data, \
    #                        train_labels, test_data, test_labels)

    #ada = AdaBoostClassifier()
    #run_k_fold_with_test(ada, "Ada boost", train_data, \
    #                        train_labels, test_data, test_labels)
    