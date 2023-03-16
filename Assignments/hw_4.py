# Author: Matt Williams
# Version: 11/14/2022


from utils import *
import pandas as pd
from sklearn.cluster import KMeans, BisectingKMeans, SpectralClustering, DBSCAN, AgglomerativeClustering
import numpy as np
from sklearn.metrics import fowlkes_mallows_score, silhouette_score
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

def run_clustering_trials(clustering, param_dict, train_x, train_y, show_graphs = False):
    fmi_scores = []
    silhouette_scores = []
    for _ in range(50):
        k_means = clustering(**param_dict)
        preds = k_means.fit_predict(train_x)
        fmi_scores.append(fowlkes_mallows_score(train_y, preds))
        silhouette_scores.append(silhouette_score(train_x, preds))
        
    
    if show_graphs:
        plt.title("Fowlkes-Mallows Index Histogram")
        plt.hist(fmi_scores, bins = 10 )
        plt.show()
        plt.clf()

        plt.title("Silhouette Score Histogram")
        plt.hist(silhouette_scores, bins = 10)
        plt.show()
        plt.clf()

    return fmi_scores, silhouette_scores

def part_two():

    train_x, train_y, _, _ = get_dataset(remove_negatives=True, \
                                                to_flatten=True, \
                                                return_hist=True, \
                                                normalize=True, \
                                                perc_test=0.0)
    cluster_scores_dict = {}

    fmi_scores, silhouette_scores = run_clustering_trials(KMeans, {"n_clusters" : 4, "init": "random"} ,train_x, train_y, show_graphs=True)
    cluster_scores_dict["K-Means(Random init)"] = (fmi_scores, silhouette_scores)

    fmi_scores, silhouette_scores = run_clustering_trials(KMeans, {"n_clusters" : 4, "init": "k-means++"}, train_x, train_y, show_graphs=True)
    cluster_scores_dict["K-Means(k-means++ init)"] = (fmi_scores, silhouette_scores)

    fmi_scores, silhouette_scores = run_clustering_trials(BisectingKMeans, {"n_clusters" : 4, "init": "random"}, train_x, train_y, show_graphs=True)
    cluster_scores_dict["Bisecting K-Means"] = (fmi_scores, silhouette_scores)

    fmi_scores, silhouette_scores = run_clustering_trials(SpectralClustering, {"n_clusters" : 4}, train_x, train_y, show_graphs=True)
    cluster_scores_dict["Spectral"] = (fmi_scores, silhouette_scores)

    for clustering, (fmi_scores, silhouette_scores) in cluster_scores_dict.items():
        print(f"{clustering} average FMI score: {np.average(fmi_scores)}")
        print(f"{clustering} average Silhouette score: {np.average(silhouette_scores)}")

def part_three(): 
    silhouette_dict = {}

    train_x, train_y, _, _ = get_dataset(remove_negatives=True, \
                                                to_flatten=True, \
                                                return_hist=True, \
                                                normalize=True, \
                                                perc_test=0.0)

    for k in range(2,9): 
        _, silhouette_scores = run_clustering_trials(KMeans, {"n_clusters" : k, "init": "random"}, train_x, train_y)
        silhouette_dict[str(k)] = np.average(silhouette_scores)

    plt.title("Average Silhoueete Scores with K-means")
    plt.xlabel("k")
    plt.ylabel("Average Silhouette Score")
    plt.plot(silhouette_dict.keys(),silhouette_dict.values())
    plt.show()
    plt.clf()    

def part_four():
    train_x, train_y, _, _ =  get_dataset(remove_negatives=True, \
                                            to_flatten=True, \
                                            return_hist=True, \
                                            normalize=True, \
                                            perc_test=0.0)

    pca = PCA(n_components=2)
    points = pca.fit_transform(train_x)
    db_scan = DBSCAN(eps = 0.25, min_samples=5)
    preds = db_scan.fit_predict(points)
    unique_labels = set(preds)
    n_clusters = len(unique_labels) - (1 if -1 in unique_labels else 0)

    print(f"Unique lables: {unique_labels}")
    print(f"Number of Clusters: {n_clusters}")
    print(f"DBSCAN Silhouette Score: {silhouette_score(points, preds)}")

    plt.title(f"Estimated number of clusters: {n_clusters}")
    core_samples_mask = np.zeros_like(preds, dtype=bool)
    core_samples_mask[db_scan.core_sample_indices_] = True
    colors = [plt.cm.Spectral(each) for each in np.linspace(0, 1, len(unique_labels))]

    for k, col in zip(unique_labels, colors):
        if k == -1: 
            col = [0,0,0,1]
        class_member_mask = preds == k
        xy = points[class_member_mask & core_samples_mask]
        plt.plot(
            xy[:,0],
            xy[:,1],
            "o",
            markerfacecolor = tuple(col),
            markeredgecolor = "k",
            markersize=14,
        )

        xy = points[class_member_mask & ~core_samples_mask]
        plt.plot(
            xy[:,0],
            xy[:,1],
            "o",
            markerfacecolor = tuple(col),
            markeredgecolor = "k",
            markersize=6,
        )
    plt.show()
    plt.clf()

def part_five(): 
    train_x, train_y, _, _ =  get_dataset(remove_negatives=True, \
                                            to_flatten=True, \
                                            return_hist=True, \
                                            normalize=True, \
                                            perc_test=0.0)
    pca = PCA(n_components=2)
    points = pca.fit_transform(train_x)

    strategies = ["ward", "complete", "average", "single"]
    color_dict = {0 : "blue", 
                  1 : "red", 
                  2 : "green", 
                  3 : "yellow"}

    for strat in strategies:
        agglo = AgglomerativeClustering(n_clusters=4, linkage=strat)
        preds = agglo.fit_predict(points)
        print(f"{strat} silhouette score: {silhouette_score(points, preds)}")
        plt.title(f"{strat} clustering plot")
        colors = [color_dict[pred] for pred in preds]
        plt.scatter(points[:,0], points[:,1], c = colors)
        plt.show()
        plt.clf()





def main():
    part_two()
    #part_three()
    #part_four()
    #part_five()
if __name__ == "__main__":
    main()