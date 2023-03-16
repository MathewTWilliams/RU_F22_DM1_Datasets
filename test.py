import cv2 as cv 
import os 
import pandas as pd 
 
 
def pixel_intensity_hist(): 
     
    images = [] 
    gray_scale_img = [] 
    hist_img = [] 
    col = [] 
     
    for i in os.listdir("Weed-4class-54"): 
        if i.endswith(".jpg"): 
            images.append(os.path.join("Weed-4class-54", i)) 
             
     
    for i in images: 
         
        imgs = cv.imread(i) 
         
        gray_img = cv.cvtColor(imgs, cv.COLOR_RGB2GRAY) 
        gray_scale = cv.cvtColor(gray_img, cv.COLOR_GRAY2BGR) 
        hist = cv.calcHist([gray_scale], [0], None, [256], [0, 256]) 
        hist_img.append(hist.flatten().tolist()) 
         
    csv = pd.read_csv("Weed-4class-54/Weed-4class-54-labels.csv") 
     
    for i in range(1, 257): 
        pix = "pixel"+str(i) 
        col.append(pix) 
         
    df = pd.DataFrame(columns = col) 
     
    for i in range(len(hist_img)): 
        df.loc[i] = hist_img[i] 
     
    file = pd.concat([csv, df], axis = 1) 
    file.to_csv("classification.csv") 
     
     
pixel_intensity_hist()  
import pandas as pd 
from sklearn.model_selection import train_test_split 
from sklearn.neighbors import KNeighborsClassifier 
from sklearn.model_selection import cross_val_score 
import numpy as np 
from sklearn import metrics 
import matplotlib.pyplot as plt 
 
 
dataset = pd.read_csv("classification.csv") 
dataset.drop('Filename', inplace=True, axis=1) 
dataset.drop('Unnamed: 0', inplace=True, axis=1) 
X=dataset.iloc[:, 2:-1] 
Y=dataset.iloc[:,0:1] 
X_train_c, X_test_c, y_train_c, y_test_c = train_test_split(X[0:1124], Y[0:1124
11/4/22, 11:43 PM Programming_Assignment_2_Saadhana_Reddy_K
localhost:8889/lab/tree/Programming_Assignment_2_Saadhana_Reddy_K.ipynb 2/51
X_train_l, X_test_l, y_train_l, y_test_l = train_test_split(X[1125:2188], Y[112
X_train_s, X_test_s, y_train_s, y_test_s = train_test_split(X[2189:3262], Y[218
X_train_p, X_test_p, y_train_p, y_test_p = train_test_split(X[3263:4324], Y[326
 
xtrain=[X_train_l,X_train_s,X_train_p] 
xtest=[X_test_l,X_test_s,X_test_p] 
ytrain=[y_train_l,y_train_s,y_train_p] 
ytest=[y_test_l,y_test_s,y_test_p] 
 
X_train=pd.DataFrame() 
X_test=pd.DataFrame() 
y_train=pd.DataFrame() 
y_test=pd.DataFrame() 
 
 
for i in xtrain: 
    X_train_c=X_train_c.append(i, ignore_index=True) 
for i in xtest: 
    X_test_c=X_test_c.append(i, ignore_index=True) 
for i in ytrain: 
    y_train_c=y_train_c.append(i, ignore_index=True) 
for i in ytest: 
    y_test_c=y_test_c.append(i, ignore_index=True) 
     
     
def model_selection(): 
     
    scores=[] 
    neighbours =[1,3,5,7] 
     
    #performing 5 cross fold validation 
    for i in neighbours: 
        knn_cv = KNeighborsClassifier(n_neighbors=i) 
        cv_scores = cross_val_score(knn_cv, X_train_c,y_train_c.values.ravel(),
        scores.append(round(np.mean(cv_scores,keepdims=False)*100,2)) 
        print(scores) 
         
#     plot graph
    xpoints = np.array(neighbours) 
    ypoints = np.array(scores) 
 
    plt.plot(xpoints, ypoints,"*") 
    plt.show() 
     
    #test_accuracy 
    knn_cv = KNeighborsClassifier(n_neighbors=7) 
    knn_cv.fit(X_train_c,y_train_c.values.ravel()) 
    pred=knn_cv.predict(X_test_c) 
    print("K with Highest Accuracy is ",7) 
    print("Accuracy with the test data ",round(metrics.accuracy_score(y_test_c.
 
model_selection() 