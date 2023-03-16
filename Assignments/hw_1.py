#Author: Matt Williams
#Version: 11/6/2022

import pandas as pd
import matplotlib.pyplot as plt
import cv2
import os
import random
from sklearn.decomposition import PCA


CWD = os.path.abspath(os.getcwd())
DATASET_DIR = os.path.join(CWD, "Weed-4class-0")
DATASET_CSV = os.path.join(DATASET_DIR, "Weed-4class-0-labels.csv")
SPECIES_COL = "Species"
FILENAME_COL = "Filename"
LABEL_COL = "Label"



def get_random_species_images(dataset_df, species_list, n_rand): 
    file_names = []
    for species in species_list: 
        cur_spec_df = dataset_df[dataset_df[SPECIES_COL] == species]
        img_numbers_chosen = random.sample(range(len(cur_spec_df)), k = n_rand)

        for num in img_numbers_chosen: 
            file_names.append(cur_spec_df[FILENAME_COL].values[num])

    return file_names

def get_image(filename, grayscale = False): 
    src = os.path.join(DATASET_DIR, filename)
    img = cv2.imread(src)
    if grayscale:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img


def part_a(dataset_df, species_list): 
    
    n_rows = 2
    n_cols = 10
    subplot_offset = 10

    image_filenames = get_random_species_images(dataset_df, species_list, n_rand=2)
    grayscale_imgs = []

    for i, filename in enumerate(image_filenames): 
        gray_img = get_image(filename, grayscale=True)
        grayscale_imgs.append(gray_img)

        #plot rgb
        plt.subplot(n_rows, n_cols, i+1)
        plt.imshow(gray_img, cmap='gray')
        plt.subplot(n_rows, n_cols, i + 1 + subplot_offset)
        plt.hist(gray_img.flatten(), 256, [0,256])
    
    plt.show()
    plt.clf()

    ## 
    equ_grayscale_imgs = []
    for i, gray_img in enumerate(grayscale_imgs):
        norm_gray_img = cv2.equalizeHist(gray_img)
        equ_grayscale_imgs.append(norm_gray_img)

        plt.subplot(n_rows, n_cols, i+1)
        plt.imshow(norm_gray_img, cmap="gray")
        plt.subplot(n_rows, n_cols, i + 1 + subplot_offset)
        plt.hist(norm_gray_img.flatten(),256, [0,256])

    plt.show()
    plt.clf()

    for i in range(len(equ_grayscale_imgs)): 
        plt.subplot(n_rows, n_cols, i + 1)
        plt.imshow(grayscale_imgs[i], cmap='gray')
        plt.subplot(n_rows, n_cols, i + 1 + subplot_offset)
        plt.imshow(equ_grayscale_imgs[i], cmap='gray')

    plt.show()
    plt.clf()

def part_b(dataset_df, species_list): 
    
    n_rows = 1
    n_cols = 2
    colors = ["Blue", "Green", "Red"]

    image_filenames = get_random_species_images(dataset_df, species_list, n_rand=1)

    for filename in image_filenames:
        img = get_image(filename, grayscale=False)

        plt.subplot(n_rows, n_cols, 1)
        plt.imshow(img)
        channels = cv2.split(img)

        plt.subplot(n_rows, n_cols, 2)
        for i, chan_values in enumerate(channels):
            plt.title(colors[i])
            hist = cv2.calcHist([chan_values], [0], None, [256], [0, 256])
            plt.plot(hist, color = colors[i])
        
        plt.show()
        plt.clf()

def show_hist_comparisons(img1_mat, img2_mat):

    hist_Sizes = [256]
    ranges = [0, 256]

    hist_1 = cv2.calcHist([img1_mat], [0], None, hist_Sizes, ranges)
    hist_2 = cv2.calcHist([img2_mat], [0], None, hist_Sizes, ranges)

    #Manhattan Distance
    manhat_distance = cv2.norm(hist_1, hist_2, normType=cv2.NORM_L1)
    print("Manhattan Distance: {}".format(manhat_distance))
 
    #Euclidean Distance
    euclid_distance = cv2.norm(hist_1, hist_2, normType=cv2.NORM_L2)
    print("Euclidean Distance: {}".format(euclid_distance))

    #Bhattacharyya Distance
    bhatt_distance = cv2.compareHist(hist_1, hist_2, method = cv2.HISTCMP_BHATTACHARYYA)
    print("Bhattacharyya Distance: {}".format(bhatt_distance))

    #Histogram Intersection
    hist_intersection = cv2.compareHist(hist_1, hist_2, method = cv2.HISTCMP_INTERSECT)
    print("Histogram Intersection: {}".format(hist_intersection))

    print("-------------------------------------------------------")


def part_c(dataset_df, species_list):
    chosen_species = random.sample(species_list, k=2)
    gray_img_mats = []
    mid_index = 1

    for i,species in enumerate(chosen_species):
        img_files = get_random_species_images(dataset_df, [species], n_rand = i + 1)
        for img_file in img_files: 
            gray_mat = get_image(img_file, grayscale=True)
            gray_img_mats.append(gray_mat)

    # gray_image_mats contain 1 image of 1 class (index 0)
    # and 2 images of another class (index 1 and 2)
    gray_img1_mat = gray_img_mats[mid_index]
    gray_img2_mat = gray_img_mats[mid_index + 1]

    # same species
    print("Same Class Comparisons:")
    show_hist_comparisons(gray_img1_mat, gray_img2_mat)

    gray_img2_mat = gray_img_mats[mid_index - 1]
    # different species
    print("Different Class Comparisons:")
    show_hist_comparisons(gray_img1_mat, gray_img2_mat)


def part_d(dataset_df): 
    dims = 128
    sift_edge_threshold = 0.73
    contrast_threshold = 0.09

    # since an image is choosen randomly here, can't guarantee that the # of key points can be between 40-50.
    # so just choose any image you want for this part
    index = random.randint(0, len(dataset_df.index) - 1)
    filename = dataset_df[FILENAME_COL].values[index]
    src = os.path.join(DATASET_DIR, filename)
    img_mat = cv2.imread(src)

    sift = cv2.SIFT_create(nfeatures=dims, edgeThreshold=sift_edge_threshold, contrastThreshold=contrast_threshold)
    key_points = sift.detect(img_mat, None)
    print(len(key_points))
    final_img = cv2.drawKeypoints(img_mat, key_points, None)
    #cv2.imshow("Output", final_img)
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()

    X = 34 // 2
    orb_edge_threshold = X
    orb = cv2.ORB_create(edgeThreshold=orb_edge_threshold, 
                        patchSize=X,
                        nlevels=8,
                        fastThreshold=20,
                        scaleFactor=1.2,
                        WTA_K=2,
                        scoreType=cv2.ORB_HARRIS_SCORE,
                        firstLevel=0,
                        nfeatures=30)

    key_points = orb.detect(img_mat, None)
    print(len(key_points))
    final_img = cv2.drawKeypoints(img_mat, key_points, None, color = (0,255,0))
    final_img2 = cv2.drawKeypoints(img_mat, key_points, None, color = (0, 255,0),
                                flags = cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    cv2.imshow("Output", final_img)
    cv2.waitKey(0)
    cv2.imshow("Output", final_img2)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    


def part_e(dataset_df, species_list): 
    color_tuple_dict = {
        0 : ("b", "o"),
        1 : ("r", "+")
    }

    hist_Sizes = [256]
    ranges = [0, 256]

    species_chosen = random.sample(species_list, k = 2)
    
    label_list = []
    hist_data_list = []
    
    for i, species in enumerate(species_chosen): 
        cur_spec_df = dataset_df[dataset_df[SPECIES_COL] == species]
        for j in range(len(cur_spec_df.index)):
            filename = cur_spec_df[FILENAME_COL].values[j] 
            gray_img_mat = get_image(filename, grayscale=True)
            gray_img_hist = cv2.calcHist([gray_img_mat], [0], None, hist_Sizes, ranges)
            hist_data_list.append(gray_img_hist.flatten())
            label_list.append(i)


    pca = PCA(2)
    points = pca.fit_transform(hist_data_list)
    print(points.shape)

    plt.figure(figsize=(20,20))
    plt.axis([-5000, 5000, -5000, 5000])

    for i, point in enumerate(points):
        label = label_list[i]
        color, marker = color_tuple_dict[label]

        plt.scatter(point[0], point[1], color = color, marker = marker, alpha=0.5)

    plt.show()


if __name__ == "__main__": 
    dataset_df = pd.read_csv(DATASET_CSV)
    species_list = list(set(dataset_df[SPECIES_COL]))
    #part_a(dataset_df, species_list)
    #part_b(dataset_df, species_list)
    part_c(dataset_df, species_list)
    #part_d(dataset_df)
    #part_e(dataset_df, species_list)
    
