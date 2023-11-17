from sklearn.feature_extraction.image import extract_patches_2d as extract
import cv2
import numpy as np



def get_features(img, causal=False):

    #create 5x5 neighborhood for L, pad so that feature list is correct dimensions
    patches = cv2.copyMakeBorder(img,2,2,2,2,cv2.BORDER_DEFAULT)


    patches = extract(patches, (5, 5))
    if causal:
        features = np.zeros((img.shape[0],img.shape[1],12))
    else:
        features = np.zeros((img.shape[0],img.shape[1],25))

    height, width = img.shape  # dimensions of the current level of the gaussian pyramid
    for i in range(height):
        for j in range(width):
                features[i, j, :] = patches[i * width + j].flatten()[0:features.shape[2]]


    return features

