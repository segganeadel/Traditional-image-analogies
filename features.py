from sklearn.feature_extraction.image import extract_patches_2d as extract
import cv2
import numpy as np
from math import floor
from parameters import color


def get_features(img, causal=False, coarse=False):


    imgG = rgb2yiq(img)
    print(imgG.shape)
    print(img.shape)
    height, width = imgG.shape  # dimensions of the current level of the gaussian pyramid
    
    padding = 1 if coarse else 2
    window = (3, 3) if coarse else (5, 5)

    dimentions_size = (window[0]*window[1]) if not causal else floor((window[0]*window[1])/2)
    features = np.zeros((height,width,dimentions_size))

    imgG_padded = cv2.copyMakeBorder(imgG,padding,padding,padding,padding,cv2.BORDER_DEFAULT)
    patchesG = extract(imgG_padded, window)[...,np.newaxis]
    print("patchesG shape", patchesG.shape)
    
    if color:
        imgC_padded = cv2.copyMakeBorder(img,padding,padding,padding,padding,cv2.BORDER_DEFAULT)
        patchesC = extract(imgC_padded, window)
        print("patchesC shape", patchesC.shape)
        patches = np.concatenate((patchesG, patchesC), axis=3)
    else:
        patches = patchesG

 # dimensions of the current level of the gaussian pyramid
    for i in range(height):
        for j in range(width):
                features[i, j, :] = patches[i * width + j].flatten()[0:features.shape[-1]]

    return features


def rgb2yiq(image):
    yiq_xform = np.array([[0.299, 0.587, 0.114],
                        [0.596, -0.275, -0.321],
                        [0.212, -0.523, 0.311]]).T

    y = np.dot(image, yiq_xform)[:,:,0]
    return y