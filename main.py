import numpy as np
import cv2
import analogy
from parameters import *
from features import get_features

"""
Frame work for looping through make_analogy
Parameters:
A.jpg, Ap.jpg, B.jpg : Inputs into the framework. Will attempt to create novel image Bp.
remap_A : specifies whether an attempt should be made to normalize the luminance of A and Ap to B. Can help with
            artifacting with images of noticeably different luminance averages.
pyr_levels : number of Gaussian pyramids to iterate along. Increases expression of low frequency features.
kappa : coherence weight, refer to Hertzmann paper for some suggested values
search_method : each works well. Three are implemented: pyflann_kmeans, pyflann_kdtree, and sk_nn. kmeans iterations 
                is set to reach convergence. If you want to speed up kmeans you may want to change it to 7-10. kdtree
                is probably the fastest, sk_nn is the slowest.
type : two types are implemented. The default is luminance since the paper predominantly did its work only in the
        luminance domain. The color setting lets you apply the color from textures and colorize grayscale features.
                 
"""


A_L = analogy.get_pyramid(imgA, pyr_levels)
B_L = analogy.get_pyramid(imgB, pyr_levels)
Ap_L = analogy.get_pyramid(imgAp, pyr_levels)

Bp_L = []
s = []
for i in range(len(B_L)):
    Bp_L.append(np.zeros(B_L[i].shape))
    s.append(np.full((B_L[i].shape[0],B_L[i].shape[1],2),-1))

# process pyramid from coursest to finest
for lvl in range(pyr_levels, -1, -1):
    print("Starting Level: ", lvl, "of ", pyr_levels)


    Bp_L[lvl] = analogy.make_analogy_color(lvl, pyr_levels, A_L, Ap_L, B_L, Bp_L, s, kappa, search_method)
    Bp_int = np.uint8(Bp_L[lvl].copy() * 255)
    cv2.imshow("bp", Bp_int)
    cv2.waitKey(1)
    imgBp = Bp_L[lvl]

    imgBp = imgBp * 255.0
    imgBp[imgBp > 255] = 255
    imgBp[imgBp < 0] = 0

    write_name = 'out/Bp_PyrLvl-'+str(lvl)+'.jpg'
    cv2.imwrite(write_name, imgBp)
