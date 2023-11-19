import numpy as np
import cv2
from parameters import *
from analogy import make_analogy

def get_pyramid(image, levels):
    img = image.copy()
    pyr = [img]
    for i in range(levels):
        img = cv2.pyrDown(img)
        pyr.append(img)

    return pyr



A_L = get_pyramid(imgA, pyr_levels)
B_L = get_pyramid(imgB, pyr_levels)
Ap_L =get_pyramid(imgAp, pyr_levels)

Bp_L = []
s = []
for i in range(len(B_L)):
    Bp_L.append(np.zeros(B_L[i].shape))
    s.append(np.full((B_L[i].shape[0],B_L[i].shape[1],2),-1))

# process pyramid from coursest to finest
for lvl in range(pyr_levels, -1, -1):
    print("Starting Level: ", lvl, "of ", pyr_levels)


    Bp_L[lvl] = make_analogy(lvl, pyr_levels, A_L, Ap_L, B_L, Bp_L, s, kappa)
    Bp_int = np.uint8(Bp_L[lvl].copy())
    #cv2.imshow("bp", Bp_int)
    #cv2.waitKey(1)
    imgBp = Bp_L[lvl]


    write_name = 'out/Bp_PyrLvl-'+str(lvl)+'.jpg'
    cv2.imwrite(write_name, imgBp)
