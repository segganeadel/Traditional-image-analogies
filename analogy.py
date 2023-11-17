
import numpy as np
import cv2
from features import get_features
from parameters import search_method
from search import nearest_neighbor

"""
To get pyflann to work I had to copy the x64 directory from Lib>site-packages>pyflann>lib>win32 
into envs>python>Lib>site-packages>pyflann>win32. If you have difficulties just use sklearn, and remove references
to pyflann.  
"""

def remap_y(imgA,imgB):
    meanA = np.mean(imgA)
    sdA = np.std(imgA)
    meanB = np.mean(imgB)
    sdB = np.std(imgB)

    imgA_remapped = sdB/sdA*(imgA-meanA)+meanB

    return imgA_remapped


def rgb2yiq(image, remap=False, remap_target=None, feature='y'):
    yiq_xform = np.array([[0.299, 0.587, 0.114],
                          [0.596, -0.275, -0.321],
                         [0.212, -0.523, 0.311]])
    yiq = np.dot(image, yiq_xform.T.copy())

    if remap:
        remap_y(image, remap_target)
    if feature == 'y':
        return yiq[:,:,0]
    elif feature == 'yiq':
        return yiq


def yiq2rgb(image):
    rgb_xform = np.array([[1., 0.956, 0.619],
                          [1., -0.272, -0.647],
                          [1., -1.106, 1.703]])
    rgb = np.dot(image, rgb_xform.T.copy())

    return rgb


def get_pyramid(image, levels):
    img = image.copy()
    pyr = [img]
    for i in range(levels):
        img = cv2.pyrDown(img)
        pyr.append(img)

    return pyr


def make_analogy_color(lvl, Nlvl, A_L, Ap_L, B_L, Bp_L, s_L, kappa=0, method='pyflann'):

    A_f = get_features(rgb2yiq(A_L[lvl], remap=True, remap_target=B_L[lvl]))
    Ap_f = get_features(rgb2yiq(Ap_L[lvl], remap=True, remap_target=B_L[lvl]), causal=True)

    A_f = np.concatenate((A_f, Ap_f), 2)
    # initialize additional feature sets and B mats
    if lvl < Nlvl:
        Ad = cv2.resize(A_L[lvl+1], (A_L[lvl].shape[1],A_L[lvl].shape[0]), interpolation=cv2.INTER_CUBIC)
        Ad_f = get_features(rgb2yiq(Ad, remap=True, remap_target=B_L[lvl+1]))
        
        Apd = cv2.resize(Ap_L[lvl + 1], (Ap_L[lvl].shape[1], Ap_L[lvl].shape[0]), interpolation=cv2.INTER_CUBIC)
        Apd_f = get_features(rgb2yiq(Apd, remap=True, remap_target=B_L[lvl+1]))

        A_f = np.concatenate((A_f, Ad_f, Apd_f), 2)

    # put feature list into index format M*N,numFeatures (25+12)
    source_f_vect = A_f.reshape(-1, A_f.shape[-1])


    # initialize mat by taking previous pyramid level and resize it to the same shape as the current level
    # for lvl=Nlvl you can initialize it with current Ap or with some randomization function
    if lvl < Nlvl:
        Bp_L[lvl] = cv2.resize(Bp_L[lvl+1], dsize=(Bp_L[lvl].shape[1], Bp_L[lvl].shape[0]), interpolation=cv2.INTER_CUBIC)
    else:
        Bp_L[lvl] = cv2.resize(B_L[lvl], dsize=(Bp_L[lvl].shape[1], Bp_L[lvl].shape[0]), interpolation=cv2.INTER_CUBIC)


    B_patches = get_features(rgb2yiq(B_L[lvl]))
    Bp_patches = get_features(rgb2yiq(Bp_L[lvl]), causal=True)
    B_patches_f = np.concatenate((B_patches, Bp_patches), 2)
    if lvl < Nlvl:
        Bd = cv2.resize(B_L[lvl + 1], dsize=(B_L[lvl].shape[1], B_L[lvl].shape[0]), interpolation=cv2.INTER_CUBIC)
        Bpd = cv2.resize(Bp_L[lvl + 1], dsize=(Bp_L[lvl].shape[1], Bp_L[lvl].shape[0]), interpolation=cv2.INTER_CUBIC)
        Bd_patches = get_features(rgb2yiq(Bd))
        Bpd_patches = get_features(rgb2yiq(Bpd))

        B_patches_f = np.concatenate((B_patches_f, Bd_patches, Bpd_patches), 2)


    target_f_vect = B_patches_f.reshape(-1, B_patches_f.shape[-1])
    print("target_f_vect shape", target_f_vect.shape)

    """  Begin Neighbor Search Methods  """
    print(f"Building {search_method} index for size:", A_f.size, "for A size", Ap_L[lvl].size)
    neighbors , distances = nearest_neighbor(source_f_vect, target_f_vect)
    print(f"{search_method} index done...")
    """  End Neighbor Search Methods  """

    coh_chosen = 0
    # coh_fact is squared to get it closer to the performance as described in Hertzmann paper
    coh_fact = (1.0 + (2.0**(lvl - Nlvl)) * kappa)**2


    for x in range(Bp_L[lvl].shape[0]):
        if x%25 == 0:
            print("Rastering row", x, "of",Bp_L[lvl].shape[0])
        for y in range(Bp_L[lvl].shape[1]):
            
            position = np.ravel_multi_index((x,y), (Bp_L[lvl].shape[0], Bp_L[lvl].shape[1]))
            neighbor_app,distance = neighbors[position],distances[position]
            distance_app = distance**2

            m,n = np.unravel_index(neighbor_app, (A_f.shape[0], A_f.shape[1]))

            if kappa > 0:
                neighbor_coh, distance_coh = get_coherent(A_f, target_f_vect[position], x, y, s_L[lvl])
                got_coh = (neighbor_coh != [-1, -1])

                if got_coh and distance_coh <= distance_app * coh_fact:
                    m,n = neighbor_coh
                    coh_chosen += 1

            Bp_L[lvl][x,y,:] = Ap_L[lvl][m,n,:]  # move value into Bprime

            # save s
            s_L[lvl][x, y, 0] = m
            s_L[lvl][x, y, 1] = n

    print("Coherent pixel chosen", coh_chosen, "/", Bp_L[lvl].size, "times.")
    return Bp_L[lvl]


def get_coherent(A_f,B_f,q_x,q_y,s):  # tuned for 5x5 patches only
    min_distance = np.inf
    cohxy = [-1, -1]
    for i in range(-2, 3, 1):
        for j in range(-2, 3, 1):

            r_i,r_j = q_x+i,q_y+j
            if i == 0 and j == 0:  # only do causal portion
                break
            if r_i >= s.shape[0] or r_j >= s.shape[1]:
                continue

            sx,sy = s[r_i,r_j]
            if sx == -1 or sy == -1:
                continue
  
            #q_x - r_i = -i
            rx, ry = sx-i, sy-j

            if rx < 0 or rx >= A_f.shape[0] or ry < 0 or ry >= A_f.shape[1]:
                continue

            rstar = np.sum((A_f[rx,ry,:]-B_f)**2)

            if rstar < min_distance:
                min_distance = rstar
                cohxy = rx, ry

    return cohxy, min_distance

