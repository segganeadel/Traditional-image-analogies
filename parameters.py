import cv2
import numpy as np

def read_images(apath, appath, bpath):
    imgA = cv2.imread(apath, cv2.IMREAD_UNCHANGED)
    imgA = np.array(imgA)/255.0

    imgAp = cv2.imread(appath, cv2.IMREAD_UNCHANGED)
    imgAp = np.array(imgAp)/255.0

    imgB = cv2.imread(bpath, cv2.IMREAD_UNCHANGED)
    imgB = np.array(imgB)/255.0

    return imgA, imgAp, imgB

"""  Start User Defined Inputs  """
path = "src/"
remap_A = 1
pyr_levels = 5
kappa = 70

#if sensitive to color
color = False

methods = ['pyflann_kmeans', 'pyflann_kdtree', 'sk_nn']
# method can be pyflann_kmeans, pyflann_kdtree or sk_nn, default:pyflann_kmeans
search_method = methods[1]



# type the path of your images here. You can change the output path at the bottom
imgA, imgAp, imgB = read_images(path+"A.jpg", path+"Ap.jpg", path+"B.jpg")
