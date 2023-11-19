import cv2

def read_images(apath, appath, bpath):

    imgA = cv2.imread(apath)
    imgAp = cv2.imread(appath)
    imgB = cv2.imread(bpath)

    return imgA, imgAp, imgB

"""  Start User Defined Inputs  """
path = "src/test/"
pyr_levels = 5
kappa = 0

#if sensitive to color
color = True

methods = ['pyflann_kmeans', 'pyflann_kdtree', 'sk_nn', 'faiss']
# method can be pyflann_kmeans, pyflann_kdtree or sk_nn, default:pyflann_kmeans
search_method = methods[1]



# type the path of your images here. You can change the output path at the bottom
imgA, imgAp, imgB = read_images(path+"A.png", path+"Ap.png", path+"B3.png")
