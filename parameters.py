import cv2

def read_images(apath, appath, bpath):
    imgA = cv2.imread(apath, cv2.IMREAD_UNCHANGED)/255.0
    imgAp = cv2.imread(appath, cv2.IMREAD_UNCHANGED)/255.0
    imgB = cv2.imread(bpath, cv2.IMREAD_UNCHANGED)/255.0

    return imgA, imgAp, imgB


def add_pairs(imgA, imgAp, path1, path2):
    imgA2 = cv2.imread(path1, cv2.IMREAD_UNCHANGED) / 255.0
    imgAp2 = cv2.imread(path2, cv2.IMREAD_UNCHANGED) / 255.0
    imgA2 = cv2.resize(imgA2,(imgA.shape[1],imgAp.shape[0]))
    imgAp2 = cv2.resize(imgAp2, (imgAp.shape[1],imgAp.shape[0]))
    matA = cv2.hconcat([imgA,imgA2])
    matAp = cv2.hconcat([imgAp,imgAp2])

    return matA, matAp

"""  Start User Defined Inputs  """

remap_A = 1
pyr_levels = 5
kappa = 1

methods = ['pyflann_kmeans', 'pyflann_kdtree', 'sk_nn']
# method can be pyflann_kmeans, pyflann_kdtree or sk_nn, default:pyflann_kmeans
search_method = methods[1]
# type can be luminance or color, most effects work with luminance. Texture synthesis works well with color
# default:luminance
types = ['luminance', 'color']
type = types[1]


# type the path of your images here. You can change the output path at the bottom
imgA, imgAp, imgB = read_images("src/A.jpg", "src/Ap.jpg", "src/B.jpg")
