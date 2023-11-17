from parameters import search_method as method
from sklearn.neighbors import NearestNeighbors
import pyflann as pyflann

def nearest_neighbor(source_f_vect,target_f_vect):
    if method == 'pyflann_kmeans':
        flann = pyflann.FLANN()
        index = flann.build_index(source_f_vect, algorithm="kmeans", branching=32, iterations=-1, checks=16)
        neighbors, distances = flann.nn_index(target_f_vect, 1, checks=index['checks'])
        return neighbors, distances
    elif method == 'pyflann_kdtree':
        flann = pyflann.FLANN()
        index = flann.build_index(source_f_vect, algorithm="kdtree")
        neighbors, distances = flann.nn_index(target_f_vect, 1, checks=index['checks'])
        return neighbors, distances
    elif method == 'sk_nn':
        index = NearestNeighbors(n_neighbors=1, algorithm='auto').fit(source_f_vect)
        distances, neighbors = index.kneighbors(target_f_vect)
        return neighbors, distances
    else:
        raise ValueError('method not recognized')


