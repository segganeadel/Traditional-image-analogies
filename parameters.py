"""  Start User Defined Inputs  """
folder = 'test3'
pyr_levels = 5
kappa = 0

#if sensitive to color
color = True

methods = ['pyflann_kmeans', 'pyflann_kdtree', 'sk_nn', 'faiss']
# method can be pyflann_kmeans, pyflann_kdtree or sk_nn, default:pyflann_kmeans
search_method = methods[1]