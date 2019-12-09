
#!/usr/bin/env python

"""
This script clusters roots features matrix and separately, sura features matrix

"""

__author__ = "Ali Khan"
__license__ = "GPL"
__version__ = "0.0.1"
__maintainer__ = "Ali Khan"
__email__ = "khan.aliasad@gmail.com"
__status__ = "dev"


import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt

from bidi import algorithm as bidialg
import arabic_reshaper

from qur_func import arabic_to_buc

import numpy as np
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs

#######################


if __name__ == '__main__':

	path = '/Users/alikhan/Downloads/qur/qcm-analysis/'

	import seaborn as sns 
	sns.set(style='white')
	rootorsura = 'sura'
	if rootorsura is 'root':
		data = pd.read_csv(path + 'data/' + 'root_subgraph_features.csv')
	else:
		data = pd.read_csv(path + 'data/' + 'sura_graph_features.csv')
		data = data.drop(['root_degree','root_in_degree','root_out_degree','root_freq','max_cooccurrence'],1)
	print(data.info())
	sns.pairplot(data)
	plt.show()
	X = data.drop(['Unnamed: 0','root','center'],1).fillna(0).as_matrix()

	from sklearn import metrics
	from scipy.spatial.distance import cdist

	# k means determine k
	distortions = []
	K = range(1,20)
	for k in K:
	    kmeanModel = KMeans(n_clusters=k).fit(X)
	    kmeanModel.fit(X)
	    distortions.append(sum(np.min(cdist(X, kmeanModel.cluster_centers_, 'euclidean'), axis=1)) / X.shape[0])

	# Plot the elbow
	plt.plot(K, distortions, 'bx-')
	plt.xlabel('k')
	plt.ylabel('Distortion')
	plt.title('The Elbow Method showing the optimal k')
	plt.show(); plt.close()

	y_pred = KMeans(n_clusters=8, random_state=170).fit_predict(X)
	plt.scatter(data['graph_order'], data['graph_size'], c=y_pred, alpha=0.5)
	plt.show();plt.close()
	import sys
	sys.exit()

	"""
	====================================
	Demonstration of k-means assumptions
	====================================

	This example is meant to illustrate situations where k-means will produce
	unintuitive and possibly unexpected clusters. In the first three plots, the
	input data does not conform to some implicit assumption that k-means makes and
	undesirable clusters are produced as a result. In the last plot, k-means
	returns intuitive clusters despite unevenly sized blobs.
	"""
	print(__doc__)

	# Author: Phil Roth <mr.phil.roth@gmail.com>
	# License: BSD 3 clause

	import numpy as np
	import matplotlib.pyplot as plt

	from sklearn.cluster import KMeans
	from sklearn.datasets import make_blobs

	plt.figure(figsize=(12, 12))

	n_samples = 1500
	random_state = 170
	X, y = make_blobs(n_samples=n_samples, random_state=random_state)

	# Incorrect number of clusters
	y_pred = KMeans(n_clusters=2, random_state=random_state).fit_predict(X)

	plt.subplot(221)
	plt.scatter(X[:, 0], X[:, 1], c=y_pred)
	plt.title("Incorrect Number of Blobs")

	# Anisotropicly distributed data
	transformation = [[0.60834549, -0.63667341], [-0.40887718, 0.85253229]]
	X_aniso = np.dot(X, transformation)
	y_pred = KMeans(n_clusters=3, random_state=random_state).fit_predict(X_aniso)

	plt.subplot(222)
	plt.scatter(X_aniso[:, 0], X_aniso[:, 1], c=y_pred)
	plt.title("Anisotropicly Distributed Blobs")

	# Different variance
	X_varied, y_varied = make_blobs(n_samples=n_samples,
	                                cluster_std=[1.0, 2.5, 0.5],
	                                random_state=random_state)
	y_pred = KMeans(n_clusters=3, random_state=random_state).fit_predict(X_varied)

	plt.subplot(223)
	plt.scatter(X_varied[:, 0], X_varied[:, 1], c=y_pred)
	plt.title("Unequal Variance")

	# Unevenly sized blobs
	X_filtered = np.vstack((X[y == 0][:500], X[y == 1][:100], X[y == 2][:10]))
	y_pred = KMeans(n_clusters=3,
	                random_state=random_state).fit_predict(X_filtered)

	plt.subplot(224)
	plt.scatter(X_filtered[:, 0], X_filtered[:, 1], c=y_pred)
	plt.title("Unevenly Sized Blobs")

	plt.show()
