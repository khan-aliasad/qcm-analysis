
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

from sklearn.cluster import KMeans
import matplotlib.cm as cm


#######################
def load_qtoc(path = '/Users/alikhan/Downloads/qur/'):
	qtoc = pd.read_csv(path + 'toc.csv')
	qtoc['Name Arabic'] = qtoc['Name Arabic'] \
							.apply(lambda x: bidialg.get_display(arabic_reshaper.reshape(x)))
	return qtoc


if __name__ == '__main__':

	# import seaborn as sns 
	# sns.set(style='whitegrid')

	path = '/Users/alikhan/Downloads/qur/qcm-analysis/'

	rootorsura = 'heatmap_features'#'root'#'sura'
	elbow = False 
	silhouette = False

	todrop = ['Unnamed: 0','root','center']
	col1 = 'graph_order'
	col2 = 'graph_size'
	if rootorsura is 'root':
		data = pd.read_csv(path + 'data/' + 'root_subgraph_features.csv')
		labs = data.index
		data = data.drop(todrop,1)
		k=8
	elif rootorsura is 'sura':
		data = pd.read_csv(path + 'data/' + 'sura_graph_features.csv')
		data = data.drop(['root_degree','root_in_degree','root_out_degree','root_freq','max_cooccurrence'],1)
		# labs = data.sura_name.values
		labs = data.sura.values
		todrop.extend(['sura_name','sura'])
		data = data.drop(todrop,1)
		k=5
	else:
		data = pd.read_csv(path + 'data/' + 'heatmap_features.csv')
		print(data.info())
		labs = data.sura.values
		data = data.drop(['sura'],1)
		if False:
			data = data.fillna(0).astype(bool).astype(int)
		k=3

	# print(data.info())
	
	# sns.pairplot(data)
	# plt.show()

	X = data.fillna(0).as_matrix()
	# print(X)

	if rootorsura is not 'root' and rootorsura is not 'sura':
		# run pca and visualise clusters
		import numpy as np
		from sklearn.decomposition import PCA
		# from sklearn.preprocessing import StandardScaler
		# X = StandardScaler().fit_transform(X)
		pca = PCA(n_components=3)#.fit(X)
		pcad = pca.fit_transform(X)
		print('Exp. variance ratio', pca.explained_variance_ratio_)
		print('Singular values', pca.singular_values_)
		pcadf = pd.DataFrame(data = pcad, columns = ['1', '2', '3'])

	if elbow:
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

	if silhouette:
		from sklearn.metrics import silhouette_samples, silhouette_score
		range_n_clusters = [2, 3, 4, 5, 6, 7, 8, 9, 10]

		for n_clusters in range_n_clusters:
		    # Create a subplot with 1 row and 2 columns
		    fig, (ax1, ax2) = plt.subplots(1, 2)
		    fig.set_size_inches(18, 7)

		    # The 1st subplot is the silhouette plot
		    # The silhouette coefficient can range from -1, 1 but in this example all
		    # lie within [-0.1, 1]
		    ax1.set_xlim([-0.1, 1])
		    # The (n_clusters+1)*10 is for inserting blank space between silhouette
		    # plots of individual clusters, to demarcate them clearly.
		    ax1.set_ylim([0, len(X) + (n_clusters + 1) * 10])

		    # Initialize the clusterer with n_clusters value and a random generator
		    # seed of 10 for reproducibility.
		    clusterer = KMeans(n_clusters=n_clusters, random_state=10)
		    cluster_labels = clusterer.fit_predict(X)

		    # The silhouette_score gives the average value for all the samples.
		    # This gives a perspective into the density and separation of the formed
		    # clusters
		    silhouette_avg = silhouette_score(X, cluster_labels)
		    print("For n_clusters =", n_clusters,
		          "The average silhouette_score is :", silhouette_avg)

		    # Compute the silhouette scores for each sample
		    sample_silhouette_values = silhouette_samples(X, cluster_labels)

		    y_lower = 10
		    for i in range(n_clusters):
		        # Aggregate the silhouette scores for samples belonging to
		        # cluster i, and sort them
		        ith_cluster_silhouette_values = \
		            sample_silhouette_values[cluster_labels == i]

		        ith_cluster_silhouette_values.sort()

		        size_cluster_i = ith_cluster_silhouette_values.shape[0]
		        y_upper = y_lower + size_cluster_i

		        color = cm.nipy_spectral(float(i) / n_clusters)
		        ax1.fill_betweenx(np.arange(y_lower, y_upper),
		                          0, ith_cluster_silhouette_values,
		                          facecolor=color, edgecolor=color, alpha=0.7)

		        # Label the silhouette plots with their cluster numbers at the middle
		        ax1.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))

		        # Compute the new y_lower for next plot
		        y_lower = y_upper + 10  # 10 for the 0 samples

		    ax1.set_title("The silhouette plot for the various clusters.")
		    ax1.set_xlabel("The silhouette coefficient values")
		    ax1.set_ylabel("Cluster label")

		    # The vertical line for average silhouette score of all the values
		    ax1.axvline(x=silhouette_avg, color="red", linestyle="--")

		    ax1.set_yticks([])  # Clear the yaxis labels / ticks
		    ax1.set_xticks([-0.1, 0, 0.2, 0.4, 0.6, 0.8, 1])
		    
		    # 2nd Plot showing the actual clusters formed
		    colors = cm.nipy_spectral(cluster_labels.astype(float) / n_clusters)

		    if rootorsura is 'root' or rootorsura is 'sura':
		    	ax2.scatter(data[col1], data[col2], marker='.', s=30, lw=0, alpha=0.7, c=colors, edgecolor='k')
		    else:
		    	ax2.scatter(pcadf['1'], pcadf['2'], marker='.', s=30, lw=0, alpha=0.7,
						c=colors, edgecolor='k')

		    # Labeling the clusters
		    centers = clusterer.cluster_centers_
		    # Draw white circles at cluster centers
		    ax2.scatter(centers[:, 0], centers[:, 1], marker='o',
		                c="white", alpha=1, s=200, edgecolor='k')

		    for i, c in enumerate(centers):
		        ax2.scatter(c[0], c[1], marker='$%d$' % i, alpha=1,
		                    s=50, edgecolor='k')

		    ax2.set_title("The visualization of the clustered data.")
		    ax2.set_xlabel("Feature space for the 1st feature")
		    ax2.set_ylabel("Feature space for the 2nd feature")

		    plt.suptitle(("Silhouette analysis for KMeans clustering on sample data "
		                  "with n_clusters = %d" % n_clusters),
		                 fontsize=14, fontweight='bold')

		plt.show();plt.close()

	if rootorsura is 'root' or rootorsura is 'sura':
		fig, ax = plt.subplots(figsize=(7,7))
		y_pred = KMeans(n_clusters=k, random_state=170).fit_predict(X)
		plt.scatter(data[col1], data[col2], c=y_pred, alpha=0.5)
		for n, xs, ys in zip(labs, data[col1], data[col2]):
			ax.text(xs, ys, n, alpha=0.6, size=10) ##bidialg.get_display(arabic_reshaper.reshape(n))
		plt.show(); plt.close()
	
	else:
		y_pred = KMeans(n_clusters=k, random_state=170).fit_predict(X)
		pcadf['cluster'] = y_pred 
		pcadf['sura'] = labs

		data['sura'] = labs
		data['cluster'] = y_pred 
		import seaborn as sns 
		sns.heatmap(data.sort_values('cluster'), xticklabels=30, yticklabels=4, cmap='Blues',center=50)#, linewidth=0.5)# cmap='YlGnBu')
		plt.show(); plt.close()

		qtoc = load_qtoc(path + 'data/')
		pcadf = qtoc.merge(pcadf, left_on='No.', right_on='sura')

		# pcadf = pcadf[pcadf['1']<-35]
		# pcadf = pcadf[pcadf['cluster']==1]
		print(pcadf)

		from mpl_toolkits.mplot3d import Axes3D 
		fig, ax = plt.subplots(figsize=(7,7))
		# fig = plt.figure(figsize=(7,7))
		ax = fig.add_subplot(111, projection='3d')
		# pcadf=pcadf[pcadf.Place == 'Meccan']
		colors = {0:'sienna',1:'gold',2:'olive',3:'green'}
		markers = {'Meccan':'o','Medinan':'^'}
		for s in set(pcadf.Place.values):
			print(s)
			t = pcadf[pcadf.Place == s]
			print(t.drop(['sura','Name','English Meaning','Place'],1))
			ax.scatter(t['1'], t['3'], t['2'], c=[colors[x] for x in t['cluster']], 
						alpha=0.7, s=t['No of verses'], #cmap=cm.viridis_r,
						marker=markers[s], edgecolor='silver')

		# for n, xs, ys, zs in zip(pcadf['Name'], pcadf['1'], pcadf['3'], pcadf['2']):
		# 	ax.text(xs, ys, zs, n, alpha=0.6, size=8)#, zorder=1) ##bidialg.get_display(arabic_reshaper.reshape(n))
	
		ax_labs = pca.explained_variance_ratio_
		ax.set_xlabel('PC1 (%.2f var. expl.)'%ax_labs[0])
		ax.set_ylabel('PC3 (%.2f var. expl.)'%ax_labs[2])
		ax.set_zlabel('PC2 (%.2f var. expl.)'%ax_labs[1])
		from matplotlib.lines import Line2D
		scatter1_proxy = Line2D([0],[0], linestyle="none", c=colors[0], marker = 'o')
		scatter2_proxy = Line2D([0],[0], linestyle="none", c=colors[1], marker = 'o')
		scatter3_proxy = Line2D([0],[0], linestyle="none", c=colors[2], marker = 'o')
		scatter4_proxy = Line2D([0],[0], linestyle="none", c='k', marker = markers['Medinan'])
		scatter5_proxy = Line2D([0],[0], linestyle="none", c='k', marker = markers['Meccan'])
		ax.legend([scatter1_proxy, scatter2_proxy, scatter3_proxy, scatter4_proxy, scatter5_proxy], 
					['Cluster 0', 'Cluster 1', 'Cluster 2', 'Medinan', 'Meccan'], 
					numpoints = 1)
		# plt.legend(set(pcadf['Place'].values))
		plt.title('Clusters of Suras on root frequency per sura (1642x114)')


	# rotate the axes and update
	# for angle in range(0, 360):
	#     ax.view_init(30, angle)
	#     plt.draw()
	#     plt.pause(.001)

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
