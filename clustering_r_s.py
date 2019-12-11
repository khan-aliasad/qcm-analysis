
#!/usr/bin/env python

"""
This script clusters roots features matrix and separately, sura features matrix

"""

__author__ = "Ali Khan"
__license__ = "GPL"
__version__ = "0.0.2"
__maintainer__ = "Ali Khan"
__email__ = "khan.aliasad@gmail.com"
__status__ = "dev"


import pandas as pd 

import matplotlib.pyplot as plt
import seaborn as sns 
sns.set(style='whitegrid')

from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

from qur_func import load_qtoc
from util_func import *

plt.rcParams['axes.facecolor'] = 'white'

#######################


if __name__ == '__main__':

	path = '/Users/alikhan/Downloads/qur/qcm-analysis/'

	k=3
	elbow = False 
	silhouette = False

	draw_fig = True

	data = pd.read_csv(path + 'data/' + 'heatmap_features.csv')
	print(data.info())

	qtoc = load_qtoc(path + 'data/')

	labs = data.sura.values
	data = data.drop(['sura'],1)
	if True:
		data = data.fillna(0).astype(bool).astype(int)

	X = data.fillna(0).as_matrix()
	# print(X)

	# from sklearn.preprocessing import StandardScaler
	# X = StandardScaler().fit_transform(X)
	pca = PCA(n_components=3)#.fit(X)
	pcad = pca.fit_transform(X)
	print('Exp. variance ratio', pca.explained_variance_ratio_)
	print('Singular values', pca.singular_values_)
	pcadf = pd.DataFrame(data = pcad, columns = ['1', '2', '3'])

	if elbow:
		run_elbow_method(X)

	if silhouette:
		run_silhouette_method(X, col1=pcadf['1'], col2=pcadf['2'])
	
	y_pred = KMeans(n_clusters=k, random_state=170).fit_predict(X)
	pcadf['cluster'] = y_pred; pcadf['sura'] = labs
	data['sura'] = labs
	# data = data.fillna(0).set_index('sura')
	# data['sura_sums'] = data.sum(axis=1, skipna=True)
	# data = data.sort_values('sura_sums', ascending=False)
	# data = data.transpose()
	# data['root_sums'] = data.sum(axis=1, skipna=True)
	# data = data.sort_values('root_sums', ascending=False)
	# print(data.drop('root_sums',1).transpose().drop('sura_sums',1))

	# if False:
	# 	sns.heatmap(data.astype(int).drop('root_sums',1).transpose().drop('sura_sums',1), xticklabels=30, yticklabels=4)
	# 	plt.show(); plt.close()
	data['cluster'] = y_pred 

	sns.heatmap(data.sort_values('cluster'), xticklabels=30, yticklabels=4, cmap='Blues',center=50)#, linewidth=0.5)# cmap='YlGnBu')
	plt.show(); plt.close()

	pcadf = qtoc.merge(pcadf, left_on='No.', right_on='sura')

	######### Redo clustering on cluster 1 with 87 suras
	orig = data.copy()
	orig_pcadf = pcadf.copy()
	print(orig.info())

	data = data[data.cluster==1]

	data = data.fillna(0).set_index('sura')
	data['sura_sums'] = data.sum(axis=1, skipna=True)
	data = data.sort_values('sura_sums', ascending=False)
	data = data.transpose()
	data['root_sums'] = data.sum(axis=1, skipna=True)
	data = data.sort_values('root_sums', ascending=False)
	data = data.drop('root_sums',1).transpose().drop('sura_sums',1)
	data = data.loc[:, (data != 0).any(axis=0)]
	print(data)

	# if True:
	sns.heatmap(data.drop('cluster',1), xticklabels=30, yticklabels=4, center=50)
	plt.show(); plt.close()


	data = data.reset_index()
	labs = data.sura.values
	data = data.drop(['sura','cluster'],1)

	X = data.fillna(0).as_matrix()

	pca = PCA(n_components=3)#.fit(X)
	pcad = pca.fit_transform(X)
	print('Exp. variance ratio', pca.explained_variance_ratio_)
	print('Singular values', pca.singular_values_)
	pcadf = pd.DataFrame(data = pcad, columns = ['1', '2', '3'])

	if elbow:
		run_elbow_method(X)
	if silhouette:
		run_silhouette_method(X, col1=pcadf['1'], col2=pcadf['2'])
	
	y_pred = KMeans(n_clusters=k, random_state=170).fit_predict(X)
	pcadf['cluster'] = y_pred;	pcadf['sura'] = labs
	data['sura'] = labs; data['cluster'] = y_pred 

	sns.heatmap(data.sort_values('cluster'), xticklabels=30, yticklabels=4, cmap='Blues',center=50)#, linewidth=0.5)# cmap='YlGnBu')
	plt.show(); plt.close()

	pcadf = qtoc.merge(pcadf, left_on='No.', right_on='sura')

	# pcadf = pcadf[pcadf['1']<-35]
	# pcadf = pcadf[pcadf['cluster']==1]

	orig = orig[orig.cluster!=1]
	data.cluster = data.cluster + 3
	final = orig.append(data).set_index('sura').sort_index()
	final.to_csv(path + 'data/heatmap_features_km_clusters_six.csv')

	orig_pcadf = orig_pcadf[orig_pcadf.cluster!=1]
	pcadf.cluster = pcadf.cluster + 3
	pcadf = orig_pcadf.append(pcadf).sort_values('sura')
	pcadf.to_csv(path + 'data/heatmap_features_km_clusters_six_pcad.csv')


	if draw_fig:
		from mpl_toolkits.mplot3d import Axes3D 
		fig, ax = plt.subplots(figsize=(7,7))
		# fig = plt.figure(figsize=(7,7))
		ax = fig.add_subplot(111, projection='3d')
		# pcadf=pcadf[pcadf.Place == 'Meccan']
		colors = {0:'sienna',1:'gold',2:'olive',3:'green',4:'gold',5:'purple'}
		markers = {'Meccan':'o','Medinan':'^'}
		for s in set(pcadf.Place.values):
			print(s)
			t = pcadf[pcadf.Place == s]
			print(t.drop(['sura','Name','English Meaning','Place'],1))
			ax.scatter(t['1'], t['3'], t['2'], c=[colors[x] for x in t['cluster']], 
						alpha=0.5, s=t['No of verses']*2, #cmap=cm.viridis_r,
						marker=markers[s], edgecolor='silver')

		for n, xs, ys, zs in zip(pcadf['Name Arabic'], pcadf['1'], pcadf['3'], pcadf['2']):
			ax.text(xs, ys, zs, n, alpha=0.6, size=8)#, zorder=1) ##bidialg.get_display(arabic_reshaper.reshape(n))

		ax_labs = pca.explained_variance_ratio_
		ax.set_xlabel('PC1 (%.2f var. expl.)'%ax_labs[0])
		ax.set_ylabel('PC3 (%.2f var. expl.)'%ax_labs[2])
		ax.set_zlabel('PC2 (%.2f var. expl.)'%ax_labs[1])
		from matplotlib.lines import Line2D
		scatter1_proxy = Line2D([0],[0], linestyle="none", c=colors[0], marker = 'o')
		scatter2_proxy = Line2D([0],[0], linestyle="none", c=colors[3], marker = 'o')
		scatter3_proxy = Line2D([0],[0], linestyle="none", c=colors[2], marker = 'o')
		scatter6_proxy = Line2D([0],[0], linestyle="none", c=colors[4], marker = 'o')
		scatter7_proxy = Line2D([0],[0], linestyle="none", c=colors[5], marker = 'o')
		scatter4_proxy = Line2D([0],[0], linestyle="none", c='k', marker = markers['Medinan'])
		scatter5_proxy = Line2D([0],[0], linestyle="none", c='k', marker = markers['Meccan'])
		ax.legend([scatter1_proxy, scatter3_proxy, scatter2_proxy, scatter6_proxy, scatter7_proxy, scatter4_proxy, scatter5_proxy], 
					['Cluster 0', 'Cluster 2', 'Cluster 3', 'Cluster 4', 'Cluster 5', 'Medinan', 'Meccan'], 
					numpoints = 1)
		# plt.legend(set(pcadf['Place'].values))
		plt.title('Clusters of Suras on root frequency per sura (1642x114)')


		# rotate the axes and update
		for angle in range(0, 360):
		    ax.view_init(30, angle)
		    plt.draw()
		    plt.pause(.001)

		plt.show();plt.close()
