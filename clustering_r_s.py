
#!/usr/bin/env python

"""
This script clusters roots features matrix and separately, sura features matrix

"""

__author__ = "Ali Khan"
__license__ = "GPL"
__version__ = "0.0.3"
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

from random import randint

plt.rcParams['axes.facecolor'] = 'white'

#######################


if __name__ == '__main__':

	path = '/Users/alikhan/Downloads/qur/qcm-analysis/'

	k=3
	rs = randint(0, 200)
	merge_with_graph_feat = True 
	recluster = True 
	clusnum = 1

	draw_fig = False

	elbow = False 
	silhouette = False

	occurrence_or_frequency = 'frequency'

	data = pd.read_csv(path + 'data/' + 'heatmap_features.csv')
	print(data.info())

	if merge_with_graph_feat:
		graph_feat = pd.read_csv(path + 'data/' + 'sura_graph_features.csv')
		graph_feat = graph_feat.drop(['Unnamed: 0', 'root', 
										'root_degree','root_in_degree',
										'root_out_degree', 'root_freq',
										'center', 'sura_name'],1)
		data = graph_feat.merge(data, on='sura')
		print(data.info())

	qtoc = load_qtoc(path + 'data/')

	labs = data.sura.values
	data = data.drop(['sura'],1)
	
	if occurrence_or_frequency is 'occurrence':
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
	
	y_pred = KMeans(n_clusters=k, random_state=rs).fit_predict(X)
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

	if recluster:
		######### Redo clustering on cluster 1 with 87 suras
		orig = data.copy()
		orig_pcadf = pcadf.copy()
		print(orig.info())

		data = data[data.cluster==clusnum]

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
		
		y_pred = KMeans(n_clusters=k, random_state=rs).fit_predict(X)
		pcadf['cluster'] = y_pred;	pcadf['sura'] = labs
		data['sura'] = labs; data['cluster'] = y_pred 

		sns.heatmap(data.sort_values('cluster'), xticklabels=30, yticklabels=4, cmap='Blues',center=50)#, linewidth=0.5)# cmap='YlGnBu')
		plt.show(); plt.close()

		pcadf = qtoc.merge(pcadf, left_on='No.', right_on='sura')

		# pcadf = pcadf[pcadf['1']<-35]
		# pcadf = pcadf[pcadf['cluster']==1]

		orig = orig[orig.cluster!=clusnum]
		data.cluster = data.cluster + 3
		final = orig.append(data).set_index('sura').sort_index()
		data = final.copy()

		orig_pcadf = orig_pcadf[orig_pcadf.cluster!=clusnum]
		pcadf.cluster = pcadf.cluster + 3
		pcadf = orig_pcadf.append(pcadf).sort_values('sura')
		######### END CODE BLOCK Redo clustering on cluster 1 with 87 suras


	data.to_csv(path + 'data/heatmap_features_km_clusters_six.csv')
	pcadf.to_csv(path + 'data/heatmap_features_km_clusters_six_pcad.csv')


	if merge_with_graph_feat:
		# pcadf[['sura','cluster']].to_csv(path + 'data/pcadf_graph+heatmap_feat_cluster_assignment.csv')	
		data.reset_index()[['sura','cluster']].to_csv(path + 'data/final_graph+heatmap_feat_cluster_assignment.csv')
	else:
		# pcadf[['sura','cluster']].to_csv(path + 'data/pcadf_heatmap_feat_cluster_assignment.csv')	
		data.reset_index()[['sura','cluster']].to_csv(path + 'data/final_heatmap_feat_cluster_assignment.csv')


	if False:
		print(pcadf.set_index('sura').sort_index()['cluster'].values-data.sort_index()['cluster'].values)


	clusters = pd.read_csv(path + 'data/final_heatmap_feat_cluster_assignment.csv').reset_index().set_index('sura')
	clusters_gr = pd.read_csv(path + 'data/final_graph+heatmap_feat_cluster_assignment.csv').reset_index().set_index('sura')

	print(clusters['cluster'].value_counts())	
	print(clusters_gr['cluster'].value_counts())

	clusters['cluster'].value_counts().plot(kind='bar'); plt.show(); plt.close()
	clusters_gr['cluster'].value_counts().plot(kind='bar'); plt.show(); plt.close()


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
