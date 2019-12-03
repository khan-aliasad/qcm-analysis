#!/usr/bin/env python

"""
This script visualises root data in terms of the forms that occur with it in all mentions 
in a specific sura or the whole Quran
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

import networkx as nx

import qur_func


#######################


def load_data_from_csv(path = '/Users/alikhan/Downloads/qur/'):
	quran = pd.read_csv(path + 'quran-morphology-final.csv', sep=",", header=0)#, index_col='Index')
	qtoc = pd.read_csv(path + 'toc.csv')
	qtoc['Name Arabic'] = qtoc['Name Arabic'].apply(lambda x: bidialg.get_display(arabic_reshaper.reshape(x)))
	quran = quran.fillna(0)
	print(quran.head())
	print(quran.info())
	return quran, qtoc


###########################


if __name__ == '__main__':

	path = '/Users/ali.khan/Documents/qcm-analysis/'

	if False:
		import seaborn as sns 
		sns.set(style='white')
		data = pd.read_csv(path + 'data/' + 'root_subgraph_features.csv')
		print(data.info())
		# sns.pairplot(data)
		# plt.show()

		# cmap = sns.cubehelix_palette(dark=.3, light=.8, as_cmap=True)
		# ax = sns.scatterplot(x="graph_order", y="graph_size",
		#                       hue="avg degree", size="strongly_connected",
		#                       # sizes=(20, 200), hue_norm=(0, 7),
		#                       palette=cmap,
		#                       data=data)
		# plt.show()

		data['degree_ratio'] = data.in_degree / data.out_degree
		data['f1'] = data.graph_size/ data.unique_edges * data.freq / data.density
		# cmap = sns.cubehelix_palette(dark=.3, light=.8, as_cmap=True)
		# ax = sns.scatterplot(x="unique_edges", y="degree_ratio",
		#                       size="freq", hue="max_cooccurrence",
		#                       # sizes=(20, 200), hue_norm=(0, 7),
		#                       palette=cmap,
		#                       data=data)
		# data = data[data.freq <40]#100]#500]
		# data = data[data.unique_edges <50]#100]#250]
		# data = data[data.max_cooccurrence <4]#10]#50]
		from mpl_toolkits.mplot3d import Axes3D 
		import matplotlib.animation as animation

		fig = plt.figure(figsize=(7,7))
		ax = fig.add_subplot(111, projection='3d')
		x='freq'
		y='avg_degree'#'unique_edges'
		z='max_cooccurrence'
		sz = 'degree_ratio'#f1
		ax.scatter(data[x], data[y], data[z], marker='o', 
					s=data[sz], 
					alpha=0.3, c="goldenrod")
		for n, xs, ys, zs in zip(data['root'], data[x], data[y], data[z]):
		    # label = '(%d, %d, %d), dir=%s' % (xs, ys, zs, n)
		    # print(label)
		    ax.text(xs, ys, zs, 
				bidialg.get_display(arabic_reshaper.reshape(n)), 
		    	alpha=0.4, size=11)

		ax.set_xlabel(x)
		ax.set_ylabel(y)
		ax.set_zlabel(z)
		plt.legend(loc=2, prop={'size': 6})
		plt.show()

	sur = 1
	sz=5
	if sur is not None:
		sz=40
	analysand = 'Root_ar'#'Lemma_ar'
	node_of_interest = u'كون'#u'حقق'#u'ذكر'#'*kr'#'kwn'#'qwl' #u'ذكر' u'ارض'. 'حرم' 'فعل' 'حرم' 'ﻏﻀﺐ'
	method = 'breadth'
	
	quran, qtoc = load_data_from_csv(path = path + 'data/')
	if sur is not None:
		quran = quran[quran.sura == sur].reset_index()

	# second = quran.groupby(['aya','word']).agg('FORM_ar').apply(''.join).reset_index()
	I = nx.read_graphml(path + 'graphml/' + qur_func.arabic_to_buc(node_of_interest) + '.graphml')
	root_in_edges = pd.DataFrame.from_dict(nx.get_edge_attributes(I, 'pos'), orient='index', columns=['pos'])
	root_in_edges[['sura','aya']] = root_in_edges['pos'].str.split(':', expand=True).astype('int64')
	print(qtoc.info())
	print(root_in_edges.info())
	print(root_in_edges.head())
	d = root_in_edges.reset_index() \
		.drop('pos',1) \
		.merge(qtoc[['No.','Place','Chronology']], 
			left_on='sura',
			right_on='No.', 
			how='left') \
		.drop('No.',1)
	print(d.head())
	print(len(d[d.Place == 'Meccan']), np.sort(d[d.Place == 'Meccan'].sura.unique()))
	print(len(d[d.Place == 'Medinan']), np.sort(d[d.Place == 'Medinan'].sura.unique()))
	import sys
	sys.exit()

	bigrams = create_ngrams(quran, col=analysand, n=2, separate=True, sep='786')	
	edges_df = create_graph_edges_dataframe(bigrams, u='one', v='two', sep='786', weight='count')
	G = create_graph(edges_df, u='one', v='two', attr_list=['count'])


	root_counts = quran[analysand].value_counts().drop(0).to_dict()
	# del root_counts[0]
	print(node_of_interest, root_counts[node_of_interest])
	# sns.distplot(quran[quran[analysand]!=0][analysand].value_counts(), bins=100)
	# plt.show()

	# FILTER THOSE NODES WHERE FREQ LESS THAN 40 AND THEN DRAW
	filter = False
	if filter:
		t = quran[analysand].value_counts().drop(0)
		# print(t[t>40])
		# print(t[t<40].index.tolist())
		for r in t[t<100].index.tolist():
			# print(r)
			# print(len(G.nodes()))
			try:
				G.remove_node(r)
			except:
				pass

	if sur is not None:
		title = str(*qtoc[qtoc['No.'] == sur].values.tolist())[1:-1] + ', ' + str(len(G.nodes())) +', '+ str(len(G.edges()))
	else:
		title = 'The Holy Quran, Roots: ' + str(len(G.nodes())) +', Cooccurrences: '+ str(len(G.edges()))
	
	# draw_graph(G, node_freq=root_counts, nodesize_multiplier = sz, weight='count', title=title)

	draw_subgraph(G, method=method, node_of_interest = node_of_interest, nodesize_multiplier = sz, node_freq=root_counts)
	#############################################
	# check how many times a given degree occurs:
	degrees = [a + ' ' + str(G.degree[a]) for a in G.nodes]
	# generate unique x-coordinates. divide by 2 for zero-centering: 
	degrees = {degree: [a for a in degrees.count(degree)/2. - np.arange(degrees.count(degree))] for degree in set(degrees)}
	print(degrees)

	# build positioning dicitonary:
	positions = {a : (degrees[G.degree[a]].pop(), G.degree[a]) for a in G.nodes}
	degrees = [G.degree[a] for a in G.nodes]
	degrees_unique = sorted(list(set(degrees)))
	y_positions = {degrees_unique[i] : i for i in range(len(degrees_unique))}
	x_positions = {}
	for degree in degrees_unique:
		x_positions[degree] = [a for a in degrees.count(degree) / 2. - np.arange(degrees.count(degree))]

	positions = {}

	for node in G.nodes:
	    deg = G.degree[node]
	    positions[node] = (x_positions[deg].pop(), y_positions[deg])
	##################################