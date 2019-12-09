#!/usr/bin/env python

"""
This script visualises root data in terms of the forms that occur with it in all mentions 
in a specific sura or the whole Quran

TODO:
Get all mentions in forms of the root
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

from qur_func import arabic_to_buc

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

	path = '/Users/alikhan/Downloads/qur/qcm-analysis/'

	if False:
		import seaborn as sns 
		sns.set(style='white')
		data = pd.read_csv(path + 'data/' + 'root_subgraph_features.csv')
		print(data.info())
		# sns.pairplot(data)
		# plt.show()

		cmap = sns.cubehelix_palette(dark=.3, light=.8, as_cmap=True)
		ax = sns.scatterplot(x="graph_order", y="graph_size",
		                      hue="avg_degree", size="strongly_connected",
		                      # sizes=(20, 200), hue_norm=(0, 7),
		                      palette=cmap,
		                      data=data)
		plt.show()

		data['degree_ratio'] = data.out_degree / data.in_degree
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

		print(data.info())
		fig = plt.figure(figsize=(7,7))
		ax = fig.add_subplot(111, projection='3d')
		x='freq'
		y='graph_order'#'avg_degree'#'unique_edges'
		z='max_cooccurrence'
		sz = 'degree_ratio'#f1
		ax.scatter(data[x], data[y], data[z], marker='o', 
					s=data[sz]*30, 
					alpha=0.6, c="goldenrod")
		for n, xs, ys, zs in zip(data['root'], data[x], data[y], data[z]):
		    # label = '(%d, %d, %d), dir=%s' % (xs, ys, zs, n)
		    # print(label)
		    ax.text(xs, ys, zs, 
				n,#bidialg.get_display(arabic_reshaper.reshape(n)), 
		    	alpha=0.6, size=10)

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
	I = nx.read_graphml(path + 'graphml/' + arabic_to_buc(node_of_interest) + '.graphml')
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

	d['counter'] = 1
	print(d.groupby(['sura','aya','Place']).agg(sum).reset_index())
	print(d.groupby(['sura','aya','Place']).agg(sum).reset_index().counter.max())
	print(d.groupby(['sura','aya','Place']).agg(sum).reset_index().groupby('Place').agg(sum).reset_index())
