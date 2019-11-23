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


def create_ngrams(quran, col='Root_ar', n=2, separate=True, sep='786'):
	mask = quran.aya.diff() == 1
	if separate:
		a = np.insert(quran[col].values, mask.index[mask == True], sep)
	else:
		a = quran[col].values
	a = a[a!=0]
	tokens = [x for x in a if type(x) is str]
	ngrams = zip(*[tokens[i:] for i in range(n)])
	return ngrams


def create_graph_edges_dataframe(ngrams, u='one', v='two', sep='786', weight='count'):
	edges_df = pd.DataFrame(list(ngrams),columns=[u, v])#[" ".join(ngram) for ngram in ngrams])
	# edges_df = edges_df[edges_df.one != edges_df.two]#.sort_values(by='one')
	edges_df = edges_df[edges_df != sep].dropna().reset_index()
	edges_df[weight] = 1
	edges_df = edges_df.groupby([u,v]).sum().reset_index().sort_values(u)
	# edges_df[weight] = np.log2(edges_df[weight])*10+1
	print(edges_df.sort_values(weight))
	return edges_df


def create_graph(edges_df, u='one', v='two', attr_list=['count']):
	# G=nx.Graph()
	G = nx.from_pandas_edgelist(edges_df, \
	                      u, v, attr_list, create_using = nx.MultiDiGraph())#MultiDiGraph())
	# G.add_nodes_from([x for x in list(set(quran.Root_ar.values)) if type(x) is str])
	# print("Nodes of graph: ")
	# print(G.nodes())
	# print("Edges of graph: ")
	# print(G.edges())
	return G


def draw_graph(G, node_freq, nodesize_multiplier=40, weight='count', title=''):
	# shpsize = 30+10*np.log2([root_counts[x] for x in filter(lambda x: shapemap[x]==shp, G.nodes().keys())])
	shpsize = [node_freq[x]*nodesize_multiplier for x in G.nodes().keys()]

	# pos = nx.spring_layout(G, k=0.8, iterations=10, weight='count')
	# pos = nx.random_layout(G) #, weight='AMOUNT')
	# pos = nx.circular_layout(G) #, weight='AMOUNT')
	# pos = nx.shell_layout(G) #, weight='AMOUNT')
	# pos = nx.spectral_layout(G) #, weight='AMOUNT')
	maxkey = max(node_freq, key=node_freq.get)
	print(maxkey, node_freq[maxkey])
	# pos = nx.fruchterman_reingold_layout(G, pos= {str(maxkey): (0.1,0.1)}, fixed=[maxkey], k=0.7, iterations=10, weight=weight)
	pos = nx.fruchterman_reingold_layout(G, k=0.7, iterations=10, weight=weight)
	
	nodes = nx.draw_networkx_nodes(G, pos=pos, with_labels=True, node_color='r', alpha=0.3, node_size=shpsize)
	
	edges = nx.draw_networkx_edges(G, pos=pos, edge_color='g',alpha=0.2,width=[len(G[u][v])/2 for u,v in G.edges()])

	nodelabels = {}
	for idx, node in enumerate(G.nodes()):
		nodelabels[node] = bidialg.get_display(arabic_reshaper.reshape(node))
	labels = nx.draw_networkx_labels(G, pos, nodelabels, alpha=0.5, font_size = 9)

	plt.title(title)
	# plt.savefig("simple_path.png") # save as png
	plt.show() # display


def draw_subgraph(G, method='breadth', node_of_interest = '', node_freq=None, nodesize_multiplier=5):
	if method is 'breadth':
		I = nx.bfs_tree(G, node_of_interest, depth_limit=1)
	elif method is 'depth':
		I = nx.dfs_tree(G, node_of_interest)
	else:
		return 'Please select either \'depth\' or \'breadth\' for the parameter \'method\''
	I = G.subgraph(I.nodes())

	df = pd.DataFrame(index=I.nodes(), columns=I.nodes())
	for row, datum in nx.shortest_path_length(I):
	    for col, dist in datum.items():
	        df.loc[row,col] = dist

	df = df.fillna(df.max().max())

	dfspos = nx.kamada_kawai_layout(I, dist=df.to_dict())

	list(nx.dfs_labeled_edges(I,node_of_interest))

	shpsize = [node_freq[x]*nodesize_multiplier for x in I.nodes().keys()]

	nodes = nx.draw_networkx_nodes(I, pos=dfspos, with_labels=True, node_color='r', alpha=0.3, node_size=shpsize)
	
	edges = nx.draw_networkx_edges(I, pos=dfspos, edge_color='g',alpha=0.2, width=[len(I[u][v])/2 for u,v in I.edges()])

	nodelabels = {}
	for idx, node in enumerate(I.nodes()):
		nodelabels[node] = bidialg.get_display(arabic_reshaper.reshape(node))
	labels = nx.draw_networkx_labels(I, dfspos, nodelabels, alpha=0.5, font_size = 9)

	plt.title(bidialg.get_display(arabic_reshaper.reshape(node_of_interest))+ 
				' Freq: ' + str(node_freq[node_of_interest]) +
				', Roots: ' + str(len(I.nodes())) +
				', Cooccurrences: '+ str(len(I.edges())))
	plt.show()


###########################


if __name__ == '__main__':

	path = '/Users/ali.khan/Documents/qcm-analysis/data/'
	dat = pd.read_csv(path + 'root_subgraph_features.csv')
	import seaborn as sns 
	sns.set(style='white')
	# sns.pairplot(dat)
	# plt.show()

	cmap = sns.cubehelix_palette(dark=.3, light=.8, as_cmap=True)
	ax = sns.scatterplot(x="graph_order", y="graph_size",
	                      hue="avg degree", size="strongly_connected",
	                      # sizes=(20, 200), hue_norm=(0, 7),
	                      palette=cmap,
	                      data=dat)
	plt.show()

	# sur = 110
	# sz=5
	# if sur is not None:
	# 	sz=40
	# analysand = 'Root_ar'#'Lemma_ar'
	# node_of_interest = u'كون'#u'حقق'#u'ذكر'#'*kr'#'kwn'#'qwl' #u'ذكر' u'ارض'. 'حرم' 'فعل' 'حرم' 'ﻏﻀﺐ'
	# method = 'breadth'
	
	# quran, qtoc = load_data_from_csv(path = path)
	# if sur is not None:
	# 	quran = quran[quran.sura == sur].reset_index()

	# # second = first.groupby(['aya','word']).agg('FORM_ar').apply(''.join).reset_index()
	# bigrams = create_ngrams(quran, col=analysand, n=2, separate=True, sep='786')	

	# edges_df = create_graph_edges_dataframe(bigrams, u='one', v='two', sep='786', weight='count')

	# G = create_graph(edges_df, u='one', v='two', attr_list=['count'])

	# # import sys
	# # sys.exit()

	# root_counts = quran[analysand].value_counts().drop(0).to_dict()
	# # del root_counts[0]
	# print(node_of_interest, root_counts[node_of_interest])
	# # sns.distplot(quran[quran[analysand]!=0][analysand].value_counts(), bins=100)
	# # plt.show()

	# # FILTER THOSE NODES WHERE FREQ LESS THAN 40 AND THEN DRAW
	# filter = False
	# if filter:
	# 	t = quran[analysand].value_counts().drop(0)
	# 	# print(t[t>40])
	# 	# print(t[t<40].index.tolist())
	# 	for r in t[t<100].index.tolist():
	# 		# print(r)
	# 		# print(len(G.nodes()))
	# 		try:
	# 			G.remove_node(r)
	# 		except:
	# 			pass

	# if sur is not None:
	# 	title = str(*qtoc[qtoc['No.'] == sur].values.tolist())[1:-1] + ', ' + str(len(G.nodes())) +', '+ str(len(G.edges()))
	# else:
	# 	title = 'The Holy Quran, Roots: ' + str(len(G.nodes())) +', Cooccurrences: '+ str(len(G.edges()))
	
	# # draw_graph(G, node_freq=root_counts, nodesize_multiplier = sz, weight='count', title=title)

	# draw_subgraph(G, method=method, node_of_interest = node_of_interest, nodesize_multiplier = sz, node_freq=root_counts)
	# #############################################
	# # check how many times a given degree occurs:
	# degrees = [a + ' ' + str(G.degree[a]) for a in G.nodes]
	# # generate unique x-coordinates. divide by 2 for zero-centering: 
	# degrees = {degree: [a for a in degrees.count(degree)/2. - np.arange(degrees.count(degree))] for degree in set(degrees)}
	# # print(degrees)

	# # # build positioning dicitonary:
	# # positions = {a : (degrees[G.degree[a]].pop(), G.degree[a]) for a in G.nodes}
	# # degrees = [G.degree[a] for a in G.nodes]
	# # degrees_unique = sorted(list(set(degrees)))
	# # y_positions = {degrees_unique[i] : i for i in range(len(degrees_unique))}
	# # x_positions = {}
	# # for degree in degrees_unique:
	# # 	x_positions[degree] = [a for a in degrees.count(degree) / 2. - np.arange(degrees.count(degree))]

	# # positions = {}

	# # for node in G.nodes:
	# #     deg = G.degree[node]
	# #     positions[node] = (x_positions[deg].pop(), y_positions[deg])
	# ###################################