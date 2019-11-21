#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
This script visualises root data in terms of frequence or occurrence, its relationships with
other roots in a specific sura or the whole Quran in terms of cooccurrence or paired frequency

TODO
1. Add sura and aya numbers with pair freq or count (aka cooccurrence) as attributes of edges
2. Add frequency or count of single root to node attribute
3. Output that graph with edge and node attrib to graphml for cytoscape exploration
4. Graph properties like 

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
plt.rcParams['axes.facecolor'] = 'black'

from bidi import algorithm as bidialg
import arabic_reshaper

import networkx as nx

import qur_func


#######################


def load_corpus_dataframe_from_csv(path = '/Users/alikhan/Downloads/qur/'):
	quran = pd.read_csv(path + 'quran-morphology-final.csv', sep=",", header=0)#, index_col='Index')
	qtoc = pd.read_csv(path + 'toc.csv')
	qtoc['Name Arabic'] = qtoc['Name Arabic'].apply(lambda x: bidialg.get_display(arabic_reshaper.reshape(x)))
	quran = quran.fillna(0)
	print(quran.head())
	print(quran.info())

	# try:
	# 	test= quran[['Root','Lemma','FORM','FORM_ar']].loc[quran.Lemma.apply(lambda x: qur_func.contains_num(x)) == True]
	# 	test.Lemma = test.Lemma.str.replace('\d+', '')
	# 	test['Lemma_ar'] = test.apply(lambda x: qur_func.buck_to_arabic(x.Lemma), axis=1)

	# 	print(test)
	# except ValueError:
	# 	print('')

	return quran, qtoc


def create_ngrams(quran, col='Root_ar', n=2, separate=True, sep='786'):
	# mask = quran.aya.diff() == 1
	# if separate: # verse wise pairing, no roots paired across two different adjacent verses, i.e. last root of a verse 
	# # with the first root of the next verse is not paired, if separate = True
	# 	a = np.insert(quran[col].values, mask.index[mask == True], sep)
	# else:
	# 	a = quran[col].values
	# a = a[a!=0]
	# tokens = [x for x in a if type(x) is str]
	# ngrams = zip(*[tokens[i:] for i in range(n)])
	# return ngrams

	mask = quran.aya.diff() == 1
	if separate: # verse wise pairing, no roots paired across two different adjacent verses, i.e. last root of a verse 
	# with the first root of the next verse is not paired, if separate = True
		a = np.insert(quran[col].values, mask.index[mask == True], sep)
		b = np.insert(quran['position'].values, mask.index[mask == True], sep)
	else:
		a = quran[col].values
		b = quran['position'].values
	b = b[np.nonzero(a)]
	a = a[a!=0]
	assert(len(a)==len(b))
	ngrams = zip(*[a.tolist()[i:] for i in range(n)])
	ngrams_pos = zip(*[b.tolist()[i:] for i in range(n)])
	# print(len(list(ngrams)))
	return ngrams, ngrams_pos


def create_graph_edges_dataframe(ngrams, ngrams_pos, sep='786'):
	edges_df = pd.DataFrame(list(ngrams),columns=['one', 'two'])#[" ".join(ngram) for ngram in ngrams])
	pos_df = pd.DataFrame(list(ngrams_pos), columns=['pos','pos1'])
	assert(len(edges_df) == len(pos_df))
	edges = pd.concat([edges_df, pos_df], ignore_index=True, axis=1)
	edges_df = edges[edges != sep].dropna().reset_index().drop('index',1)
	edges_df['count'] = 1
	temp = edges_df.groupby([0,1]).sum().reset_index()#.sort_values('count')
	print(temp.sort_values('count'))
	edges_df = edges_df.drop('count',1).merge(temp, how='left', on=[0,1])
	if edges_df[2].equals(edges_df[3]):
		edges_df = edges_df.drop(3,1)
	edges_df = edges_df.rename(columns={0:'one', 1:'two', 2:'pos'})
	print(edges_df)
	# edges_df['count'] = np.log2(edges_df['count'])*10+1
	print('Total # edges: ', len(edges_df))
	print('Unique edges: ', len(temp))
	return edges_df, len(temp)


def create_graph(edges_df, node_freq, u='one', v='two'):
	# G=nx.Graph()
	G = nx.from_pandas_edgelist(edges_df, \
	                      u, v, edge_attr=True, create_using = nx.MultiDiGraph())#DiGraph())
	# G.add_nodes_from([x for x in list(set(quran.Root_ar.values)) if type(x) is str])
	# print("Nodes of graph: ")
	# print(G.nodes())
	# print("Edges of graph: ")
	# print(G.edges())

	nx.set_node_attributes(G, node_freq, 'freq')
	# print(G.nodes['دين']['freq'])
	# for u,v in G.edges():
	# 	print(G[u][v])

	N, K = G.order(), G.size()
	print('Graph order (# nodes)', N)
	print('Graph size (# edges)', K)
	print('Avg deg ',float(K)/N)
	print('Strongly connected: ',nx.number_strongly_connected_components(G))
	print('Weakly connected: ', nx.number_weakly_connected_components(G))
	# print(nx.average_node_connectivity(G))
	# ecc = nx.eccentricity(G)

	# in_degrees = pd.DataFrame(list(G.in_degree()),columns=['node','degree']) 
	# in_values = sorted(set(in_degrees.degree.values.tolist()))
	# in_hist = [in_degrees.degree.values.count(x) for x in in_values]

	# out_degrees = pd.DataFrame(list(G.out_degree()),columns=['node','degree']) 
	# out_values = sorted(set(out_degrees.degree.values.tolist()))
	# out_hist = [out_degrees.degree.values.count(x) for x in out_values]

	# plt.figure() # you need to first do 'import pylab as plt'
	# plt.grid(True)
	# plt.plot(in_values, in_hist, 'ro-') # in-degree
	# plt.plot(out_values, out_hist, 'bv-') # out-degree
	# plt.legend(['In-degree', 'Out-degree'])
	# plt.xlabel('Degree')
	# plt.ylabel('Number of nodes')
	# plt.xlim([0, 2*10**2])
	# # plt.savefig('./output/cam_net_degree_distribution.pdf')
	# plt.show()
	# plt.close()

	# degree_df = pd.DataFrame(list(G.degree()),columns=['node','degree'])
	# degree_df.hist(bins=100)
	# plt.show()
	
	# print(nx.center(G.to_undirected()))
	# # print(nx.diameter(G.to_undirected()))
	# # print(nx.radius(G.to_undirected()))
	# import sys
	# sys.exit()
	return G


def draw_graph(G, node_freq, nodesize_multiplier=40, weight='count', title=''):
	# shpsize = 30+10*np.log2([root_counts[x] for x in filter(lambda x: shapemap[x]==shp, G.nodes().keys())])
	# shpsize = [node_freq[x]*nodesize_multiplier*30 for x in G.nodes().keys()]

	# pos = nx.spring_layout(G, k=0.7, iterations=10, weight='count')
	# pos = nx.random_layout(G) #, weight='AMOUNT')
	# pos = nx.circular_layout(G) #, weight='AMOUNT')
	# pos = nx.shell_layout(G) #, weight='AMOUNT')
	# pos = nx.spectral_layout(G) #, weight='AMOUNT')
	maxkey = max(node_freq, key=node_freq.get) 
	# pos = nx.fruchterman_reingold_layout(G, pos= {str(maxkey): (0.1,0.1)}, fixed=[maxkey], k=0.7, iterations=10, weight=weight)
	pos = nx.fruchterman_reingold_layout(G, k=0.5, iterations=10, weight=weight)
	
	fig = plt.figure(figsize=(40.10, 20.80))	

	node_labels = dict((n,d['freq']) for n,d in G.nodes(data=True))
	# print(G.nodes(data=True))
	# print(node_labels)
	# print(G.edges(data=True))
	nodes = nx.draw_networkx_nodes(G, 
									pos=pos,
									with_labels=True, 
									# labels=node_labels,
									node_color='silver', 
									alpha=0.6, 
									node_size=[G.nodes[x]['freq']*nodesize_multiplier*2 for x in G.nodes().keys()])
	
	edges = nx.draw_networkx_edges(G, pos=pos, edge_color='gold',alpha=0.3)#width=[len(G[u][v])/2 for u,v in G.edges()])

	nodelabels = {}
	for idx, node in enumerate(G.nodes()):
		nodelabels[node] = bidialg.get_display(arabic_reshaper.reshape(node)) + '\n' + str(G.nodes[node]['freq'])
	labels = nx.draw_networkx_labels(G, pos, nodelabels, font_color='white',alpha=0.7, font_size = 9)
	
	# edglab = nx.get_edge_attributes(G, 'pos')
	edglab = dict([((u,v,),d['pos'])
             for u,v,d in G.edges(data=True)])
	# print(edglab)
	edge_labels = nx.draw_networkx_edge_labels(G, pos, edglab, font_size=7, 
												bbox=dict(facecolor='none', edgecolor='none', pad=0.0), 
												font_color='white', alpha=0.6 )

	plt.title(title)
	# plt.savefig('simple_path.png') # save as png
	plt.savefig('./fig/graph.png', dpi=300, facecolor='w', edgecolor='w',
    orientation='landscape', papertype=None, format='png', transparent=False, bbox_inches=None, pad_inches=0.1,
    frameon=None)
	plt.show() # display


def create_subgraph_from_edges_dataframe(edges_df, node_of_interest='', u='one', v='two', weight='count'):
	edges_df['count'] = 1
	sg_edges_df = edges_df[(edges_df.one == node_of_interest) | (edges_df.two == node_of_interest)].dropna().reset_index().drop('index',1)
	temp = sg_edges_df.groupby([u,v]).sum().reset_index()#.sort_values(weight)
	sg_edges_df = sg_edges_df.drop('count',1).merge(temp, how='left', on=['one','two'])
	print(sg_edges_df)
	print(temp.sort_values(weight))
	print('Total # edges: ', len(sg_edges_df))
	print('Unique edges: ', len(temp))
	I = nx.from_pandas_edgelist(sg_edges_df, u, v, edge_attr=True, create_using = nx.MultiDiGraph())

	# nx.set_node_attributes(G, node_freq, 'freq')
	# for u,v in I.edges():
	# 	print(I[u][v])
	# print(G.nodes(data=True))
	nx.set_node_attributes(I, dict([(n,d['freq']) for n,d in G.nodes(data=True)]), 'freq')
	N, K = I.order(), I.size()
	print('Subgraph order (# nodes)', N)
	print('Subraph size (# edges)', K)
	print('Avg deg ',float(K)/N)
	print('Strongly connected: ',nx.number_strongly_connected_components(I))
	print('Weakly connected: ', nx.number_weakly_connected_components(I))

	return I


def create_subgraph(G, method='breadth', node_of_interest = ''): 
# DEPRECATED: This is non-purist subgraph
# it gives depth level 1 root neighbours and their interconnections too instead of only the 
# root or node of interest and its connection with level 1 neighbours 
	if method is 'breadth':
		I = nx.bfs_tree(G, node_of_interest, depth_limit=1)
	elif method is 'depth':
		I = nx.dfs_tree(G, node_of_interest)
	else:
		return 'Please select either \'depth\' or \'breadth\' for the parameter \'method\''
	I = G.subgraph(I.nodes())
	# print(G[node_of_interest])

	# if method is 'breadth':	
	# 	print(list(nx.bfs_labeled_edges(I,node_of_interest)))
	# else:
	# 	print(list(nx.dfs_labeled_edges(I,node_of_interest)))

	# import sys
	# sys.exit()
	return I


def draw_subgraph(I, node_of_interest = '', node_freq=None, nodesize_multiplier=5, title= ''):
	# df = pd.DataFrame(index=I.nodes(), columns=I.nodes())
	# for row, datum in nx.shortest_path_length(I):
	#     for col, dist in datum.items():
	#         df.loc[row,col] = dist

	# df = df.fillna(df.max().max())

	# dfspos = nx.kamada_kawai_layout(I, dist=df.to_dict())

	dfspos = nx.fruchterman_reingold_layout(I, k=1, iterations=10)

	# shpsize = [node_freq[x]*nodesize_multiplier for x in I.nodes().keys()]

	fig = plt.figure(figsize=(15.10, 20.80))	
	nodes = nx.draw_networkx_nodes(I, pos=dfspos, with_labels=True, 
									node_color='gold', 
									alpha=0.7, 
									node_size=[I.nodes[x]['freq']*nodesize_multiplier*2 for x in I.nodes().keys()])
	# print([I[u][v]['count'] for u,v in I.edges()])
	edges = nx.draw_networkx_edges(I, pos=dfspos, edge_color='silver',alpha=0.2)#, width=[len(I[u][v])/2 for u,v in I.edges()])

	nodelabels = {}
	for idx, node in enumerate(I.nodes()):
		nodelabels[node] = bidialg.get_display(arabic_reshaper.reshape(node)) + '\n' + str(I.nodes[node]['freq'])
	labels = nx.draw_networkx_labels(I, dfspos, nodelabels, font_color='white', alpha=0.8, font_size = 10)

	# edglab = nx.get_edge_attributes(G, 'pos')
	edglab = dict([((u,v,),d['pos'])
             for u,v,d in I.edges(data=True)])
	# print(edglab)
	edge_labels = nx.draw_networkx_edge_labels(I, dfspos, edglab, font_size=7, 
												bbox=dict(facecolor='none', edgecolor='none', pad=0.0), 
												font_color='white', alpha=0.6 )

	plt.title('Root: ' + bidialg.get_display(arabic_reshaper.reshape(node_of_interest))+ 
				' [Freq: ' + str(node_freq[node_of_interest]) +
				', Roots: ' + str(len(I.nodes())) +
				', Cooccurrences: '+ str(len(I.edges())) + ']\nIn ' +
				title)
	plt.savefig('./fig/'+qur_func.arabic_to_buc(node_of_interest) + '.png', dpi=300, facecolor='w', edgecolor='w',
    orientation='landscape', papertype=None, format='png', transparent=False, bbox_inches=None, pad_inches=0.1,
    frameon=None)
	plt.show()
	# from networkx.drawing.nx_agraph import write_dot
	# write_dot(I, 'test.dot')

###########################


###########################




if __name__ == '__main__':

	path = '/Users/alikhan/Downloads/qur/qcm-analysis/'
	sur =None
	sz=15
	if sur is not None:
		sz=40
	analysand = 'Root_ar'#'Lemma_ar'
	node_of_interest = u'دين'#'فقه'#'كون'#'دين'#'فقه'#'شرع'#u'حقق'#u'ذكر'#'*kr'#'kwn'#'qwl' #u'ذكر' u'ارض'. 'حرم' 'فعل' 'حرم' 'ﻏﻀﺐ'
	method = 'breadth'
	
	quran, qtoc = load_corpus_dataframe_from_csv(path = path)
	if sur is not None:
		quran = quran[quran.sura == sur].reset_index()

	quran['position'] = quran['sura'].astype(str) + ':' + quran['aya'].astype(str)

	# second = first.groupby(['aya','word']).agg('FORM_ar').apply(''.join).reset_index()
	bigrams, bigrams_pos = create_ngrams(quran, col=analysand, n=2, separate=True, sep='786')	

	edges_df, lenuniq = create_graph_edges_dataframe(bigrams, bigrams_pos, sep='786')

	root_counts = quran[analysand].value_counts().drop(0).to_dict()
	# del root_counts[0]
	print(node_of_interest, root_counts[node_of_interest])
	# sns.distplot(quran[quran[analysand]!=0][analysand].value_counts(), bins=100)
	# plt.show()

	G = create_graph(edges_df, root_counts)

	# FILTER THOSE NODES WHERE FREQ LESS THAN 40 AND THEN DRAW
	filter = False
	if filter:
		t = quran[analysand].value_counts().drop(0)
		# print(t[t>10])
		# print(t[t<40].index.tolist())
		for r in t[t<11].index.tolist():
			# print(r)
			# print(len(G.nodes()))
			try:
				G.remove_node(r)
			except:
				pass

	if sur is not None:
		title = str(*qtoc[qtoc['No.'] == sur].values.tolist())[1:-1] + '\nRoots: ' + str(len(G.nodes())) + \
				', Cooccurrences: '+ str(len(G.edges())) + ', Unique cooccurrences: ' + str(lenuniq)
	else:
		title = 'The Holy Quran [Roots: ' + str(len(G.nodes())) +', Cooccurrences: '+ str(len(G.edges())) + \
				 ', Unique cooccurrences: ' + str(lenuniq) + ']'
	
	# nx.write_graphml(G, path + 'test.graphml')
	draw_graph(G, node_freq=root_counts, nodesize_multiplier=sz, weight='count', title=title)

	# I = create_subgraph(G, method=method, node_of_interest=node_of_interest)
	I = create_subgraph_from_edges_dataframe(edges_df, node_of_interest=node_of_interest)
	draw_subgraph(I, node_of_interest=node_of_interest, nodesize_multiplier = sz, node_freq=root_counts, title= title)
	

	#############################################


	# check how many times a given degree occurs:
	degrees = [a + ' ' + str(G.degree[a]) for a in G.nodes]
	# generate unique x-coordinates. divide by 2 for zero-centering: 
	degrees = {degree: [a for a in degrees.count(degree)/2. - np.arange(degrees.count(degree))] for degree in set(degrees)}
	# print(degrees)

	# # build positioning dicitonary:
	# positions = {a : (degrees[G.degree[a]].pop(), G.degree[a]) for a in G.nodes}
	# degrees = [G.degree[a] for a in G.nodes]
	# degrees_unique = sorted(list(set(degrees)))
	# y_positions = {degrees_unique[i] : i for i in range(len(degrees_unique))}
	# x_positions = {}
	# for degree in degrees_unique:
	# 	x_positions[degree] = [a for a in degrees.count(degree) / 2. - np.arange(degrees.count(degree))]

	# positions = {}

	# for node in G.nodes:
	#     deg = G.degree[node]
	#     positions[node] = (x_positions[deg].pop(), y_positions[deg])
	###################################