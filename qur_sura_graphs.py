#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
This script visualises sura data in a loop for 114 suras and creates graphs to study 
graph properties per sura and save those in a feature matrix as CSV 

TODO
Loop through all suras and create graphs for suras
Save graph topology viz
Save graph ml
Save graph props in a feature matrix  
"""

__author__ = "Ali Khan"
__license__ = "GPL"
__version__ = "0.0.3"
__maintainer__ = "Ali Khan"
__email__ = "khan.aliasad@gmail.com"
__status__ = "dev"


# import pandas as pd 
# import numpy as np
# import matplotlib.pyplot as plt
# plt.rcParams['axes.facecolor'] = 'black'

# from bidi import algorithm as bidialg
# import arabic_reshaper

# import networkx as nx

# import qur_func
from qur_root_analysis import *


if __name__ == '__main__':
	#########################
	path = '/Users/alikhan/Downloads/qur/qcm-analysis/'
	
	analysand = 'Root_ar'#'Lemma_ar'
	noi = u'اله'
	sz = 40
	
	draw_full_graph = False
	save_graphml = False

	#########################
	
	quran, qtoc = load_corpus_dataframe_from_csv(path = path)
	quran['position'] = quran['sura'].astype(str) + ':' + quran['aya'].astype(str)
	print(qtoc.info())
	features = []
	feat_header.extend(['sura', 'sura_name'])
	print(feat_header)

	for sur in qtoc['No.'].values:
		print(sur)
		qu = quran[quran.sura == sur].reset_index()
		bigrams, bigrams_pos = create_ngrams(qu, col=analysand, n=2, separate=True, sep='786')	
		edges_df, lenuniq = create_graph_edges_dataframe(bigrams, bigrams_pos, sep='786')
		root_counts = qu[analysand].value_counts().drop(0).to_dict()
		G, f = create_graph(edges_df, root_counts, node_of_interest=noi)
		title = str(*qtoc[qtoc['No.'] == sur].values.tolist())[1:-1] + '\nRoots: ' + str(len(G.nodes())) + \
				', Cooccurrences: '+ str(len(G.edges())) + ', Unique cooccurrences: ' + str(lenuniq)

		sura_name = qtoc[qtoc['No.'] == sur]['Name'].values[0]
		print(sura_name)
		f.extend([sur, sura_name])
		# print(f)
		features.append(f)

		if save_graphml:
			nx.write_graphml(G, path + 'graphml/'+ sura_name + '.graphml')

		if draw_full_graph:
			if f[0] != '':
				f[0] = bidialg.get_display(arabic_reshaper.reshape(f[0]))
			from itertools import zip_longest
			feat = '\n'.join(':  '.join(x) for x in zip_longest(feat_header, [str(i) for i in f], fillvalue=''))
			draw_graph(G, node_freq=root_counts, feat=feat, nodesize_multiplier=sz, 
						weight='count', title=title, filename=sura_name)

	features = pd.DataFrame(features, columns=feat_header)
	print(features.info())
	features.to_csv(path + 'data/sura_graph_features.csv')

	feat_header.remove('sura')
	feat_header.remove('sura_name')
	print(feat_header)


	# ##################################################################
	# ############ Subgraph for node of interest (root) ################
	# ##################################################################

	# # I = create_subgraph(G, method=method, node_of_interest=node_of_interest)
	# I, f = create_subgraph_from_edges_dataframe(edges_df, node_of_interest=node_of_interest)
	
	# if save_subgraphml:
	# 	# nx.write_graphml(I, path + 'graphml/' + node_of_interest + '.graphml')
	# 	nx.write_graphml(I, path + 'graphml/' + qur_func.arabic_to_buc(node_of_interest) + '.graphml')

	# if draw_subgraph:
	# 	draw_subgraph(I, node_of_interest=noi, nodesize_multiplier = sz, title= title)

	# if loop_to_subgraphs:
	# 	features = []
	# 	for noi in G.nodes().keys():
	# 		print(noi)
	# 		I, feat = create_subgraph_from_edges_dataframe(edges_df, node_of_interest=noi)
	# 		print(len(I.in_degree()))
	# 		print(len(I.out_degree()))
	# 		features.append(feat)

	# 	features = pd.DataFrame(features, columns=['root', 'graph_order', 'graph_size', 
	# 			'edges', 'unique_edges', 'avg_degree', 'strongly_connected', 
	# 			'weakly_connected', 'radius', 'diameter','center','density', 
	# 			'in_degree', 'out_degree','max_cooccurrence','freq'])
	# 	if (features.graph_size - features.edges).sum()==0:
	# 		features = features.drop('edges',1)
	# 	print(features.info())
	# 	features.to_csv(path + 'data/root_subgraph_features.csv')

