#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
This script visualises root data in terms of frequence or occurrence, its relationships with
other roots in a specific sura or the whole Quran in terms of cooccurrence or paired frequency

DONE
1. Add sura and aya numbers with pair freq or count (aka cooccurrence) as attributes of edges
2. Add frequency or count of single root to node attribute
3. Output that graph with edge and node attrib to graphml for cytoscape exploration
4. Graph properties like 

"""

__author__ = "Ali Khan"
__license__ = "GPL"
__version__ = "0.0.8"
__maintainer__ = "Ali Khan"
__email__ = "khan.aliasad@gmail.com"
__status__ = "dev"


from qur_func import * 


if __name__ == '__main__':
	#########################
	path = '/Users/ali.khan/Documents/src/qcm-analysis/'
	
	chronological = False

	sur = None#[8]

	if sur is not None:
		sz = 80
	
	sz = 15

	analysand = 'Lemma_ar'#'Root_ar'
	node_of_interest = ''#u'نصح'#'عِلْم'#'فقه'#'نصح'#'دين'#'كون'#'فقه'#'دين'#'شرع'#u'حقق'#u'ذكر'#'*kr'#'kwn'#'qwl' #u'ذكر' u'ارض'. 'حرم' 'فعل' 'حرم' 'ﻏﻀﺐ'
	method = 'breadth'

	draw_full_graph = False
	save_graphml = False

	save_subgraphml = False 
	draw_root_subgraph = False 
	
	loop_to_subgraphs = True
	#########################
	
	quran, qtoc = load_corpus_dataframe_from_csv(path = path)
	quran = quran.merge(qtoc.drop('Place',1), left_on='sura', right_on='No.')
	print(quran.info())
	# print(qtoc)
	# assert((quran.Place_x == quran.Place_y).all())
	if chronological:
		quran['sura'] = quran['Chronology'].copy()
		assert((quran.sura == quran.Chronology).all())

	if sur is not None:
		quran = quran[quran.sura.isin(sur)].reset_index()

	quran['position'] = quran['sura'].astype(str) + ':' + quran['aya'].astype(str)

	# second = first.groupby(['aya','word']).agg('FORM_ar').apply(''.join).reset_index()
	bigrams, bigrams_pos = create_ngrams(quran, col=analysand, n=2, separate=True, sep='786')	

	edges_df, lenuniq = create_graph_edges_dataframe(bigrams, bigrams_pos, sep='786')

	root_counts = quran[analysand].value_counts().drop(0).to_dict()
	# del root_counts[0]
	#print(root_counts)
	#print(node_of_interest, root_counts[node_of_interest])
	# sns.distplot(quran[quran[analysand]!=0][analysand].value_counts(), bins=100)
	# plt.show()

	G = create_graph(edges_df, root_counts, node_of_interest)
	#print(edges_df)
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
		if chronological:
			num = 'Chronology'
		else:
			num = 'No.'
		title = str(*qtoc[qtoc[num] == sur].values.tolist())[1:-1] + \
									'\n{}s: '.format([analysand.split('_')[0] if '_' in analysand else analysand][0]) + \
									str(len(G.nodes())) + \
									', Cooccurrences: '+ str(len(G.edges())) + ', Unique cooccurrences: ' + str(lenuniq)
		# title = ''

	else:
		title = 'The Holy Quran [{}s: '.format([analysand.split('_')[0] if '_' in analysand else analysand][0]) + \
					str(len(G[0].nodes())) + \
					', Cooccurrences: ' + \
					str(len(G[0].edges())) + \
				 ', Unique cooccurrences: ' + str(lenuniq) + ']'
	
	if save_graphml:
		nx.write_graphml(G[0], path + 'graphml/test.graphml')

	if draw_full_graph:
		draw_graph(G[0], node_freq=root_counts, 
						nodesize_multiplier=sz, 
						weight='count', 
						title=title)

	##################################################################
	############ Subgraph for node of interest (root) ################
	##################################################################

	# I = create_subgraph(G, method=method, node_of_interest=node_of_interest)
	#I, f = create_subgraph_from_edges_dataframe(G[0], edges_df, node_of_interest=node_of_interest)
	
	if save_subgraphml:
		# nx.write_graphml(I, path + 'graphml/' + node_of_interest + '.graphml')
		nx.write_graphml(I, path + 'graphml/' + arabic_to_buc(node_of_interest) + '.graphml')

	if draw_root_subgraph:
		f[0] = bidialg.get_display(arabic_reshaper.reshape(f[0]))
		from itertools import zip_longest
		feat = '\n'.join(':  '.join(x) for x in zip_longest(feat_header, [str(i) for i in f], fillvalue=''))
		draw_subgraph(I, feat, node_of_interest=node_of_interest, 
						nodesize_multiplier = sz, 
						title= title, 
						root_or_lemma = [analysand.split('_')[0] if '_' in analysand else analysand][0])

	if loop_to_subgraphs:
		features = []
		#print(G[0],G[1])
		for noi in G[0].nodes().keys():
			print(noi)
			I, f = create_subgraph_from_edges_dataframe(G[0], edges_df, node_of_interest=noi)
			print(len(I.in_degree()))
			print(len(I.out_degree()))
			features.append(f)
			#f[0] = bidialg.get_display(arabic_reshaper.reshape(f[0]))
			from itertools import zip_longest
			feat = '\n'.join(':  '.join(x) for x in zip_longest(feat_header, [str(i) for i in f], fillvalue=''))
			draw_subgraph(I, feat, node_of_interest=noi, 
						nodesize_multiplier = sz, 
						title= title,
						root_or_lemma = [analysand.split('_')[0] if '_' in analysand else analysand][0])
			nx.write_graphml(I, path + 'graphml/{}.graphml'.format(arabic_to_buc(noi)))


		features = pd.DataFrame(features, columns=feat_header)
		# if (features.graph_size - features.edges).sum()==0:
		# 	features = features.drop('edges',1)
		print(features.info())
		features.to_csv(path + 'data/{}_subgraph_features.csv'.format(analysand))


	#############################################


	# check how many times a given degree occurs:
	degrees = [a + ' ' + str(G[0].degree[a]) for a in G[0].nodes]
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