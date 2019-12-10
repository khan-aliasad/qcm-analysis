#!/usr/bin/env python

"""
This script takes 5 functions from AbdulBaqi Muhammad (Sharaf) 
http://abdulbaqi.io/2018/12/04/makki-madani-word-count/ 
and his work as reported at http://textminingthequran.com/

The remainder of the script has utility functions written for other 
analyses scripts as included in the repo
"""

__author__ = "Ali Khan"
__license__ = "GPL"
__version__ = "0.0.8"
__maintainer__ = "Ali Khan"
__email__ = "khan.aliasad@gmail.com"
__status__ = "dev"


import pandas as pd
import numpy as np
from bidi import algorithm as bidialg
import arabic_reshaper
import networkx as nx

import matplotlib.pyplot as plt
plt.rcParams['axes.facecolor'] = 'black'



feat_header = ['root', 'graph_order', 'graph_size', 
                'unique_edges', 'avg_degree', 'strongly_connected', 
                'weakly_connected', 'radius', 'diameter','center','density', 
                'root_degree', 'root_in_degree', 'root_out_degree','max_cooccurrence','root_freq']


###############################################################
# This code block includes src from AbdulBaqi Muhammad (Sharaf) 
# http://abdulbaqi.io/2018/12/04/makki-madani-word-count/ 
# and his work as reported at http://textminingthequran.com/
###############################################################

abjad = {u"\u0627":'A',
u"\u0628":'b', u"\u062A":'t', u"\u062B":'v', u"\u062C":'j',
u"\u062D":'H', u"\u062E":'x', u"\u062F":'d', u"\u0630":'*', u"\u0631":'r',
u"\u0632":'z', u"\u0633":'s', u"\u0634":'$', u"\u0635":'S', u"\u0636":'D',
u"\u0637":'T', u"\u0638":'Z', u"\u0639":'E', u"\u063A":'g', u"\u0641":'f',
u"\u0642":'q', u"\u0643":'k', u"\u0644":'l', u"\u0645":'m', u"\u0646":'n',
u"\u0647":'h', u"\u0648":'w', u"\u0649":'Y', u"\u064A":'y'}


abjad[' ']=' '
abjad[u"\u0621"] = '\''
abjad[u"\u0623"] = '>'
abjad[u"\u0625"] = '<'
abjad[u"\u0624"] = '&'
abjad[u"\u0626"] = '}'
#abjad[u"\u0655"] = '\'' # Hamza below
abjad[u"\u0622"] = '|'
abjad[u"\u064E"] = 'a'
abjad[u"\u064F"] = 'u'
abjad[u"\u0650"] = 'i'
abjad[u"\u0651"] = '~'
abjad[u"\u0652"] = 'o'
abjad[u"\u064B"] = 'F'
abjad[u"\u064C"] = 'N'
abjad[u"\u064D"] = 'K'
abjad[u"\u0640"] = '_'
abjad[u"\u0670"] = '`'
abjad[u"\u0629"] = 'p'
abjad[u"\u0653"] = '^'
abjad[u"\u0654"] = '#'
abjad[u"\u0671"] = '{'
abjad[u"\u06DC"] = ':'
abjad[u"\u06DF"] = '@'
abjad[u"\u0653"] = '^'
abjad[u"\u06E0"] = '"'
abjad[u"\u06E2"] = '['
abjad[u"\u06E3"] = ';'
abjad[u"\u06E5"] = ','
abjad[u"\u06E6"] = '.'
abjad[u"\u06E8"] = '!'
abjad[u"\u06EA"] = '-'
abjad[u"\u06EB"] = '+'
abjad[u"\u06EC"] = '%'
abjad[u"\u06ED"] = ']'


# Create the reverse
alphabet = {}
for key in abjad:
    alphabet[abjad[key]] = key

def arabic_to_buc(ara):
    return ''.join(map(lambda x:abjad[x], list(ara)))


def buck_to_arabic(buc):
    if type(buc) is not str:
        return ''
    else:
        return ''.join(map(lambda x:alphabet[x], list(buc)))


def contains_num(x):
    import re 
    if type(x) is not str:
        return False
    else:
        found = re.search('\d', x)
        if found is None:
            return False
        else:
            return True


# function to return words given a list of sura
def sura_words(quran, s_list, kind='W'):
    if (kind=='R'):
        result = quran[quran.sura.isin(s_list)].Root.dropna().unique().tolist()
    elif (kind=='L'):
        result = quran[quran.sura.isin(s_list)].Lemma.dropna().unique().tolist()
    else:
        result = quran[quran.sura.isin(s_list)].FORM.unique().tolist()
    return [buck_to_arabic(x) for x in result]


# function to return words given a list of sura
def unique_sura_words(quran, s_list, kind='W'):
    if (kind=='R'):
        first = quran[quran.sura.isin(s_list)].Root.dropna().unique().tolist()
        second = quran[~quran.sura.isin(s_list)].Root.dropna().unique().tolist()
        result = list(set(first)-set(second))
    elif (kind=='L'):
        first = quran[quran.sura.isin(s_list)].Lemma.dropna().unique().tolist()
        second = quran[~quran.sura.isin(s_list)].Lemma.dropna().unique().tolist()
        result = list(set(first)-set(second))
    else:
        first = quran[quran.sura.isin(s_list)].FORM.dropna().unique().tolist()
        second = quran[~quran.sura.isin(s_list)].FORM.dropna().unique().tolist()
        result = list(set(first)-set(second))
    return [buck_to_arabic(x) for x in result]

###############################################################
####################### END Code Block ########################
###############################################################


def load_corpus_dataframe_from_csv(path = '/Users/alikhan/Downloads/qur/'):
    quran = pd.read_csv(path + 'data/quran-morphology-final.csv', sep=",", header=0)#, index_col='Index')
    qtoc = pd.read_csv(path + 'data/toc.csv')
    qtoc['Name Arabic'] = qtoc['Name Arabic'].apply(lambda x: bidialg.get_display(arabic_reshaper.reshape(x)))
    quran = quran.fillna(0)
    print(quran.head())
    print(quran.info())

    # try:
    #   test= quran[['Root','Lemma','FORM','FORM_ar']].loc[quran.Lemma.apply(lambda x: qur_func.contains_num(x)) == True]
    #   test.Lemma = test.Lemma.str.replace('\d+', '')
    #   test['Lemma_ar'] = test.apply(lambda x: qur_func.buck_to_arabic(x.Lemma), axis=1)

    #   print(test)
    # except ValueError:
    #   print('')

    return quran, qtoc


def create_ngrams(quran, col='Root_ar', n=2, separate=True, sep='786'):
    # mask = quran.aya.diff() == 1
    # if separate: # verse wise pairing, no roots paired across two different adjacent verses, i.e. last root of a verse 
    # # with the first root of the next verse is not paired, if separate = True
    #   a = np.insert(quran[col].values, mask.index[mask == True], sep)
    # else:
    #   a = quran[col].values
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


def create_graph_features(G, node_of_interest):
    N, K = G.order(), G.size()
    # print('Subgraph order (# nodes)', N)
    # print('Subraph size (# edges)', K)
    # print('Avg deg ',float(K)/N)
    # print('Strongly connected: ',nx.number_strongly_connected_components(G))
    # print('Weakly connected: ', nx.number_weakly_connected_components(G))
    # print("radius: %d" % nx.radius(G.to_undirected()))
    # print("diameter: %d" % nx.diameter(G.to_undirected()))
    # print("eccentricity: %s" % nx.eccentricity(G.to_undirected()))
    # print("center: %s" % nx.center(G.to_undirected()))
    # print("periphery: %s" % nx.periphery(G.to_undirected()))
    # print("density: %s" % nx.density(G))
    scipy_mat = nx.convert_matrix.to_scipy_sparse_matrix(G)
    
    fre = ''
    if node_of_interest != '':
        try:
            fre = G.nodes[node_of_interest]['freq']
        except:
            print('Graph noi does not exist ...')
            node_of_interest = ''

    feat = [node_of_interest, N, K, scipy_mat.nnz, float(K)/N, nx.number_strongly_connected_components(G), nx.number_weakly_connected_components(G)]
    try:
        rad = nx.radius(G.to_undirected())
    except:
        print('Graph radius issues ....')
        rad = ''
    try:
        dia = nx.diameter(G.to_undirected())
    except:
        print('Graph diameter issues ....')
        dia = ''
    try:
        cent = nx.center(G.to_undirected())
        cent = [bidialg.get_display(arabic_reshaper.reshape(a)) for a in cent]
    except:
        print('Graph center issues ....')
        cent = ''
    try:
        dens = nx.density(G)
    except:
        print('Graph density issues ....')
        dens = ''

    feat.extend([rad, dia, cent, dens])
    if node_of_interest != '':
        feat.extend([G.degree(node_of_interest), G.in_degree(node_of_interest), G.out_degree(node_of_interest)])
    else:
        feat.extend(['','',''])

    feat.extend([scipy_mat.max(), fre])
    return feat 


def create_graph(edges_df, node_freq, node_of_interest='', u='one', v='two'):
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
    #   print(G[u][v])

    N, K = G.order(), G.size()
    print('Graph order (# nodes)', N)
    print('Graph size (# edges)', K)
    print('Avg deg ',float(K)/N)
    if node_of_interest != '':
        print('Deg (noi)',G.degree(node_of_interest))
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

    # print("radius: %d" % nx.radius(G.to_undirected()))
    # print("diameter: %d" % nx.diameter(G.to_undirected()))
    # print("eccentricity: %s" % nx.eccentricity(G.to_undirected()))
    # print("center: %s" % nx.center(G.to_undirected()))
    # print("periphery: %s" % nx.periphery(G.to_undirected()))
    print("density: %s" % nx.density(G))

    feat = create_graph_features(G, node_of_interest)
    return G, feat


def draw_graph(G, node_freq, feat=None, nodesize_multiplier=40, weight='count', title='', filename='graph'):
    # shpsize = 30+10*np.log2([root_counts[x] for x in filter(lambda x: shapemap[x]==shp, G.nodes().keys())])
    # shpsize = [node_freq[x]*nodesize_multiplier*30 for x in G.nodes().keys()]

    pos = nx.spring_layout(G, k=0.7, iterations=10, weight='count')
    # pos = nx.random_layout(G) #, weight='AMOUNT')
    # pos = nx.circular_layout(G) #, weight='AMOUNT')
    # pos = nx.shell_layout(G) #, weight='AMOUNT')
    # pos = nx.spectral_layout(G) #, weight='AMOUNT')
    maxkey = max(node_freq, key=node_freq.get) 
    # pos = nx.fruchterman_reingold_layout(G, pos= {str(maxkey): (0.1,0.1)}, fixed=[maxkey], k=0.7, iterations=10, weight=weight)
    # pos = nx.fruchterman_reingold_layout(G, k=0.5, iterations=10, weight=weight)
    
    # build a rectangle in axes coords
    left, width = -1, .5
    bottom, height = 1.5, .5
    right = left - width
    top = bottom - height

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

    if feat is not None:
        plt.text(left, top, feat, fontsize=6, color='white',
                horizontalalignment='left',
                verticalalignment='top', 
                # xy=left, xytext=top,
                # xycoords='axes fraction', textcoords='offset points',
                bbox=dict(fill=False, facecolor='k',edgecolor='white', linewidth=0.5))

    # plt.savefig('simple_path.png') # save as png
    plt.savefig('./fig/'+filename+'.png', dpi=300, facecolor='w', edgecolor='w',
    orientation='landscape', papertype=None, format='png', transparent=False, bbox_inches=None, pad_inches=0.1,
    frameon=None)
    # plt.show() # display
    plt.close()


def create_subgraph_from_edges_dataframe(G, edges_df, node_of_interest='', u='one', v='two', weight='count'):
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
    #   print(I[u][v])
    # print(G.nodes(data=True))
    nx.set_node_attributes(I, dict([(n,d['freq']) for n,d in G.nodes(data=True)]), 'freq')

    feat = create_graph_features(I, node_of_interest)
    return I, feat


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
    #   print(list(nx.bfs_labeled_edges(I,node_of_interest)))
    # else:
    #   print(list(nx.dfs_labeled_edges(I,node_of_interest)))

    # import sys
    # sys.exit()
    return I


def draw_subgraph(I, feat, node_of_interest = '', node_freq=None, nodesize_multiplier=5, title= ''):
    # df = pd.DataFrame(index=I.nodes(), columns=I.nodes())
    # for row, datum in nx.shortest_path_length(I):
    #     for col, dist in datum.items():
    #         df.loc[row,col] = dist

    # df = df.fillna(df.max().max())

    # dfspos = nx.kamada_kawai_layout(I, dist=df.to_dict())

    dfspos = nx.spring_layout(I, k=0.7, iterations=10, weight='count')
    # print(dfspos)
    # shpsize = [node_freq[x]*nodesize_multiplier for x in I.nodes().keys()]

    # # build a rectangle in axes coords
    left, width = -1, .5
    bottom, height = 1.5, .5
    right = left - width
    top = bottom - height

    # fig = plt.figure()
    fig = plt.figure(figsize=(15.10, 20.80))  
    # ax = fig.add_axes([0,0,1,1])

    # # axes coordinates are 0,0 is bottom left and 1,1 is upper right
    # import matplotlib.patches as patches
    # p = patches.Rectangle(
    #     (left, bottom), width, height,
    #     fill=False, transform=ax.transAxes, clip_on=False
    #     )

    # ax.add_patch(p)

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
    plt.text(left, top, feat, fontsize=6, color='white',
                horizontalalignment='left',
                verticalalignment='top', 
                # xy=left, xytext=top,
                # xycoords='axes fraction', textcoords='offset points',
                bbox=dict(fill=False, facecolor='k',edgecolor='white', linewidth=0.5))
    
    plt.title('Root: ' + bidialg.get_display(arabic_reshaper.reshape(node_of_interest))+ 
                ' [Freq: ' + str(I.nodes[node_of_interest]['freq']) +
                ', Roots: ' + str(len(I.nodes())) +
                ', Cooccurrences: '+ str(len(I.edges())) + ']\nIn ' +
                title)
    # ax.set_axis_off()
    plt.savefig('./fig/'+ arabic_to_buc(node_of_interest) + '.png', dpi=300, facecolor='w', edgecolor='w',
    orientation='landscape', papertype=None, format='png', transparent=False, bbox_inches=None, pad_inches=0.1,
    frameon=None)
    # plt.show()
    plt.close()
    # from networkx.drawing.nx_agraph import write_dot
    # write_dot(I, 'test.dot')
