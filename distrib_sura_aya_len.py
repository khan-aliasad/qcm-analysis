#!/usr/bin/env python

"""
This script visualises box plots of lengths of aya and
lengths of suras in traditional and chronologica sequences

TODO
Check for errors: double check computations

Heat map of suras and roots (or lemmas) as columns to see where roots (or lemmas) appear again and again by sura
Scatter plot of roots (or lemmas) in total number of suras they appear in vs average freq which can be 
computed as (total appearances / total suras appeared in)
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
sns.set(style='white')

from bidi import algorithm as bidialg
import arabic_reshaper



#######################


def load_data_from_csv(path = '/Users/ali.khan/Documents/qcm-analysis/'):
	quran = pd.read_csv(path + 'quran-morphology-final.csv', sep=",", header=0)#, index_col='Index')
	qtoc = pd.read_csv(path + 'toc.csv')
	qtoc['Name Arabic'] = qtoc['Name Arabic'].apply(lambda x: bidialg.get_display(arabic_reshaper.reshape(x)))
	quran = quran.fillna(0)
	print(quran.head())
	print(quran.info())
	return quran, qtoc


###########################


if __name__ == '__main__':

	path = '/Users/ali.khan/Documents/src/qcm-analysis/'

	analysand = 'Lemma_ar'#'Root_ar'

	quran, qtoc = load_data_from_csv(path = path + 'resources/')
	quran = quran.merge(qtoc.drop('Place',1), left_on='sura', right_on='No.')
	quran['number of {}s'.format(analysand)] = quran[analysand].astype(bool).astype(int)
	
	# qrn = quran.groupby(['Chronology',analysand]).agg(sum).reset_index()
	qrn = quran.groupby(['sura',analysand]).agg(sum).reset_index()
	qrn = qrn[qrn[analysand] != 0]
	# qpiv = qrn.pivot(index='Chronology', columns=analysand, values='number of {}s'.format(analysand))
	qpiv = qrn.pivot(index='sura', columns=analysand, values='number of {}s'.format(analysand))
	qpiv.to_csv(path + 'data/heatmap_{}_features.csv'.format(analysand))

	if True:
		sns.heatmap(qpiv.sort_index(), xticklabels=30, yticklabels=4, cmap='Blues',center=50)#, linewidth=0.5)# cmap='YlGnBu')
		plt.show(); plt.close()

	analysand_counts = qpiv.sum()
	analysand_sura_counts = qpiv.fillna(0).astype(bool).sum()
	analysand_sura_freq = analysand_counts/analysand_sura_counts
	datf = pd.DataFrame({'analysand_counts':analysand_counts, 'analysand_sura_counts':analysand_sura_counts, 'analysand_sura_freq':analysand_sura_freq})
	print(datf)
	fig, ax = plt.subplots()
	plt.scatter(datf['analysand_sura_counts'], datf['analysand_sura_freq'], s=datf['analysand_counts'], alpha=0.2)# hue='analysand_counts'
	for n, xs, ys in zip(datf.index, datf['analysand_sura_counts'], datf['analysand_sura_freq']):
		ax.text(xs, ys, bidialg.get_display(arabic_reshaper.reshape(n)), alpha=0.6, size=10) ##bidialg.get_display(arabic_reshaper.reshape(n))
	plt.xlabel('Total number of Suras in which analysand appears')
	plt.ylabel('Total analysand appearances divided by Total number of Suras in which analysand appears')
	plt.show(); plt.close()

	if True:
		ax = sns.kdeplot(datf['analysand_sura_counts'], datf['analysand_sura_freq'], shade=True)
		plt.show(); plt.close()

	if False:
		# print(quran.sura.max())
		sequence = 'chronological' #'chronological'
		if sequence is 'traditional':
			col = 'sura'
		elif sequence is 'chronological':
			col = 'Chronology'

		########### ROOTS BOX
		groups = quran.groupby([col,'aya']).agg(sum).reset_index()
		if sequence is 'chronological':
			groups = groups.merge(qtoc, left_on='Chronology', right_on='Chronology')
		else:
			groups = groups.merge(qtoc, left_on='sura', right_on='No.')
		print(groups)
		ax = sns.boxplot(x=col, y='number of {}s'.format(analysand), hue='Place', 
					width=1, fliersize=2, linewidth=0.8, notch=False,
					data=groups)
		ax.set_xticklabels(ax.get_xticklabels(),rotation=90)
		plt.title('Distribution of density of aya (in number of analysands) by sura')
		plt.show()
		plt.close()

		########### ROOTS VIOLIN
		groups = quran.groupby([col]).agg(sum).reset_index()
		print(groups)
		ax = sns.violinplot(y=groups['number of {}s'.format(analysand)], inner='stick', color='red')
		# ax.set_xticklabels(ax.get_xticklabels(),rotation=90)
		plt.title('Distribution of density of sura (in number of analysands)')
		plt.show()
		plt.close()

		########### AYA VIOLIN
		groups = quran.groupby([col]).agg(max).reset_index().rename(columns={'aya':'number of aya'})
		print(groups)
		print(groups['number of aya'].describe())
		# if False:
		ax = sns.violinplot(y=groups['number of aya'], inner='stick', color='green')
		# ax.set_xticklabels(ax.get_xticklabels(),rotation=90)
		plt.title('Distribution of lengths of sura (in number of aya)')
		plt.show()
		plt.close()

		########### WORDS BOX
		groups = quran.groupby([col,'aya']).agg(max).reset_index().rename(columns={'word':'number of words'})
		print(groups)
		print(groups['number of words'].describe())
		# if False:
		ax = sns.boxplot(x=col, y='number of words', hue='Place', 
					width=1, fliersize=2, linewidth=0.8, notch=False,
					data=groups)
		ax.set_xticklabels(ax.get_xticklabels(),rotation=90)
		plt.title('Distribution of density of aya (in number of words) by sura')
		plt.show()
		plt.close()

		########### WORDS VIOLIN ???????????????????? CHECK
		groups = quran.groupby([col]).agg(sum).reset_index().rename(columns={'word':'number of words'})
		print(groups[['Chronology','number of words']])
		ax = sns.violinplot(y=groups['number of words'], inner='stick', color='yellow')
		# ax.set_xticklabels(ax.get_xticklabels(),rotation=90)
		plt.title('Distribution of lengths of sura (in number of words)')
		plt.show()
		plt.close()

		# d = root_in_edges.reset_index() \
		# 	.drop('pos',1) \
		# 	.merge(qtoc[['No.','Place','Chronology']], 
		# 		left_on='sura',
		# 		right_on='No.', 
		# 		how='left') \
		# 	.drop('No.',1)
		# print(d.head())
		# print(len(d[d.Place == 'Meccan']), np.sort(d[d.Place == 'Meccan'].sura.unique()))
		# print(len(d[d.Place == 'Medinan']), np.sort(d[d.Place == 'Medinan'].sura.unique()))
		# 