#!/usr/bin/env python

"""
This script visualises box plots of lengths of aya and
lengths of suras in traditional and chronologica sequences

TODO
Check for errors: double check computations

Heat map of suras and roots as columns to see where roots appear again and again by sura
Scatter plot of roots in total number of suras they appear in vs average freq which can be 
computed as (total appearances / total suras appeared in)
"""

__author__ = "Ali Khan"
__license__ = "GPL"
__version__ = "0.0.1"
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

	quran, qtoc = load_data_from_csv(path = path + 'data/')
	# quran = quran.merge(qtoc.drop('Place',1), left_on='sura', right_on='No.')
	quran['number of roots'] = quran.Root_ar.astype(bool).astype(int)
	quran = quran.groupby(['sura','Root_ar']).agg(sum).reset_index()
	quran = quran[quran.Root_ar != 0]
	qpiv = quran.pivot(index='sura', columns='Root_ar', values='number of roots')
	if False:
		sns.heatmap(qpiv, xticklabels=30, yticklabels=4, cmap='Blues',center=50)#, linewidth=0.5)# cmap='YlGnBu')
		plt.show()
	root_counts = qpiv.sum()
	root_sura_counts = qpiv.fillna(0).astype(bool).sum()
	root_sura_freq = root_counts/root_sura_counts
	datf = pd.DataFrame({'root_counts':root_counts, 'root_sura_counts':root_sura_counts, 'root_sura_freq':root_sura_freq})
	print(datf)
	fig, ax = plt.subplots()
	plt.scatter(datf['root_sura_counts'], datf['root_sura_freq'], s=datf['root_counts'], alpha=0.2)# hue='root_counts'
	for n, xs, ys in zip(datf.index, datf['root_sura_counts'], datf['root_sura_freq']):
		ax.text(xs, ys, bidialg.get_display(arabic_reshaper.reshape(n)), alpha=0.6, size=10) ##bidialg.get_display(arabic_reshaper.reshape(n))
	plt.xlabel('Total number of Suras in which root appears')
	plt.ylabel('Total root appearances divided by Total number of Suras in which root appears')
	plt.show()

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
		ax = sns.boxplot(x=col, y='number of roots', hue='Place', 
					width=1, fliersize=2, linewidth=0.8, notch=False,
					data=groups)
		ax.set_xticklabels(ax.get_xticklabels(),rotation=90)
		plt.title('Distribution of density of aya (in number of roots) by sura')
		plt.show()
		plt.close()

		########### ROOTS VIOLIN
		groups = quran.groupby([col]).agg(sum).reset_index()
		print(groups)
		ax = sns.violinplot(y=groups['number of roots'], inner='stick', color='red')
		# ax.set_xticklabels(ax.get_xticklabels(),rotation=90)
		plt.title('Distribution of density of sura (in number of roots)')
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