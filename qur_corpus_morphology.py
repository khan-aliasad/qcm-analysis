import numpy as np
import pandas as pd

import qur_func

def load_data(path='/Users/alikhan/Downloads/qur/'):
	qdforiginal = data = pd.read_csv(path + 'qur-corpus-morphology-0.4.txt', sep="\t", header=0)
	tmp1 = qdforiginal.LOCATION.str.extract(r'(?P<sura>\d*):(?P<aya>\d*):(?P<word>\d*):(?P<w_seg>\d*)')
	tmp2 = qdforiginal.FEATURES.str.extract(r'ROOT:(?P<Root>[^|]*)')
	tmp3 = qdforiginal.FEATURES.str.extract(r'LEM:(?P<Lemma>[^|]*)')
	df_qruan = pd.concat([tmp1, qdforiginal, tmp2,tmp3], axis=1)
	df_qruan.sura = df_qruan.sura.astype('int')
	df_qruan.aya = df_qruan.aya.astype('int')
	df_qruan.word = df_qruan.word.astype('int')
	df_qruan.w_seg = df_qruan.w_seg.astype('int')
	print(df_qruan.info())

	# qtoc = pd.read_clipboard()
	# print(qtoc.head())
	# qtoc.to_csv('/Users/alikhan/Downloads/toc.csv', index=False)
	qtoc = pd.read_csv(path + 'toc.csv')
	print(qtoc.info())

	quran = df_qruan.merge(qtoc.loc[:,['No.', 'Place']], how='left', left_on='sura', right_on='No.')

	quran.drop(columns=['LOCATION','No.'], inplace=True)

	quran['FORM_ar'] = quran.apply(lambda x: qur_func.buck_to_arabic(x.FORM), axis=1)
	quran['Root_ar'] = quran.apply(lambda x: qur_func.buck_to_arabic(x.Root), axis=1)
	##TEMPORARY FIX WHERE 78 ROWS HAVE LEMMA ENDING IN 2 e.g. EaAd2
	quran.Lemma = quran.Lemma.str.replace('\d+', '') #####
	quran['Lemma_ar'] = quran.apply(lambda x: qur_func.buck_to_arabic(x.Lemma), axis=1)
	quran['POW'] = quran['FEATURES'].apply(lambda x: x.split('|')[0])

	quran.to_csv('/Users/alikhan/Downloads/qur/quran-morphology-final.csv', index=False)

	return quran 


if __name__ == '__main__':
	print(qur_func.buck_to_arabic('EalaY'))
	# print(qur_func.arabic_to_buc('اﻟﺤﻤﺪ ﻟﻠﻪ'))

	quran = load_data(path='/Users/alikhan/Downloads/qur/')

	k = set(quran[quran.Place == 'Meccan'].Root.unique().tolist())
	d = set(quran[quran.Place == 'Medinan'].Root.unique().tolist())

	makki_words = k-d; print(len(makki_words))
	madani_words = d - k; print(len(madani_words))
	# print(makki_words)
	# print(madani_words)

	both = k & d
	print(len(both))

	print(qur_func.sura_words(quran, [111],'L'))
	print(qur_func.unique_sura_words(quran, [113],'R'))
