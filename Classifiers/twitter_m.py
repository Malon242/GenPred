#!/usr/bin/env python3

'''
Explanation...
'''

#classifiers = [svm, logistic regression]

import re
import emoji
import pandas as pd
from ast import literal_eval
from string import punctuation

from sklearn import svm
from sklearn.pipeline import Pipeline
from sklearn.dummy import DummyClassifier
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

def read_data():
	"""Read csv files"""
	train = pd.read_csv('../Data/Final Data/tw_train.csv', converters={'tweets':literal_eval})
	val = pd.read_csv('../Data/Final Data/tw_validation.csv', converters={'tweets':literal_eval})
	test = pd.read_csv('../Data/Final Data/tw_test.csv', converters={'tweets':literal_eval})
	
	return [train, val, test]


def baseline(data):
	"""Gets baseline accuracy using a dummy classifier"""
	dum = DummyClassifier(strategy='most_frequent', random_state=1)
	dum.fit(data[0]['tweets'], data[0]['label'])
	pred = dum.predict(data[1]['tweets'])

	print("----------BASELINE-----------")
	print("Accuracy score: {}\n".format(accuracy_score(data[1]['label'], pred)))
	print("Classification report:")
	print(classification_report(data[1]['label'], pred))
	print("----------------------------")


def separate_emojis(lst):
	"""In every text, separate emojis so they count as both 
	a single character and a single word"""
	nw_lst = []

	for line in lst:
		for char in line:
			if char in emoji.UNICODE_EMOJI:
				line = line.replace(char, ' {} '.format(char))
		nw_lst.append(line)

	return nw_lst


def features(df):
	"""Add several columns with possible features"""
	df['length'] = df['tweets'].apply(len)
	df['comb_tweets'] = df['tweets'].apply(lambda x: ' '.join(x))

	# Length
	df['char_length_tot'] = df['comb_tweets'].apply(len)
	df['char_length_avg'] = df['comb_tweets'].apply(len)/df['length']
	df['length_tot'] = df['comb_tweets'].apply(lambda x: len(x.split()))
	df['length_avg'] = df['comb_tweets'].apply(lambda x: len(x.split()))/df['length']

	# Emojis
	df['emoji_tot'] = df['comb_tweets'].apply(lambda x: len([char for char in x if char in emoji.UNICODE_EMOJI]))
	df['emoji_avg'] = df['comb_tweets'].apply(lambda x: len([char for char in x if char in emoji.UNICODE_EMOJI]))/df['length']

	# Case
	df['upper_tot'] = df['comb_tweets'].apply(lambda x: len([i for i in re.findall(r'(?:<[A-Z]+)>|([A-Z])', x) if i !='']))
	df['upper_avg'] = df['comb_tweets'].apply(lambda x: len([i for i in re.findall(r'(?:<[A-Z]+)>|([A-Z])', x) if i !='']))/df['length']
	df['lower_tot'] = df['comb_tweets'].apply(lambda x: len(re.findall(r'[a-z]', x)))
	df['lower_avg'] = df['comb_tweets'].apply(lambda x: len(re.findall(r'[a-z]', x)))/df['length']

	# Punctuation
	punct = '!_@}+\-~{;*./`?,:\])\\#[=\"&%\'(^|$—“”’—'
	df['punct_tot'] = df['comb_tweets'].apply(lambda x: len(re.findall('[{}]'.format(punct),x)))
	df['punct_avg'] = df['comb_tweets'].apply(lambda x: len(re.findall('[{}]'.format(punct),x)))/df['length']

	# Repetition
	df['rep_tot'] = df['comb_tweets'].apply(lambda x: len(re.findall(r"([aeiou!?.])\1{2,}",x, flags=re.I)))
	df['rep_avg'] = df['comb_tweets'].apply(lambda x: len(re.findall(r"([aeiou!?.])\1{2,}",x, flags=re.I)))/df['length']
	
	return df


def main():
	pd.set_option('display.max_colwidth', -1)
	data = read_data()

	for df in data:
		# Preprocessing
		df['label'] = df['gender'].apply(lambda x: 1 if x == 'M' else 0)
		df['tweets'] = df['tweets'].apply(lambda x: [re.sub(r"#\w+", '<HASHTAG>', i) for i in x])
		df['tweets'] = df['tweets'].apply(lambda x: [re.sub("&amp;", '&', i) for i in x])
		df['tweets'] = df['tweets'].apply(lambda x: [re.sub("tl;dr:", '<SUMMARY>', i, flags=re.I) for i in x])
		df['tweets'] = df['tweets'].apply(separate_emojis)

		df = features(df)



	#baseline(data)
	print(data[1][['rep_wrd_tot', 'rep_wrd_avg']])

if __name__ == '__main__':
	main()
