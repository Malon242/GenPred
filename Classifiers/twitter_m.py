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

import spacy
from spacy.symbols import ORTH

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


def add_cases(nlp):
	"""Add special cases to Spacy, making sure tags stay in the format <TAG>"""
	nlp.tokenizer.add_special_case("<LINK>", [{"ORTH":"<LINK>"}])
	nlp.tokenizer.add_special_case("<USER>", [{"ORTH":"<USER>"}])
	nlp.tokenizer.add_special_case("<HASHTAG>", [{"ORTH":"<HASHTAG>"}])
	nlp.tokenizer.add_special_case("<DESCRIPTION>", [{"ORTH":"<DESCRIPTION>"}])
	nlp.tokenizer.add_special_case("<CATASK>", [{"ORTH":"<CATASK>"}])
	nlp.tokenizer.add_special_case("<SUBREDDIT>", [{"ORTH":"<SUBREDDIT>"}])
	nlp.tokenizer.add_special_case("<QUOTE>", [{"ORTH":"<QUOTE>"}])
	nlp.tokenizer.add_special_case("<SUMMARY>", [{"ORTH":"<SUMMARY>"}])
	nlp.tokenizer.add_special_case("<BOLD-ITALIC>", [{"ORTH":"<BOLD-ITALIC>"}])
	nlp.tokenizer.add_special_case("<BOLD>", [{"ORTH":"<BOLD>"}])
	nlp.tokenizer.add_special_case("<ITALIC>", [{"ORTH":"<ITALIC>"}])
	nlp.tokenizer.add_special_case("<STRIKE>", [{"ORTH":"<STRIKE>"}])
	nlp.tokenizer.add_special_case("<SPOILER>", [{"ORTH":"<SPOILER>"}])
	nlp.tokenizer.add_special_case("<SUPER>", [{"ORTH":"<SUPER>"}])
	nlp.tokenizer.add_special_case("<HEADING>", [{"ORTH":"<HEADING>"}])

	return nlp


def separate_emojis(lst):
	"""Add spaces around emojis"""
	nw_lst = []

	for line in lst:
		for char in line:
			if char in emoji.UNICODE_EMOJI:
				line = line.replace(char, ' {} '.format(char))
		nw_lst.append(line)

	return nw_lst


def tagger(txt, nlp, tags, abbrev_lst, punct_lst, prof_lst):
	"""Explanation"""
	doc = nlp(txt)

	new_doc = []
	for token in doc:
		if token.text in (item for sublist in [tags, punct_lst] for item in sublist):
			new_doc.append(token.text)
		elif token.text in abbrev_lst:
			new_doc.append("ABBREV")
		elif token.pos_ == "PRON" or token.pos_ == "SPACE":
			new_doc.append(token.text)
		elif token.text in emoji.UNICODE_EMOJI:
			new_doc.append(token.text)
		else:
			new_doc.append(token.pos_)

		# Profanity as tag??

	
	return ' '.join(new_doc)


def features(df, nlp, prof_lst, abbrev_lst):
	"""Add several columns with possible features"""
	df['length'] = df['tweets'].apply(len)
	df['comb_tweets'] = df['tweets'].apply(lambda x: ' '.join(x))
	df['no_rep'] = df['no_rep'].apply(lambda x: ' '.join(x))

	# Length
	df['char_length_tot'] = df['comb_tweets'].apply(len)
	df['char_length_avg'] = df['char_length_tot']/df['length']
	df['length_tot'] = df['comb_tweets'].apply(lambda x: len(x.split()))
	df['length_avg'] = df['length_tot']/df['length']

	# Emojis
	df['emoji_tot'] = df['comb_tweets'].apply(lambda x: len([char for char in x if char in emoji.UNICODE_EMOJI]))
	df['emoji_avg'] = df['emoji_tot']/df['length']

	# Case
	df['upper_tot'] = df['comb_tweets'].apply(lambda x: len([i for i in re.findall(r'(?:<[A-Z]+)>|([A-Z])', x) if i !='']))
	df['upper_avg'] = df['upper_tot']/df['length']
	df['lower_tot'] = df['comb_tweets'].apply(lambda x: len(re.findall(r'[a-z]', x)))
	df['lower_avg'] = df['lower_tot']/df['length']

	# Punctuation
	punct = '!_@}+\-~{;*./`?,:\])\\#[=\"&%\'(^|$—“”’—'
	df['punct_tot'] = df['comb_tweets'].apply(lambda x: len(re.findall('[{}]'.format(punct),x)))
	df['punct_avg'] = df['punct_tot']/df['length']
	df['newline_tot'] = df['comb_tweets'].apply(lambda x: len(re.findall('\r\n', x)))
	df['newline_avg'] = df['newline_tot']/df['length']

	# Repetition
	df['rep_tot'] = df['comb_tweets'].apply(lambda x: len(re.findall(r"([aeiouy!?.])\1{2,}",x, flags=re.I)))
	df['rep_avg'] = df['rep_tot']/df['length']

	# Profanity
	df['prof_tot'] = df['no_rep'].apply(lambda x: len(re.findall(r"\b({})\b".format('|'.join(prof_lst)), x, flags=re.I)))
	df['prof_avg'] = df['prof_tot']/df['length']

	# POS tags
	tags = ['<LINK>', '<USER>', '<HASHTAG>', '<DESCRIPTION>', '<CATASK>', '<SUBREDDIT>',
	'<QUOTE>', '<SUMMARY>', '<BOLD-ITALIC>', '<BOLD>', '<ITALIC>', '<STRIKE>', '<SPOILER>',
	'<SUPER>', '<HEADING>']
	punct_lst = list(punct)
	df['tagger'] = df['no_rep'].apply(lambda x: tagger(x, nlp, tags, abbrev_lst, punct_lst, prof_lst))
	
	return df


def main():
	pd.set_option('display.max_colwidth', 3000)
	data = read_data()

	nlp = spacy.load("en_core_web_sm", disable=['parser', 'ner', 'textcat'])
	nlp = add_cases(nlp)

	with open('profanity.txt', 'r') as f:
		prof_lst = f.read().splitlines()

	with open('abbreviations.txt', 'r') as f:
		abbrev_dict = literal_eval(f.read())
	abbrev_lst = [*abbrev_dict]

	for df in data:
		# Preprocessing
		df['label'] = df['gender'].apply(lambda x: 1 if x == 'M' else 0)
		df['tweets'] = df['tweets'].apply(lambda x: [re.sub(r"#\w+", '<HASHTAG>', i) for i in x])
		df['tweets'] = df['tweets'].apply(lambda x: [re.sub("&amp;", '&', i) for i in x])
		df['tweets'] = df['tweets'].apply(lambda x: [re.sub("\xa0", ' ', i) for i in x])
		df['tweets'] = df['tweets'].apply(lambda x: [re.sub("tl;dr:", '<SUMMARY>', i, flags=re.I) for i in x])
		df['tweets'] = df['tweets'].apply(separate_emojis) # Add spaces around emojis
		df['no_rep'] = df['tweets'].apply(lambda x: [re.sub(r"([aeiouy])\1\1+", r'\1', i) for i in x]) # Remove repeating letters

		# Features
		df = features(df, nlp, prof_lst, abbrev_lst)


	#baseline(data)
	print(data[1]['tagger'])


if __name__ == '__main__':
	main()
