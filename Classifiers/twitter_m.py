#!/usr/bin/env python3

'''
Explanation...
'''

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
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

def read_data():
	"""Read csv files"""
	train = pd.read_csv('../Data/Final Data/tw_train.csv', converters={'posts':literal_eval})
	#val = pd.read_csv('../Data/Final Data/tw_validation.csv', converters={'posts':literal_eval})
	test = pd.read_csv('../Data/Final Data/tw_test.csv', converters={'posts':literal_eval})
	test_r = pd.read_csv('../Data/Final Data/r_test.csv', converters={'posts':literal_eval})
	
	return [train, test, test_r]


def preprocess(df):
	"""Preprocess the data frame: add label column, change some formatting"""
	df['label'] = df['gender'].apply(lambda x: 'M' if x == 'M' else 'NOT M')
	df['posts'] = df['posts'].apply(lambda x: [re.sub(r"#\w+", '<HASHTAG>', i) for i in x])
	df['posts'] = df['posts'].apply(lambda x: [re.sub("&amp;", '&', i) for i in x])
	df['posts'] = df['posts'].apply(lambda x: [re.sub("\xa0", ' ', i) for i in x])
	df['posts'] = df['posts'].apply(lambda x: [re.sub("tl;dr:", '<SUMMARY>', i, flags=re.I) for i in x])
	df['posts'] = df['posts'].apply(separate_emojis) # Add spaces around emojis
	df['no_rep'] = df['posts'].apply(lambda x: [re.sub(r"([aeiouy])\1\1+", r'\1', i) for i in x]) # Remove repeating letters

	return df


def baseline(data):
	"""Get baseline accuracy using a dummy classifier"""
	dum = DummyClassifier(strategy='most_frequent', random_state=1)
	dum.fit(data[0]['posts'], data[0]['label'])
	pred = dum.predict(data[1]['posts'])

	print("----------BASELINE-----------")
	print("Accuracy score: {}\n".format(accuracy_score(data[1]['label'], pred)))
	print("-----------------------------")


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
	"""Change most tokens to their corresponding pos tag, except for puctuation,
	pronouns and emojis. Use a custom tag for abbreviations and profanity"""
	doc = nlp(txt)

	new_doc = []
	for token in doc:
		if token.text in (item for sublist in [tags, punct_lst] for item in sublist):
			new_doc.append(token.text)
		elif token.text in abbrev_lst:
			new_doc.append("ABBREV")
		elif token.text in prof_lst:
			new_doc.append("PROFANITY")
		elif token.pos_ == "PRON":
			new_doc.append(token.text)
		elif token.text in emoji.UNICODE_EMOJI:
			new_doc.append(token.text)
		else:
			new_doc.append(token.pos_)
	
	return ' '.join(new_doc)


def features(df, nlp, prof_lst, abbrev_lst):
	"""Add several columns with possible features"""
	df['length'] = df['posts'].apply(len)
	df['comb_posts'] = df['posts'].apply(lambda x: ' '.join(x))
	df['no_rep'] = df['no_rep'].apply(lambda x: ' '.join(x))

	# Case
	df['upper_tot'] = df['comb_posts'].apply(lambda x: len([i for i in re.findall(r'(?:<[A-Z]+)>|([A-Z])', x) if i !='']))
	
	# Punctuation
	punct = '!_@}+\-~{;*./`?,:\])\\#[=\"&%\'(^|$—“”’—'
	df['newline_tot'] = df['comb_posts'].apply(lambda x: len(re.findall('\r\n', x)))
	df['newline_avg'] = df['newline_tot']/df['length']
	
	# POS tags
	tags = ['<LINK>', '<USER>', '<HASHTAG>', '<DESCRIPTION>', '<CATASK>', '<SUBREDDIT>',
	'<QUOTE>', '<SUMMARY>', '<BOLD-ITALIC>', '<BOLD>', '<ITALIC>', '<STRIKE>', '<SPOILER>',
	'<SUPER>', '<HEADING>']
	punct_lst = list(punct)
	df['tagged'] = df['no_rep'].apply(lambda x: tagger(x, nlp, tags, abbrev_lst, punct_lst, prof_lst))
	
	return df


def main():
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
		df = preprocess(df)
		# Features
		df = features(df, nlp, prof_lst, abbrev_lst)

	# Baseline
	print("----------TWITTER MALE----------")
	baseline(data)
	parameters = [{'clf__C': [0.1, 0.5, 1.0, 5.0, 10]}]
	labels = ['M', 'NOT M']

	# Classifier
	pipeline = Pipeline([
		('union', ColumnTransformer([
			('vecword', TfidfVectorizer(ngram_range = (1,2), analyzer='word', token_pattern=r'\S+'), 'tagged'),
			('newtot', MinMaxScaler(), ['newline_tot']),
			('newavg', MinMaxScaler(), ['newline_avg']),
			], remainder='drop')),
		('clf', LogisticRegression(max_iter=100000, C=5.0))])
	model = pipeline.fit(data[0], data[0]['label'])
	
	# Twitter
	pred = model.predict(data[1])
	cm = confusion_matrix(data[1]['label'], pred, labels=labels)

	# Reddit
	pred_r = model.predict(data[2])
	cm_r = confusion_matrix(data[2]['label'], pred_r, labels=labels)

	print("\n\n----------LOGISTIC REGRESSION-----------\n")
	print("----------TWITTER----------")
	print("Accuracy score: {}\n".format(accuracy_score(data[1]['label'], pred)))
	print("Classification report:")
	print(classification_report(data[1]['label'], pred))
	print("\nConfusion matrix: \n{}".format(pd.DataFrame(cm, index=labels, columns=labels)))
	print("---------------------------")

	print("\n----------REDDIT----------")
	print("Accuracy score: {}\n".format(accuracy_score(data[2]['label'], pred_r)))
	print("Classification report:")
	print(classification_report(data[2]['label'], pred_r))
	print("\nConfusion matrix: \n{}".format(pd.DataFrame(cm_r, index=labels, columns=labels)))
	print("--------------------------")
	print("--------------------------------------")


if __name__ == '__main__':
	main()
