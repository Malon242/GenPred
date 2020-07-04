#!/usr/bin/env python3

'''
This program is a binary classifier used to predict whether a Reddit user
is 'Non-binary' or not. It uses a Logistics Regression classifier with 1,2 n-grams,
and the total length in characters of a text written by a user.
'''

import re
import emoji
import pandas as pd
from ast import literal_eval
from string import punctuation

import spacy
from spacy.symbols import ORTH

from sklearn.pipeline import Pipeline
from sklearn.dummy import DummyClassifier
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

def read_data():
	"""Read csv files"""
	train = pd.read_csv('../Data/Final Data/r_train.csv', converters={'posts':literal_eval})
	val = pd.read_csv('../Data/Final Data/r_validation.csv', converters={'posts':literal_eval})
	test = pd.read_csv('../Data/Final Data/r_test.csv', converters={'posts':literal_eval})
	test_tw = pd.read_csv('../Data/Final Data/tw_test.csv', converters={'posts':literal_eval})
	
	return [train, test, test_tw, val]


def preprocess(df):
	"""Preprocess the data frame: add label column, change some formatting"""
	df['label'] = df['gender'].apply(lambda x: 'NB' if x == 'NB' else 'NOT NB')
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
	tag_list = ['NOUN', 'PROPN', 'VERB', 'AUX', 'NUM', 'PART']

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
		elif token.pos_ == "INTJ":
			new_doc.append(token.text)
		elif token.text in emoji.UNICODE_EMOJI:
			new_doc.append(token.text)
		elif token.pos_ in tag_list:
			new_doc.append(token.pos_)
		else:
			new_doc.append(token.text)
	
	return ' '.join(new_doc)


def features(df, nlp, prof_lst, abbrev_lst):
	"""Add several columns with possible features"""
	df['length'] = df['posts'].apply(len)
	df['comb_posts'] = df['posts'].apply(lambda x: ' '.join(x))
	df['no_rep'] = df['no_rep'].apply(lambda x: ' '.join(x))

	# Length
	df['char_length_tot'] = df['comb_posts'].apply(len)
	
	# Punctuation
	punct = '!_@}+\-~{;*./`?,:\])\\#[=\"&%\'(^|$—“”’—...'
	
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
	print("----------REDDIT NON-BINARY----------")
	baseline(data)
	labels = ['NB', 'NOT NB']

	# Classifier
	pipeline = Pipeline([
		('union', ColumnTransformer([
			('vecword', TfidfVectorizer(ngram_range = (1,2), analyzer='word', token_pattern=r'\S+'), 'tagged'),
			('charlentot', MinMaxScaler(), ['char_length_tot']),
			], remainder='drop')),
		('clf', LogisticRegression(max_iter=100000, C=1.0))])
	model = pipeline.fit(data[0], data[0]['label'])

	# Coefficient with feature names (most important features)
	feature_names = model['union'].transformers_[0][1].get_feature_names() + ['charlentot']
	top_features = pd.DataFrame(-(model.named_steps['clf'].coef_[0]), index=feature_names, columns=['coef'])
	top_features = top_features.reindex(top_features.coef.abs().sort_values(ascending=False).index)

	# Validation
	pred_val = model.predict(data[3])

	# Reddit
	pred = model.predict(data[1])
	cm = pd.crosstab(pd.Series(data[1]['label'], name='Actual'), pd.Series(pred, name='Predicted'))
	r_results = data[1][['username', 'gender', 'label']].copy()
	r_results['prediction'] = pred
	r_results.to_csv("../Results/r_nb.csv", index=False)

	# Twitter
	pred_tw = model.predict(data[2])
	cm_tw = pd.crosstab(pd.Series(data[2]['label'], name='Actual'), pd.Series(pred_tw, name='Predicted'))
	tw_results = data[2][['username', 'gender', 'label']].copy()
	tw_results['prediction'] = pred_tw
	tw_results.to_csv("../Results/tw_r_nb.csv", index=False)

	# Print results
	print("\n----------TOP FEATURES----------")
	print(top_features.head(20))
	print("--------------------------------")

	print("\n\n----------LOGISTIC REGRESSION-----------\n")
	print("----------VALIDATION----------")
	print("Accuracy score: {}".format(accuracy_score(data[3]['label'], pred_val)))
	print("---------------------------\n")

	print("----------REDDIT----------")
	print("Accuracy score: {}\n".format(accuracy_score(data[1]['label'], pred)))
	print("Classification report:")
	print(classification_report(data[1]['label'], pred))
	print("\nConfusion matrix: \n{}".format(cm))
	print("\n\nWrong prediction sample:")
	print(r_results[r_results.label != r_results.prediction].sample(5, random_state=1))
	print("--------------------------")

	print("\n----------TWITTER----------")
	print("Accuracy score: {}\n".format(accuracy_score(data[2]['label'], pred_tw)))
	print("Classification report:")
	print(classification_report(data[2]['label'], pred_tw))
	print("\nConfusion matrix: \n{}".format(cm_tw))
	print("\n\nWrong prediction sample:")
	print(tw_results[tw_results.label != tw_results.prediction].sample(5, random_state=1))
	print("---------------------------")
	print("--------------------------------------")


if __name__ == '__main__':
	main()
